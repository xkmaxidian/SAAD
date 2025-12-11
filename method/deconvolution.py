import pandas as pd
from .train import train_model, predict, reproducibility
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scanpy as sc
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree as _KDTree
from tqdm import tqdm
from numpy.random import choice
import time

from .utils import plot_spot_piecharts


def _knn_gaussian_weights(coords, k=6, bandwidth=None):
    """
    输入:
      coords: (N,2) float，每个 spot 的 (x,y)
      k: 每个点取的近邻数（含自身）
      bandwidth: 高斯核带宽（与坐标单位一致）；None 则自动估计

    返回:
      idx: (N,k) int，近邻索引（含自身）
      w:   (N,k) float，行归一化后的高斯核权重
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]
    k = min(k, N)

    if _KDTree is not None:
        tree = _KDTree(coords)
        dists, idx = tree.query(coords, k=k)  # 含自身，距离第一个为0
    else:
        # 纯 numpy（O(N^2)）：N≲2000 时可接受
        d2 = np.sum(coords**2, axis=1, keepdims=True)
        dist2 = d2 + d2.T - 2.0 * (coords @ coords.T)
        dist2 = np.maximum(dist2, 0.0)
        dists_full = np.sqrt(dist2, dtype=np.float64)
        idx = np.argpartition(dists_full, kth=np.arange(k), axis=1)[:, :k]
        row = np.arange(N)[:, None]
        dists = dists_full[row, idx]
        order = np.argsort(dists, axis=1)
        idx = idx[row, order]
        dists = dists[row, order]

    # 自动估带宽（忽略自身0距离）
    if bandwidth is None:
        if k > 1:
            bw_est = np.median(dists[:, 1:].reshape(-1))
            bandwidth = max(bw_est, 1e-6)
        else:
            bandwidth = 1.0

    # 高斯核 + 行归一化
    w = np.exp(-(dists ** 2) / (2.0 * (bandwidth ** 2)))
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.maximum(w_sum, 1e-12)
    return idx.astype(int), w.astype(np.float64)

def generate_simulated_data(sc_data, outname=None,
                            d_prior=None, rangePick=False,
                            n=10, samplenum=5000, min_cells_per_spot=6, max_cells_per_spot=10,
                            random_state=None, sparse=True, sparse_prob=0.5,
                            rare=False, rare_percentage=0.4
                            ,
                            coords=None,  # 空间坐标
                            spatial_strength=0.6,  # [0,1]，平滑强度（0=不平滑）
                            knn_k=6,  # kNN 邻居数
                            bandwidth=None  # 高斯核带宽
                            ):
    """
    sc_data: 单细胞数据， cell * gene
    d_prior: celltype的比列先验, 基于此在进行Dirichlet采样
    n: 每个spot所含细胞数的上限
    samplenum: 生成的spot数
    sparse: 允许大多数celltype含量为0
    rare: 允许很多celltype含量接近0
    random_state: 随机种子
    """


    print('select top 5000 hvg genes to generate test ST')
    sc_adata = anndata.AnnData(sc_data)
    # print("filter sc min_cells=3, max_genes=5000")
    # sc.pp.filter_cells(sc_adata, max_genes=5000)
    # sc.pp.filter_genes(sc_adata, min_cells=3)
    sc.pp.highly_variable_genes(sc_adata, flavor="seurat_v3", n_top_genes=5000)
    sc_adata = sc_adata[:, sc_adata.var['highly_variable']]


    # 单独处理单细胞细胞得到标准处理流后,拿到celltype的均值表达mean_S
    print("get celltype mean expression data mean_S")
    tmp_adata = sc_adata.copy()
    sc.pp.normalize_total(tmp_adata, target_sum=1e4)  # normalize模拟空转 (真实空转目前未norm)
    sc.pp.log1p(tmp_adata)
    sc.pp.scale(tmp_adata, zero_center=False, max_value=10)
    standard_procees_mean_S = tmp_adata.to_df().groupby(sc_data.index).mean()

    # sc_data should be a cell*gene matrix, no null value, txt file, sep='\t'
    # index should be cell names
    # columns should be gene labels
    sc_data = sc_adata.to_df()
    sc_data.dropna(inplace=True)
    sc_data['celltype'] = sc_data.index
    sc_data.index = range(len(sc_data))
    print('Reading dataset is done')

    num_celltype = len(sc_data['celltype'].value_counts())
    genename = sc_data.columns[:-1]

    celltype_groups = sc_data.groupby('celltype').groups
    sc_data.drop(columns='celltype', inplace=True)

    # use ndarray to accelerate
    # change to C_CONTIGUOUS, 10x faster
    sc_data = np.ascontiguousarray(sc_data, dtype=np.float32)

    # make random cell proportions
    if random_state is not None and isinstance(random_state, int):
        print('generate simualted ST random_state is', random_state)

    if d_prior is None:
        print('Generating cell fractions using Dirichlet distribution without prior info (actually random)')
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(np.ones(num_celltype), samplenum) # cellType * spot
        print('RANDOM cell fractions is generated')
    elif d_prior is not None:
        print('Using prior info to generate cell fractions in Dirichlet distribution')
        assert len(d_prior) == num_celltype, 'dirichlet prior is a vector, its length should equals ' \
                                             'to the number of cell types'
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(d_prior, samplenum)
        print('Dirichlet cell fractions is generated')

    # make the dictionary
    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)

    prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    # sparse cell fractions
    if sparse:
        print("the sparse probability is", sparse_prob)
        ## Only partial simulated data is composed of sparse celltype distribution
        for i in range(int(prop.shape[0] * sparse_prob)): # sparse_prob比例的spot
            # sparse_prob比例的celltype列选中置0
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    if rare:
        ## choose celltype
        if isinstance(random_state, int):
            np.random.seed(random_state)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf


    """ 空间平滑，使空间相邻 spot 的组成更相似 === """
    W_full = None
    if coords is not None:
        # 饼图绘制
        plot_spot_piecharts(coords, prop, title="Before spatial smoothing")

        coords = np.asarray(coords, dtype=float)
        assert coords.shape == (samplenum, 2)
        spatial_strength = float(spatial_strength)
        if spatial_strength > 0:
            print(
                f"Applying spatial smoothing: knn_k={knn_k}, bandwidth={'auto' if bandwidth is None else bandwidth}, spatial_strength={spatial_strength}")
            idx, w = _knn_gaussian_weights(coords, k=min(int(knn_k), samplenum), bandwidth=bandwidth)
            P_smooth = np.zeros_like(prop, dtype=np.float64)
            for i in range(samplenum):
                P_smooth[i] = (w[i][:, None] * prop[idx[i]]).sum(axis=0)
            prop = (1.0 - spatial_strength) * prop + spatial_strength * P_smooth
            prop = prop / np.sum(prop, axis=1, keepdims=True).clip(min=1e-12)
            print("prop nan count:", np.isnan(prop).sum())
            print('construst KNN graph W')
            N, k = idx.shape
            W_full = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                W_full[i, idx[i]] = w[i]
            # 对角为0, 行归一化
            np.fill_diagonal(W_full, 0)
            row_sums = W_full.sum(axis=1, keepdims=True).clip(min=1e-12)
            W_full = W_full / row_sums
            #print("W 对角元素（self-loop 权重）：", np.diag(W_full)[:10])
            #print("W 行和（应接近1）：",  W_full.sum(axis=1))
            # 可视化
            W_vis = W_full
            plt.figure(figsize=(6, 6))
            plt.imshow(W_vis, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='edge weight')
            plt.title(f'Adjacency matrix heatmap (n={N}, k={knn_k})')
            plt.xlabel('node index');
            plt.ylabel('node index')
            plt.show()

            # 饼图绘制
            plot_spot_piecharts(coords, prop, title="After spatial smoothing")
    else:
        print("skip spatial smoothing.")

    if rangePick:
        print(
            f"Assigning random cell numbers per spot in [{min_cells_per_spot}, {max_cells_per_spot}] using multinomial sampling")
        cell_num = np.zeros_like(prop, dtype=int)
        for i in range(prop.shape[0]):
            n_i = np.random.randint(min_cells_per_spot, max_cells_per_spot + 1)  # 每个spot总细胞数随机
            p = prop[i].copy()
            if np.allclose(p, 0):
                p = np.ones_like(p) / len(p)
            else:
                p = p / p.sum()
            cell_num[i] = np.random.multinomial(n_i, p)
    else:
        # precise number for each celltype
        cell_num = np.floor(n * prop)

    # precise proportion based on cell_num
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)

    # start sampling
    sample = np.zeros((prop.shape[0], sc_data.shape[1]))
    allcellname = celltype_groups.keys()
    print('Sampling cells to compose pseudo-bulk data')
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data[select_index].sum(axis=0)

    prop = pd.DataFrame(prop, columns=celltype_groups.keys())
    simudata = anndata.AnnData(X=sample, # spot * gene
                               obs=prop, # spot * cellType
                               var=pd.DataFrame(index=genename)) # var: gene

    print('Sampling is done')
    if outname is not None:
        simudata.write_h5ad(outname + '.h5ad')
    return simudata, W_full, standard_procees_mean_S

def ProcessInputData(train_data, test_data,
                     variance_threshold=0.98, scaler="mms"):
    # train_data: 模拟ST, Anndata
    # test_data: 真实ST, DataFrame

    ### read train data
    print('Reading train data')
    if type(train_data) is anndata.AnnData:
        pass
    elif type(train_data) is str:
        train_data = anndata.read_h5ad(train_data)

    print('train_data normalize_total -> log1p -> scale')
    sc.pp.normalize_total(train_data, target_sum=1e4) # normalize模拟空转 (真实空转目前未norm)
    sc.pp.log1p(train_data)
    sc.pp.scale(train_data, zero_center=False, max_value=10) # zero_center: 基因归一化, 方差1, 是否减去均值使得整体均值为0

    train_x = pd.DataFrame(train_data.X, columns=train_data.var.index) # spot*gene
    train_y = train_data.obs # spot*cellType
    print('Reading train data is done, shape: ', train_x.shape)

    ### read test data
    print('Reading test data')
    if type(test_data) is pd.DataFrame:
        test_x = test_data
    else:
        raise Exception("sp data should be DataFrame!")

    ### find intersected genes
    #  这一步必须在test_data norm处理前(因为模拟空转是基于单细胞的top 5000基因生成的, 尺度已经缩小, 所以test提前取到同尺度的基因)
    print('Finding intersected genes...')
    inter = train_x.columns.intersection(test_x.columns)
    train_x = train_x[inter]
    test_x = test_x[inter]

    print('test_data normalize_total -> log1p -> scale')
    test_data_adata = sc.AnnData(X=test_x)
    sc.pp.normalize_total(test_data_adata, target_sum=1e4)
    sc.pp.log1p(test_data_adata)
    sc.pp.scale(test_data_adata, zero_center=False, max_value=10)
    test_x = test_data_adata.to_df()

    print('Reading test data is done, shape: ',  test_x.shape)

    ### variance cutoff
    if variance_threshold < 1:
        print('Cutting variance threshold ', variance_threshold)
        var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_threshold)]
        train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]
        var_cutoff = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_threshold)]
        test_x = test_x.loc[:, test_x.var(axis=0) > var_cutoff]
        print('Finding intersected genes after cutting variance')
        inter = train_x.columns.intersection(test_x.columns)
        train_x = train_x[inter]
        test_x = test_x[inter]

    genename = list(inter) # gene
    celltypes = train_y.columns # celltypes
    samplename = test_x.index # spot

    print('Intersected gene number is ', len(inter))

    ### MinMax process
    colors = sns.color_palette('RdYlBu', 10)
    fig = plt.figure()
    sns.histplot(data=np.mean(train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
    sns.histplot(data=np.mean(test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
    plt.legend(title='datatype', labels=['trainingdata', 'testdata'])
    plt.title('pure data')
    plt.show()

    ### 归一化处理
    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x.T).T
        ss_test_x = ss.fit_transform(test_x.T).T
        sns.histplot(data=np.mean(ss_train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
        sns.histplot(data=np.mean(ss_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['trainingdata', 'testdata'])
        plt.title('standard after data')
        plt.show()

        return ss_train_x, train_y.values, ss_test_x, genename, celltypes, samplename

    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x.T).T
        mms_test_x = mms.fit_transform(test_x.T).T
        sns.histplot(data=np.mean(mms_train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
        sns.histplot(data=np.mean(mms_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['trainingdata', 'testdata'])
        plt.title('minmax after data')
        plt.show()


        return mms_train_x, train_y.values, mms_test_x, genename, celltypes, samplename
    else:
        return train_x.values, train_y.values, test_x.values, genename, celltypes, samplename

def Deconvolution(necessary_data, real_bulk, coords=None,
                  samplenum=1000, rangePick=False,
                  d_prior=None, sparse=True, sparse_prob=0.8, spatial_strength=1, random_state=1,
                  variance_threshold=0.98, scaler='ss',
                  mode='real', adaptive=True, step=300, max_iter=5, save_model_name=None,
                  alpha=0.1, batch_size=32, epochs=256, seed=1):

    if type(necessary_data) is pd.DataFrame:
        # 基于高斯核函数得到的图W, 既用于平滑模拟数据, 又用于训练时作为空间约束
        simudata, W, mean_S = generate_simulated_data(sc_data=necessary_data,
                                           samplenum=samplenum, rangePick=rangePick,
                                           d_prior=d_prior,sparse=sparse, sparse_prob=sparse_prob, random_state=random_state,
                                           coords=coords, spatial_strength=spatial_strength)
    else:
        raise Exception('Please give the correct input')

    train_x, train_y, test_x, genename, celltypes, samplename = \
        ProcessInputData(simudata, real_bulk,
                         variance_threshold=variance_threshold, scaler=scaler)

    # mean_S进一步保留交集基因
    common_genes = mean_S.columns.intersection(genename)
    mean_S = mean_S[common_genes].values
    print("mean celltype expre mean_S, ", mean_S.shape)

    print('training data shape is ', train_x.shape, '\ntest data shape is ', test_x.shape)
    print('random seed is ', random_state, '\nspatail regulation param alpha is ', alpha)

    start_time = time.time()  # 记录开始时间
    if save_model_name is not None:
        reproducibility(seed)
        model = train_model(train_x, train_y, save_model_name, W, alpha, mean_S, batch_size=batch_size, epochs=epochs, seed=random_state)
    else:
        reproducibility(seed)
        model = train_model(train_x, train_y, W, alpha, mean_S, batch_size=batch_size, epochs=epochs, seed=random_state)
    print('Notice that you are using parameters: mode=' + str(mode) + ' and adaptive=' + str(adaptive))
    if adaptive is True:
        Sigm, Pred = \
            predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                    model=model, model_name=save_model_name,
                    adaptive=adaptive, mode=mode, step=step, max_iter=max_iter, seed=seed)
        elapsed_time = time.time() - start_time
        return Sigm, Pred, elapsed_time, mean_S
    else:
        Sigm, Pred = predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                       model=model, model_name=save_model_name,
                       adaptive=adaptive, mode=mode, step=step, max_iter=max_iter, seed=seed)
        elapsed_time = time.time() - start_time
        return Sigm, Pred, elapsed_time, mean_S


