
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Wedge

from sklearn.neighbors import NearestNeighbors

from scipy import stats
import scipy.stats as st

def plot_spot_piecharts(coords, prop, title="Spot composition", show_num=200, radius=0.5):
    """
    在空间坐标上绘制每个 spot 的细胞类型组成饼图。
    - coords: (n,2) numpy 数组，spot 坐标
    - prop: (n,k) numpy 数组，cell type 含量，行归一化
    - show_num: 只绘制前多少个 spot
    - radius: 每个饼图的半径
    """
    coords = np.asarray(coords)
    prop = np.asarray(prop)
    n, k = prop.shape
    show_num = min(show_num, n)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

    # 随机色板
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i in range(show_num):
        x, y = coords[i]
        sizes = prop[i]
        start_angle = 0
        for j, frac in enumerate(sizes):
            if frac <= 0:
                continue
            theta1 = start_angle
            theta2 = start_angle + frac * 360
            wedge = Wedge(center=(x, y), r=radius, theta1=theta1, theta2=theta2,
                          facecolor=colors[j], edgecolor='white', lw=0.5)
            ax.add_patch(wedge)
            start_angle = theta2

    ax.set_xlim(coords[:, 0].min() - radius, coords[:, 0].max() + radius)
    ax.set_ylim(coords[:, 1].min() - radius, coords[:, 1].max() + radius)
    plt.tight_layout()
    plt.show()

def showloss(loss, ylabel):
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel(ylabel)
    plt.show()

### myPreprocess
# -I(W, A)
def gmi_loss(pre, gnd):
    nodes_n = gnd.shape[0]
    edges_n = torch.sum(gnd) / 2
    weight1 = (nodes_n * nodes_n - edges_n) / edges_n
    weight2 = nodes_n * nodes_n / (nodes_n * nodes_n - edges_n)
    temp1 = gnd * torch.log(pre + 1e-10) * (-weight1)
    temp2 = (1 - gnd) * torch.log(1 - pre + 1e-10)
    return torch.mean(temp1 - temp2) * weight2

def conctruct_KNN(position, n_neighbors=6, show=True):
    if isinstance(position, pd.DataFrame):
        position = position.values
    n_spot = position.shape[0]
    print("spot num ", n_spot)

    nbrs = NearestNeighbors(algorithm='ball_tree').fit(position)
    graph_out = nbrs.kneighbors_graph(n_neighbors=n_neighbors,
                                      mode="distance")
    graph_out.data = 1 / (graph_out.data + 1e-6)
    # row_normalize
    for start_ptr, end_ptr in zip(graph_out.indptr[:-1], graph_out.indptr[1:]):
        row_sum = graph_out.data[start_ptr:end_ptr].sum()
        if row_sum != 0:
            graph_out.data[start_ptr:end_ptr] /= row_sum
    if show:
        W_vis = graph_out.toarray()
        plt.figure(figsize=(6, 6))
        plt.imshow(W_vis, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='edge weight')
        plt.title(f'Adjacency matrix heatmap (n={n_spot}, k={n_neighbors})')
        plt.xlabel('node index');
        plt.ylabel('node index')
        plt.show()
    return graph_out.toarray()

def scale_max(df):
    """
        Divided by maximum value to scale the data between [0,1].
        Please note that these datafrmae are scaled data by column.

        Parameters
        -------
        df: dataframe, each col is a feature.

    """

    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result

def scale_z_score(df):
    """
        scale the data by Z-score to conform the data to the standard normal distribution, that is, the mean value is 0, the standard deviation is 1, and the conversion function is 0.
        Please note that these datafrmae are scaled data by column.

        Parameters
        -------
        df: dataframe, each col is a feature.
    """

    result = pd.DataFrame()
    for label, content in df.items():
        content = stats.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result

def scale_plus(df):
    """
        Divided by the sum of the data to scale the data between (0,1), and the sum of data is 1.
        Please note that these datafrmae are scaled data by column.

        Parameters
        -------
        df: dataframe, each col is a feature.
    """

    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result

def cal_ssim(im1, im2, M):
    """
        calculate the SSIM value between two arrays.

    Parameters
        -------
        im1: array1, shape dimension = 2
        im2: array2, shape dimension = 2
        M: the max value in [im1, im2]
    """

    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim

def ssim(raw, impute, scale='scale_max'):
    ## This was used for calculating the SSIM value between two arrays.

    if scale == 'scale_max':
        raw = scale_max(raw)
        impute = scale_max(impute)
    else:
        print('Please note you do not scale data by max')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]

            M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
            raw_col_2 = np.array(raw_col)
            raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)

            impute_col_2 = np.array(impute_col)
            impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)

            ssim = cal_ssim(raw_col_2, impute_col_2, M)

            ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
            result = pd.concat([result, ssim_df], axis=1)
        return result
    else:
        print("columns error")

def pearsonr(raw, impute, scale=None):
    ## This was used for calculating the PCC between two arrays.
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]
            pearsonr, _ = st.pearsonr(raw_col, impute_col)
            pearson_df = pd.DataFrame(pearsonr, index=["Pearson"], columns=[label])
            result = pd.concat([result, pearson_df], axis=1)
        return result

def JS(raw, impute, scale='scale_plus'):
    ## This was used for calculating the JS value between two arrays.

    if scale == 'scale_plus':
        raw = scale_plus(raw)
        impute = scale_plus(impute)
    else:
        print('Please note you do not scale data by plus')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            eps = 1e-12
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]
            raw_col = raw_col.values + eps
            impute_col = impute_col.values + eps

            M = (raw_col + impute_col) / 2
            KL = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
            KL_df = pd.DataFrame(KL, index=["JS"], columns=[label])

            result = pd.concat([result, KL_df], axis=1)
        return result


def RMSE(raw, impute, scale='zscore'): # 'zscore'
    ## This was used for calculating the RMSE value between two arrays.

    if scale == 'zscore':
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)
    else:
        print('Please note you do not scale data by zscore')
    if raw.shape[1] == impute.shape[1]:
        result = pd.DataFrame()
        for label in raw.columns:
            raw_col = raw.loc[:, label]
            impute_col = impute.loc[:, label]
            RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())
            RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])

            result = pd.concat([result, RMSE_df], axis=1)
        return result

def Moran_I(genes_exp,x, y, k=5, knn=True):
    XYmap=pd.DataFrame({"x": x, "y":y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto',metric = 'euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0],genes_exp.shape[0]))
        for i in range(0,genes_exp.shape[0]):
            W[i,XYindices[i,:]]=1
        for i in range(0,genes_exp.shape[0]):
            W[i,i]=0

    I = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X_minus_mean = np.array(genes_exp[k] - np.mean(genes_exp[k]))
        X_minus_mean = np.reshape(X_minus_mean,(len(X_minus_mean),1))
        Nom = np.sum(np.multiply(W,np.matmul(X_minus_mean,X_minus_mean.T)))
        Den = np.sum(np.multiply(X_minus_mean,X_minus_mean))
        I[k] = (len(genes_exp[k])/np.sum(W))*(Nom/Den)
    return I

def Geary_C(genes_exp,x, y, k=5, knn=True):
    XYmap=pd.DataFrame({"x": x, "y":y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto',metric = 'euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0],genes_exp.shape[0]))
        for i in range(0,genes_exp.shape[0]):
            W[i,XYindices[i,:]]=1
        for i in range(0,genes_exp.shape[0]):
            W[i,i]=0

    C = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X=np.array(genes_exp[k])
        X_minus_mean = X - np.mean(X)
        X_minus_mean = np.reshape(X_minus_mean,(len(X_minus_mean),1))
        Xij=np.array([X,]*X.shape[0]).transpose()-np.array([X,]*X.shape[0])
        Nom = np.sum(np.multiply(W,np.multiply(Xij,Xij)))
        Den = np.sum(np.multiply(X_minus_mean,X_minus_mean))
        C[k] = (len(genes_exp[k])/(2*np.sum(W)))*(Nom/Den)
    return C

def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]
