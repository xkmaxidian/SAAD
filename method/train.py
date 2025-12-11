import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import simdatset, AutoEncoder, device
from .utils import showloss


def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def training_stage(model, train_loader, W, alpha, mean_S, optimizer, epochs=128):

    model.train()
    model.state = 'train'
    loss = [] # 记录celltype含量的和ground_truth的损失
    recon_loss = [] # 记录重建损失
    smooth_loss = [] # 记录空间位置约束
    S_loss = [] # 记录均值celltype约束

    if W is not None:
        W = torch.tensor(W, dtype=torch.float32, device=device)
    mean_S = torch.tensor(mean_S, dtype=torch.float32, device=device)


    for i in tqdm(range(epochs)):
        for k, (data, label, idx) in enumerate(train_loader):
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)

            batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data) # ground_truth损失 + 重建损失

            # TODO: 空间网络约束
            # GMI
            # sub_W = W[idx][:, idx] # 当前batch的子图
            # pre_W = torch.matmul(cell_prop, cell_prop.t())
            # spatial_loss = torch.nn.BCEWithLogitsLoss()(pre_W, sub_W)
            # batch_loss += alpha * spatial_loss

            # 局部保距
            if W is not None:
                sub_W = W[idx][:, idx]  # 当前batch的子图
                D = torch.diag(sub_W.sum(dim=1))
                L = D - sub_W
                spatial_loss = torch.trace(cell_prop.T @ L @ cell_prop)
                batch_loss += alpha * spatial_loss

            # KL散度约束 min Wij * KL(Pi | Pj) -> Wij类似指示矩阵: 只约束邻居 分布相似
            # sub_W = W[idx][:, idx]  # 当前batch的子图
            # p = F.softmax(cell_prop, dim=1)
            # log_p = torch.log(p)
            # kl_ij = (p.unsqueeze(1) * (log_p.unsqueeze(1) - log_p.unsqueeze(0))).sum(dim=2)
            # kl_ji = (p.unsqueeze(0) * (log_p.unsqueeze(0) - log_p.unsqueeze(1))).sum(dim=2)
            # kl_matrix = 0.5 * (kl_ij + kl_ji)
            # spatial_loss = (sub_W * kl_matrix).sum() / sub_W.sum()
            # batch_loss += alpha * spatial_loss

            # TODO: 均值S约束
            # batch_loss += F.l1_loss(mean_S, sigm)

            # TODO: 特征稀疏约束
            # L21: torch.sum(torch.sqrt(torch.sum(cell_prop ** 2, dim=1) + 1e-8))
            # L1: torch.mean(torch.abs(cell_prop))
            # sparse AE: KL(p|p^)
            # rho = 0.1
            # rho_hat = torch.mean(cell_prop, dim=0)
            # sparse_item = torch.sum(
            #     rho * torch.log(rho / (rho_hat + 1e-10)) +
            #     (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
            # )
            # batch_loss += sparse_item


            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).detach().cpu().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).detach().cpu().detach().numpy())
            if W is not None:
                smooth_loss.append(alpha * spatial_loss.detach().cpu().numpy())
            S_loss.append(F.l1_loss(mean_S, sigm).detach().cpu().numpy())
    return model, loss, recon_loss, smooth_loss, S_loss

def adaptive_stage_v1(model, data, optimizerD, optimizerE, step=300, max_iter=5):
    data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, ori_pred, ori_sigm = model(data)
    ori_sigm = ori_sigm.detach() # 预测阶段: 输入数据拿到S0: cellType * gene
    ori_pred = ori_pred.detach() # 预测阶段: 输入数据拿到嵌入z: spot*64 -> test归一化后得到 spot*cellType
    model.state = 'train'

    for k in range(max_iter):
        model.train()
        for i in range(step): # step1: 固定S0训练解码器
            optimizerD.zero_grad()
            x_recon, pred, sigm = model(data)
            batch_loss = F.l1_loss(x_recon, data) + F.l1_loss(sigm, ori_sigm) # 重建损失 + 初始的S0和后续S的损失

            batch_loss.backward()
            optimizerD.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

        for i in range(step): # step2: 固定X0训练编码器
            optimizerE.zero_grad()
            x_recon, pred, sigm = model(data)
            batch_loss = F.l1_loss(x_recon, data) + F.l1_loss(ori_pred, pred)

            batch_loss.backward()
            optimizerE.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())


    model.eval()
    model.state = 'test'
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()

def adaptive_stage_v2(model, data, optimizerD, optimizerE, step=10, max_iter=5):
    data = torch.from_numpy(data).float().to(device)
    loss = []

    for k in range(max_iter):
        model.train()
        for i in range(step):
            optimizerE.zero_grad()
            optimizerD.zero_grad()
            x_recon, pred, sigm = model(data)
            batch_loss = F.l1_loss(x_recon, data)

            batch_loss.backward()
            optimizerE.step()
            optimizerD.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    model.eval()
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()

def train_model(train_x, train_y, W, alpha, mean_S,
                model_name=None,
                batch_size=128, epochs=128, seed=1):
    
    train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = AutoEncoder(train_x.shape[1], train_y.shape[1]).to(device)
    reproducibility(seed=seed)
    optimizer = Adam(model.parameters(), lr=1e-4)
    print('Start training')
    # loss: cellType丰度标签和预测的损失, reconloss: 重建损失
    model, loss, reconloss, smooth_loss, mean_S_loss = training_stage(model, train_loader, W, alpha, mean_S, optimizer, epochs=epochs)

    print('Training is done')
    #print('prediction label loss:')
    showloss(loss, 'label')
    #print('reconstruction loss:')
    showloss(reconloss, 'reconstruction')
    #print('spatial loss:')
    showloss(smooth_loss, 'smooth loss')
    #print('mean_S loss:')
    showloss(mean_S_loss, 'mean_S loss')
    if model_name is not None:
        print('Model is saved')
        torch.save(model, model_name+".pth")
    return model

def predict(test_x, genename, celltypes, samplename,
            model_name=None, model=None,
            adaptive=True, mode='real', seed=1, step=300, max_iter=5):
    
    if model is not None and model_name is None:
        print('Model is saved without defined name')
        torch.save(model, 'model.pth')

    reproducibility(seed=seed)

    if adaptive is True:
        if model_name is not None and model is None:
            model = torch.load(model_name + ".pth")
        elif model is not None and model_name is None:
            model = torch.load("model.pth")
        decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
        encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
        optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
        optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
        print('Start adaptive training for all the samples, step:', step, '  max_iter:', max_iter)
        if mode=='simulated':
            test_sigm, loss, test_pred = adaptive_stage_v1(model, test_x, optimizerD, optimizerE, step=step, max_iter=max_iter)
        else:
            test_sigm, loss, test_pred = adaptive_stage_v2(model, test_x, optimizerD, optimizerE, step=step, max_iter=max_iter)
        #print('adaptive reconstruct loss:')
        showloss(loss, 'adaptive reconstruct loss')
        print('Adaptive stage is done')
        test_sigm = pd.DataFrame(test_sigm,columns=genename,index=celltypes)
        test_pred = pd.DataFrame(test_pred,columns=celltypes,index=samplename)

        return test_sigm, test_pred

    else:
        if model_name is not None and model is None:
            model = torch.load(model_name+".pth")
        elif model is not None and model_name is None:
            model = model
        print('Predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.from_numpy(test_x).float().to(device)
        _, pred, sigm = model(data)
        pred = pred.cpu().detach().numpy()
        pred = pd.DataFrame(pred, columns=celltypes, index=samplename)
        sigm = sigm.cpu().detach().numpy()
        sigm = pd.DataFrame(sigm, columns=genename, index=celltypes)
        print('Prediction is done')
        return sigm, pred




