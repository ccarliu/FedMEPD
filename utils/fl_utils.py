import collections
import torch
import torchvision
import numpy as np
import random
import copy
import faiss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.clustering import run_kmeans

import torch.nn.functional as F

def get_len(t):
    return torch.norm(t,2,1)

def get_client_weights4(exp_avg_s, exp_avg_c, weight_per, count_per, alpha = 0., client_idx = 0):

    for index, cweight in enumerate(weight_per):

        ccount = count_per[index]

        if not index < 172:
            weight_per[index] = torch.zeros_like(cweight)
            count_per[index] = torch.zeros_like(cweight)
            continue

        if index % 2:
            continue
        
        # the shape: filters, channel, s, s, s
        filters, channel, s,s,s = exp_avg_s[index].shape

        data_num_list = [21, 22, 20, 22, 21, 22, 34, 35]
        weight_l2 = (get_len(exp_avg_s[index].view(filters, -1)) / get_len(exp_avg_c[index].view(filters, -1))).view(filters, 1,1,1,1) * (data_num_list[client_idx] / np.sum(data_num_list))  # * (data_num_list[client_idx] / np.sum(data_num_list))
        weight_f = torch.cosine_similarity(exp_avg_s[index].view(filters, -1), exp_avg_c[index].view(filters, -1)).view(filters, 1,1,1,1)
        weight_f = (weight_f + 1) / 2
        ccount[torch.logical_and(ccount > 0, weight_f < 0.5)] -= 1
        ccount[torch.logical_and(ccount > 0, weight_f > 0.5)] = patience

        count_per[index] = ccount
        count_per[index+1] = count_per[index].squeeze().clone()

        cweight = cweight * 0.95 + weight_l2 * 0.05
        
        cweight[ccount <= 0] = 0
        weight_per[index] = cweight
        weight_per[index + 1] = weight_per[index].squeeze().clone()

    return weight_per, count_per

def avg_EW(ws, ms):
    m = torch.stack(ms, dim=0)
    count = int((m==True).sum())
    weight_keys = ws[0].keys()
    avg_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = None
        for c in range(len(ws)):
            if key_sum is None:
                key_sum = ms[c] * ws[c][key].data.cpu()
            else:
                key_sum += ms[c] * ws[c][key].data.cpu()

        avg_state_dict[key] = key_sum / count
    return avg_state_dict

def uploadLCweightsandGLBupdate(masks, server_model, local_weights, train_loader, protos, round, client_weights_per, optimizer):
    
    model_old = copy.deepcopy(server_model)

    lc1_mask, lc2_mask, lc3_mask, lc4_mask = masks[0], masks[1], masks[2], masks[3]
    modals_E = []
    for i in range(len(lc1_mask)):  # 针对每一个模态的特异编码器
        c_E = avg_EW([local_weights[c][i] for c in range(len(local_weights))], [masks[c][i] for c in range(len(local_weights))])
        modals_E.append(c_E)

    c_D, clients_weights = avg_EW_per_4([local_weights[i][-1] for i in range(len(local_weights))], client_weights_per)

    c_D = avg_EW_per_4_(c_D, server_model.decoder_fuse.state_dict(), clients_weights, retain_ratio = 0.3)

    server_model.c1_encoder.load_state_dict(modals_E[0])
    server_model.c2_encoder.load_state_dict(modals_E[1])
    server_model.c3_encoder.load_state_dict(modals_E[2])
    server_model.c4_encoder.load_state_dict(modals_E[3])
    server_model.decoder_fuse.load_state_dict(c_D)
    
    ### global training
    glb_w, glb_protos, glb_Pnames, exp_avg = global_training(args, args.device, glb_trainloader, server_model, None, round, optimizer)   

    exp_avg = model_diff(model_old, server_model)

    return glb_w, glb_protos, glb_Pnames, client_weights_per, exp_avg

def downloadGLBweights(glb_w, model_clients, client_weights_per):
    c1_w = {k.replace('c1_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c1_encoder' in k}
    c2_w = {k.replace('c2_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c2_encoder' in k}
    c3_w = {k.replace('c3_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c3_encoder' in k}
    c4_w = {k.replace('c4_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c4_encoder' in k}
    df_w = {k.replace('decoder_fuse.',''): v.cpu() for k,v in glb_w.items() if 'decoder_fuse' in k} 


    for i in range(len(model_clients)): # 简单起见，直接全部加载，但训练时其实只用到了存在模态编码器
        model_clients[i].c1_encoder.load_state_dict(c1_w)    # flair模态 
        model_clients[i].c2_encoder.load_state_dict(c2_w)    # t1ce模态
        model_clients[i].c3_encoder.load_state_dict(c3_w)    # t1模态
        model_clients[i].c4_encoder.load_state_dict(c4_w)    # t2模态
        model_clients[i].decoder_fuse.load_state_dict(avg_EW_per_4_(df_w, model_clients[i].decoder_fuse.state_dict(), client_weights_per[i]))


def avg_EW_per_4(ws, per):
    weight_keys = ws[0].keys()
    avg_state_dict = collections.OrderedDict()
    clients_weight = []
    for idx, key in enumerate(weight_keys):
        tweight = None
        key_sum =None
        for i in range(len(ws)): # 8 is the client num
            cweight = per[i][idx].clone()
            for l in range(len(ws[i][key].shape) - len(cweight.shape)):
                cweight = cweight.unsqueeze(-1)

            if tweight is None:
                tweight = cweight
                key_sum = cweight * ws[i][key].data.cpu()
            else:
                tweight += cweight
                key_sum += cweight * ws[i][key].data.cpu()

        fweight = tweight.clone()
        fweight[fweight == 0] = 1

        avg_state_dict[key] = key_sum / fweight

        clients_weight.append(tweight)

    return avg_state_dict, clients_weight

def avg_EW_per_4_(w1, w2, per, retain_ratio = 0): # add w1 to w2
    weight_keys = w1.keys()
    avg_state_dict = collections.OrderedDict()
    for idx, key in enumerate(weight_keys):
        weight = per[idx].clone()
        
        weight[weight <= 0] = -1
        weight[weight > 0] = retain_ratio
        weight[weight < 0] = 1
        
        for l in range(len(w1[key].shape) - len(weight.shape)):
            weight = weight.unsqueeze(-1)
        #print(per[idx].shape, w1[key].shape)
        key_sum = (1-weight) * w1[key].data.cpu()
        key_sum += weight * w2[key].data.cpu()
        avg_state_dict[key] = key_sum
    return avg_state_dict

def get_exp_avg(optimizer, model):

    exp_avg = []

    param = optimizer.param_groups[0]["params"]
    keys = list(model.state_dict().keys())

    for idx, p in enumerate(param):
        if p.grad is not None:
            if "decoder_fuse" in keys[idx]:
                state = optimizer.state[p]
                exp_avg.append(state['exp_avg'].clone().cpu())

    return exp_avg

def model_diff(oldmodel, model):
    exp_avg = []

    old_para = oldmodel.decoder_fuse.parameters()
    para = list(model.decoder_fuse.parameters())
    for idx, l in enumerate(old_para):
        exp_avg.append((para[idx] - l).detach().clone().cpu())

    return exp_avg

def apply_morphological_operations(mask, operation='dilation', kernel_size=3, iterations=1):
    """
    Apply morphological operations (dilation or erosion) to a binary mask using PyTorch.

    Parameters:
    - mask: torch tensor of shape (1, 4, 80, 80, 80)
    - operation: 'dilation' or 'erosion'
    - kernel_size: size of the kernel for the operation
    - iterations: number of iterations for the operation

    Returns:
    - result_mask: torch tensor of shape (1, 4, 80, 80, 80) after applying the operation
    """
    # Create a kernel for the morphological operation
    kernel = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)#  .repeat((1,mask.shape[1],1,1,1))  # Shape (1, 1, kernel_size, kernel_size, kernel_size)

    result_mask = mask.clone()
    # print(mask.shape)
    for _ in range(iterations):
        # print(result_mask.shape)
        if operation == 'dilation':
            for channel in range(mask.shape[1]):
                result_mask[:, channel: channel+1] = F.conv3d(result_mask[:, channel: channel+1].float(), kernel, padding=kernel_size//2) > 0
        elif operation == 'erosion':
            for channel in range(mask.shape[1]):
                result_mask[:, channel: channel+1] = F.conv3d(result_mask[:, channel: channel+1].float(), kernel, padding=kernel_size//2) == kernel.numel()
        else:
            raise ValueError("Operation must be 'dilation' or 'erosion'")
    
    return result_mask.float()

def getClusDict(glb_Pnames, labels):
    glb_Pdict = {}
    for i in range(len(labels)):
        label = labels[i]
        pname = glb_Pnames[i]
        glb_Pdict[pname] = int(label)
    
    return glb_Pdict

def EMA_cls_Fs(prior_Fs, glb_protos, glb_Pnames, glb_clusDict, round):
    '''
    prior_Fs : (k, C)
    glb_protos : (len, C)

    '''
    C = glb_protos.shape[1]

    clu_Fs = {0:[], 1:[], 2:[], 3:[]} #, 4:[]}#, 5:[]}# , 4:[]}   #, 3:[] , 4:[] , 5:[]}   # , 6:[], 7:[]} # 数量与cluseter数目一致
    clu_clsFs = np.zeros(shape=[4, C])
    for i in range(len(glb_protos)):
        protos = glb_protos[i].numpy()  # .astype('float32')    # (C, )
        Pname = glb_Pnames[i]
        label = glb_clusDict[Pname]
        clu_Fs[label].append(protos)

    for c in clu_Fs:
        clu_clsFs[c] = (sum(clu_Fs[c])/len(clu_Fs[c])).astype('float32')
    
    alpha = 0.999
    if round==1:
        glb_Fs = clu_clsFs
    else:
        glb_Fs = alpha * prior_Fs.numpy() + (1-alpha)* clu_clsFs
    return glb_Fs

def EMA_Fs(prior_Fs, glb_protos, glb_Pnames, glb_clusDict, round):
    cls, C = glb_protos[0].shape
    clu_Fs = {0:[], 1:[] , 2:[], 3:[]} # , 4:[] , 5:[]}   # , 6:[], 7:[]} # 数量与cluseter数目一致
    clu_clsFs = np.zeros(shape=[4, cls, C])
    for i in range(len(glb_protos)):
        protos = glb_protos[i].numpy()  # .astype('float32')    # (cls=4, C)
        Pname = glb_Pnames[i]
        label = glb_clusDict[Pname]
        clu_Fs[label].append(protos)
    for c in clu_Fs:
        clu_clsFs[c] = (sum(clu_Fs[c])/len(clu_Fs[c])).astype('float32')
    
    alpha = 0.999
    if round==1:
        glb_Fs = clu_clsFs
    else:
        glb_Fs = alpha * prior_Fs.numpy() + (1-alpha)* clu_clsFs
    return glb_Fs

def calDist(fts, prototype):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: B x C x N x H x W
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None, None], dim=1)
    return dist

def getPrototype(features, masks):
    """Extract class prototypes via masked average pooling

    Args:
        features : [B, C, N, H, W]
        masks : [B, 1, N, H, W] 类别特定
    """
    c_protos = []
    for b in range(features.shape[0]):  # = Batch_Size
        fts, msk = features[b:b+1, ...], masks[b:b+1, ...]  # [1, C, N, H, W]  [1, 1, N, H, W]
        # fts - torch.Size([1, 128, 10, 10, 10])
        # msk - torch.Size([1, 1, 80, 80, 80])
        fts = F.interpolate(fts, size=msk.shape[-3:])   # torch.Size([1, 128, 80, 80, 80])
        masked_fts = torch.sum(fts * msk, dim=(2,3,4)) / (msk.sum(dim=(2,3,4))+1e-5)
        # [1, C]
        c_protos.append(masked_fts.detach().cpu())
    return sum(c_protos)/len(c_protos)        

def getClsPrototypes(features, masks):    
    """Extract class prototypes via masked average pooling

    Args:
        features : [B, C, N, H, W]
        masks : [B, cls, N, H, W]
    """
    batchFs = []
    for b in range(features.shape[0]):  # = Batch_Size
        cFs = []
        for c in range(masks.shape[1]): # = num_class
            fts, msk = features[b:b+1, ...], masks[b:b+1, c:c+1, ...]  # [1, C, N, H, W]  [1, 1, N, H, W]
            # fts - torch.Size([1, 128, 10, 10, 10])
            # msk - torch.Size([1, 1, 80, 80, 80])
            fts = F.interpolate(fts, size=msk.shape[-3:])   # torch.Size([1, 128, 80, 80, 80])
            masked_fts = torch.sum(fts * msk, dim=(2,3,4)) / (msk.sum(dim=(2,3,4))+1e-5)
            cFs.append(masked_fts.squeeze().detach().cpu())
        Fs = torch.stack(cFs, dim=0)    # [cls, C]
        batchFs.append(Fs)
        # [1, C]
        
    return batchFs

def del_tensor_0_cloumn(Cs):    # 删除全为0的列
    idx = torch.where(torch.all(Cs[..., :] == 0, axis=0))[0]
    all = torch.arange(Cs.shape[1])
    for i in range(len(idx)):
        all = all[torch.arange(all.size(0))!=idx[i]-i] 
    Cs = torch.index_select(Cs, 1, all)
    return Cs

def sub_sample(arrs, num):
    # 通过索引对二维数据进行随机采样
    ind = np.arange(len(arrs[1]))
    sub_ind = np.random.choice(ind, num, replace=False)
    sub_arrs = np.array(arrs)[:, sub_ind]
    return sub_arrs

def getClsFeatures(features, masks):    
    """Extract class features via masked
    Args:
        features : [B, C, N, H, W]
        masks : [B, cls, N, H, W]
    """
    batchFs = []
    maxl = 10000
    for b in range(features.shape[0]):  # = Batch_Size
        cFs = []
        for c in range(masks.shape[1]): # = num_class
            fts, msk = features[b:b+1, ...], masks[b:b+1, c:c+1, ...]  # [1, C, N, H, W]  [1, 1, N, H, W]
            # fts - torch.Size([1, 128, 10, 10, 10])
            # msk - torch.Size([1, 1, 80, 80, 80])
            # fts = F.interpolate(fts, size=msk.shape[-3:])   # torch.Size([1, 128, 80, 80, 80])
            msk = F.interpolate(msk, size=fts.shape[-3:])
            masked_fts = (fts * msk).squeeze()  # [1, C, n,h,w]
            cls_F = masked_fts.reshape(masked_fts.shape[0], -1).detach().cpu().numpy()
            # 删除类别掩码后为0的无效像素点
            cls_F = cls_F[:, [not np.all(cls_F[:, i] == 0) for i in range(cls_F.shape[1])]]
            # 对像素点进行随机采样
            if cls_F.shape[1] > maxl:
                cls_F = sub_sample(cls_F, maxl)
            cFs.append(torch.from_numpy(cls_F))
        # Fs = torch.stack(cFs, dim=0)    # [cls, CNHW]
        batchFs.append(cFs)
        # [1, C]
        
    return batchFs

def avgProtos(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]   #.data.cpu()
            for i in proto_list:
                proto += i  # .data.cpu()
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def getClsF_samplePixel(scale_Fs, masks):
    """Extract class features via masked
    Args:
        scale_Fs : len * [B, C, N, H, W]
        masks : [B, cls, N, H, W]
    """
    maxl = 10000
    assert len(scale_Fs)==len(masks)
    for i in range(len(scale_Fs)):
        Fs = scale_Fs[i]
        msk = masks[i]
        msk = F.interpolate(msk, size=Fs.shape[-3:])
        masked_fts = (Fs * msk).squeeze()
        cls_F = masked_fts.reshape(masked_fts.shape[0], -1).detach().cpu().numpy()
        # 删除类别掩码后为0的无效像素点
        cls_F = cls_F[:, [not np.all(cls_F[:, i] == 0) for i in range(cls_F.shape[1])]]
        # 对像素点进行随机采样
        if cls_F.shape[1] > maxl:
            cls_F = sub_sample(cls_F, maxl) 
    

def cluster_Fs(Fs, asCls=False, num_cluster = 4):
    '''
    Fs - tensor [len, cls, C]  
    return: array [k, cls*C]
    '''
    img_Fs = torch.stack(Fs, dim=0)     # (len, cls, C)
    if asCls:
        cls_clusCents, cls_indexes = [], { }
        for c in range(img_Fs.shape[1]):
            cls_imgFs = img_Fs[:, c]    # (len, C)
            clus_cents, indexes = cluster(cls_imgFs, num_cluster, 30)
            cls_clusCents.append(clus_cents)
            cls_indexes[c] = indexes
        return cls_clusCents, cls_indexes
    else:
        img_Fs = img_Fs.reshape(img_Fs.shape[0], img_Fs.shape[1]*img_Fs.shape[2])   # (len, cls*C)
        clus_cents, indexes = cluster(img_Fs, num_cluster, 30)
        return clus_cents, indexes

def cluster(clusX, centroid_num, iter_num):
    clusX = clusX.numpy().astype('float32')
    ncentroids = centroid_num
    niter = iter_num
    d = clusX.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=True, gpu=False)
    kmeans.max_points_per_centroid = 10000000

    kmeans.train(clusX)
    # get the result
    cluster_centroids = kmeans.centroids
    cluster_loss = kmeans.obj
    D, I = kmeans.index.search(clusX, 1)

    return cluster_centroids, I.squeeze()

def avgGLBprotos(modal_glb_protos, isFusion):
    modal_protos = {}
    label_protos = {}
    for modal in range(len(modal_glb_protos)):
        m_protos = modal_glb_protos[modal]  # 特定模态的类别原型特征
        for [label, proto_list] in m_protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0][0]   # .data.cpu()
                for i in proto_list:
                    proto += i[0]  # .data.cpu()
                m_protos[label] = proto / len(proto_list)
            else:
                m_protos[label] = proto_list[0]        
                
        modal_protos[modal] = m_protos
        
    if isFusion:
        for label in range(len(m_protos)):
            label_protos[label] = 0
            for modal in range(len(modal_glb_protos)):
                label_protos[label] += modal_protos[modal][label]
            label_protos[label] = label_protos[label]/len(modal_glb_protos)       
        return label_protos
    else:
        return modal_protos   


def avg_EW_per(w1, w2, w3, w4, per):
    weight_keys = w1.keys()
    avg_state_dict = collections.OrderedDict()
    for idx, key in enumerate(weight_keys):

        cweight1 = per[0][idx]
        for l in range(len(w1[key].shape) - len(cweight1.shape)):
            cweight1 = cweight1.unsqueeze(-1)

        key_sum = cweight1 * w1[key].data.cpu()

        cweight2 = per[1][idx]
        for l in range(len(w1[key].shape) - len(cweight2.shape)):
            cweight2 = cweight2.unsqueeze(-1)

        key_sum += cweight2 * w2[key].data.cpu()

        cweight3 = per[2][idx]
        for l in range(len(w1[key].shape) - len(cweight3.shape)):
            cweight3 = cweight3.unsqueeze(-1)

        key_sum += cweight3 * w3[key].data.cpu()

        cweight4 = per[3][idx]
        for l in range(len(w1[key].shape) - len(cweight4.shape)):
            cweight4 = cweight4.unsqueeze(-1)

        key_sum += cweight4 * w4[key].data.cpu()
        avg_state_dict[key] = key_sum / (cweight1 + cweight2 + cweight3 + cweight4)
    return avg_state_dict

def avg_EW_per_(w1, w2, per): # w2: target
    weight_keys = w1.keys()
    avg_state_dict = collections.OrderedDict()
    for idx, key in enumerate(weight_keys):
        weight = per[idx].clone()
        for l in range(len(w1[key].shape) - len(weight.shape)):
            weight = weight.unsqueeze(-1)

        key_sum = (1-weight) * w1[key].data.cpu()
        key_sum += weight * w2[key].data.cpu()
        avg_state_dict[key] = key_sum
    return avg_state_dict

def avg_EW_per_4_(w1, w2, per, retain_ratio = 0):
    weight_keys = w1.keys()
    avg_state_dict = collections.OrderedDict()
    for idx, key in enumerate(weight_keys):
        weight = per[idx].clone()
        
        weight[weight <= 0] = -1
        weight[weight > 0] = retain_ratio
        weight[weight < 0] = 1
        
        for l in range(len(w1[key].shape) - len(weight.shape)):
            weight = weight.unsqueeze(-1)
        #print(per[idx].shape, w1[key].shape)
        key_sum = (1-weight) * w1[key].data.cpu()
        key_sum += weight * w2[key].data.cpu()
        avg_state_dict[key] = key_sum
    return avg_state_dict

def avg_EW_(ws, ms):
    m = torch.stack(ms, dim=0)
    count = int((m==True).sum())
    weight_keys = ws[0].keys()
    avg_state_dict = collections.OrderedDict()
    for key in weight_keys:
        for idx, cw in enumerate(ws):
            if idx == 0:
                key_sum = ms[idx] * cw[key].data.cpu()
            else:
                key_sum += ms[idx] * cw[key].data.cpu()
        avg_state_dict[key] = key_sum / count
    return avg_state_dict

def c6_avg_EW(w1, w2, w3, w4, w5, w6, m1, m2, m3, m4, m5, m6):
    m = torch.stack((m1,m2,m3,m4,m5,m6), dim=0)
    count = int((m==True).sum())
    weight_keys = w1.keys()
    avg_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = m1 * w1[key].data.cpu()
        key_sum += m2 * w2[key].data.cpu()
        key_sum += m3 * w3[key].data.cpu()
        key_sum += m4 * w4[key].data.cpu()
        key_sum += m5 * w5[key].data.cpu()
        key_sum += m6 * w6[key].data.cpu()
        avg_state_dict[key] = key_sum / count
    return avg_state_dict  

             
def average_weights(w, client_weights):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # num_batches_tracked is a non trainable LongTensor and
        # num_batches_tracked are the same for all clients for the given datasets
        if 'num_batches_tracked' in key:
            w_avg[key].data.copy_(w[0][key])
        else:
            tmp = torch.zeros_like(w_avg[key]).cpu()
            for c_idx in range(0, len(client_weights)):
                tmp += client_weights[c_idx] * w[c_idx][key].cpu()
            w_avg[key].data.copy_(tmp)
    return w_avg

def covid_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def covid_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 4, 12500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def get_Cloader(args, dataset, idxs):

    # # split indexes for train, validation, and test (80, 10, 10)
    # idxs_train = idxs[:int(0.8*len(idxs))]
    # idxs_val = idxs[int(0.8*len(idxs)):]
    from data.data_utils import init_fn
    dataloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, 
                            shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=init_fn)
    return dataloader

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        return sample


