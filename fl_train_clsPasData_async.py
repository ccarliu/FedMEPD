from asyncore import write
from subprocess import check_output
from tabnanny import check
import torch
import os
import random
import numpy as np
import time
from torch import nn
from tqdm import tqdm
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.data_utils import init_fn
import collections
import copy


from models import model
from utils.fl_utils import EMA_cls_Fs, cluster_Fs, getClsPrototypes, getClusDict, get_exp_avg, model_diff, avg_EW_per_4, avg_EW_per_4_, downloadGLBweights, avg_EW, get_client_weights4
from utils.lr_scheduler import LR_Scheduler
from utils import criterions
from dataset.datasets import Brats_test, Brats_train, GLB_Brats_train
from options import args_parser
from utils.predict import global_test, local_test, test_softmax

from multiprocessing import Pool

def self_cuda(obj, device):
    if isinstance(obj, list):
        return [self_cuda(l, device) for l in obj]
    elif isinstance(obj, dict):
        return {k:self_cuda(obj[k], device) for k in obj}
    elif torch.is_tensor(obj):
        return obj.to(device)
    return obj

def local_training(args, device, mask, dataloader, model, client_idx, global_Fs, global_round, optimizer):
    # set mode to train model

    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    model.train()
    model = model.to(device)
    model_old = copy.deepcopy(model)
    start = time.time()
    epoch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[]}
    optim = optimizer.state_dict()
    optimizer.load_state_dict({k:self_cuda(optim[k], device) for k in optim})


    step_lr = lr_schedule(optimizer, global_round)
    
    for iter in range(args.local_ep):
        batch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[]}
        
        for batch_idx, data in enumerate(dataloader):

            vol_batch, msk_batch = data[0].to(device), data[1].to(device)
            names = data[-1]
            msk = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)  # change lh
            
            msk = msk.to(device)
            model.is_training = True

            Px1, Px2, Px3, Px4 = global_Fs['x1'].to(device), global_Fs['x2'].to(device), global_Fs['x3'].to(device), global_Fs['x4'].to(device)
            Px1, Px2 = Px1.reshape(-1, Px1.shape[-1]), Px2.reshape(-1, Px2.shape[-1])
            Px3, Px4 = Px3.reshape(-1, Px3.shape[-1]), Px4.reshape(-1, Px4.shape[-1])   # torch.Size([40, C])
            
            fuse_pred, prm_preds, _, sep_preds = model(vol_batch, msk, Px1, Px2, Px3, Px4)

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1).float().to(device)
            prm_dice_loss = torch.zeros(1).float().to(device)
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, msk_batch, num_cls=args.num_class)
                prm_dice_loss += criterions.dice_loss(prm_pred, msk_batch, num_cls=args.num_class)
            prm_loss = prm_cross_loss + prm_dice_loss

            sep_cross_loss = torch.zeros(1).float().to(device)
            sep_dice_loss = torch.zeros(1).float().to(device)
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            loss = fuse_loss + prm_loss + sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            batch_loss['total'].append(loss.item())
            batch_loss['fuse'].append(fuse_loss.item())
            batch_loss['prm'].append(prm_loss.item())
            batch_loss['sep'].append(sep_loss.item())
            
        epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
        epoch_loss['fuse'].append(sum(batch_loss['fuse'])/len(batch_loss['fuse']))
        epoch_loss['prm'].append(sum(batch_loss['prm'])/len(batch_loss['prm']))
        epoch_loss['sep'].append(sum(batch_loss['sep'])/len(batch_loss['sep']))
        
    epoch_loss['total'] = sum(epoch_loss['total'])/len(epoch_loss['total'])
    epoch_loss['fuse'] = sum(epoch_loss['fuse'])/len(epoch_loss['fuse'])
    epoch_loss['prm'] = sum(epoch_loss['prm'])/len(epoch_loss['prm'])
    epoch_loss['sep'] = sum(epoch_loss['sep'])/len(epoch_loss['sep'])
    
    
    exp_avg = model_diff(model_old, model)
    msg = 'client_{} local training total time: {:.4f} hours'.format(client_idx+1, (time.time() - start)/3600)
    msg_2 = 'client_{} local training loss: {:.4f}'.format(client_idx+1, epoch_loss['total'])
    print(msg)
    print(msg_2)

    
    model = model.cpu()
    return [model.c1_encoder.state_dict(), model.c2_encoder.state_dict(), model.c3_encoder.state_dict(),
            model.c4_encoder.state_dict(), model.decoder_fuse.state_dict()], epoch_loss, model, exp_avg, optimizer.state_dict()

def global_training(args, device, dataloader, model, modal_protos, round, optimizer):

    model.train()
    model = model.to(device)
    start = time.time()
    model_old = copy.deepcopy(model)

    Xscale_list = ['x1', 'x2', 'x3', 'x4']
    glb_Fs = {'x1':[], 'x2':[], 'x3':[], 'x4':[]}
    glb_Pnames = []

    optim = optimizer.state_dict()
    optimizer.load_state_dict({k:self_cuda(optim[k], device) for k in optim})

    
    step_lr = lr_schedule(optimizer, round)
    writer.add_scalar('lr_glb', step_lr, global_step=round)
    logging.info('############# global training on the server ############')
    
    for epoch in range(args.global_ep):
        
        e = time.time()
        step = epoch*len(dataloader) + len(dataloader)*round*args.global_ep
        
        for iter, data in enumerate(dataloader):
            vol, target, msk, p_name = data
            glb_Pnames += p_name
            vol_batch, msk_batch = vol.to(device), target.to(device)
            mask = msk.to(device)   # tensor([[True, True, True, True]], device='cuda:0')
            # vol_batch - torch.Size([B, 4, 80, 80, 80])
            # msk_batch - torch.Size([B, 4, 80, 80, 80])
            # mask - torch.Size([B, 4])
            model.is_training = True
            fuse_pred, prm_preds, features, sep_preds = model(vol_batch, mask, None,None,None,None) 
            # fuse_pred - torch.Size([1, 4, 80, 80, 80])
            # sep_preds - 4 * torch.Size([1, 4, 80, 80, 80])
            # prm_preds - 4 * torch.Size([1, 4, 80, 80, 80])
                    
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1).float().to(device)
            prm_dice_loss = torch.zeros(1).float().to(device)
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, msk_batch, num_cls=args.num_class)
                prm_dice_loss += criterions.dice_loss(prm_pred, msk_batch, num_cls=args.num_class)
            prm_loss = prm_cross_loss + prm_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            for i in range(len(features)):
                # Example usage
                # mask = torch.randint(0, 2, (1, 4, 80, 80, 80), dtype=torch.float32)  # Example binary mask
                
                scale = Xscale_list[i]
                # 对应当前尺度下的融合模态特征图
                fusion_features = features[i]       # Fx4 - torch.Size([1, 128, 10, 10, 10])
                cls_F = getClsPrototypes(fusion_features, msk_batch)  # (cls, C)
                glb_Fs[scale]  += cls_F

            loss = fuse_loss + prm_loss + sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('GlobalTrain/loss', loss.item(), global_step=step)
            # writer.add_scalar('GlobalTrain/proto_align_loss', proto_align_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/prm_dice_loss', prm_dice_loss.item(), global_step=step)

            if args.verbose and (iter%10==0):
                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.local_ep, (iter), len(dataloader), loss.item())
                msg += 'fusecross:{:.4f}, fusedice:{:.4f}, '.format(fuse_cross_loss.item(), fuse_dice_loss.item())
                msg += 'sepcross:{:.4f}, sepdice:{:.4f}, '.format(sep_cross_loss.item(), sep_dice_loss.item())
                msg += 'prmcross:{:.4f}, prmdice:{:.4f}'.format(prm_cross_loss.item(), prm_dice_loss.item())
                # msg += 'ProtoAlignLoss:{:.4f}'.format(proto_align_loss.item())
                logging.info(msg)

    #exp_avg = get_exp_avg(optimizer, model)
    exp_avg = model_diff(model_old, model)

    msg = 'server global training total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    return model.state_dict(), glb_Fs, glb_Pnames, exp_avg

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

if __name__ == '__main__':
    ### global model - 4 模态特异Encoder & 1 模态融合Decoder
    ### local model - 模态特异Encoder & Decoder
    ### FL过程中只传递Encoder参数，不做其他处理
    args = args_parser()
    
    # 数据预处理遵循RFNet
    args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    args.save_path = args.save_root + '/' + str(args.version)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    args.modelfile_path = os.path.join(args.save_path, 'model_files')
    if not os.path.exists(args.modelfile_path):
        os.makedirs(args.modelfile_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=args.save_path + '/fl_log.txt')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    
    writer = SummaryWriter(os.path.join(args.save_path, 'TBlog'))
    
    ##### modality missing mask
    if "m3" in args.setting_options:
        masks = [[True, True, True, True], [True, True, True,False], [True, False, True, True], [True, True, False, True], [False, True, True, True]]
        mask_name = ['flairt1cet1t2', 'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1t2']
    elif "m2" == args.setting_options:
        masks = [[True, True, True, True], [True, True, False,False],  [False, True, False, True], [True, False, False, True], [False, True, True, False], [False, False, True, True], [True, False, True, False]]
        mask_name = ['flairt1cet1t2', 'flairt1ce', 't1t2', 'flairt1', 't1cet2', 'flairt2', 't1cet1']
    elif "m1" in args.setting_options:
        masks = [[True, True, True, True], [True, False, False,False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
        mask_name = ['flairt1cet1t2', 'flair', 't1ce', 't1', 't2']
    elif "c8" in args.setting_options:
        masks = [[True, True, True, True], [True,  True, True,  False], [True,  False, True,  True], [True, True, False, False], [False, False, True, True], 
             [False, True, False, False], [False, False, False, True], [True,  True,  True, True], [True,  True,  True, True]]
        mask_name = ['m1111', 'm1110', 'm1011', 'm1100', 'm0011', 'm0100', 'm0001', 'm1111', 'm1111']
    
    if "c8" in args.setting_options:
        if args.dataname == "BRATS2020":
            args.train_file = {'glb':"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/server.csv", 
                                    1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c1.csv", 
                                    2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c2.csv", 
                                    3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c3.csv", 
                                    4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c4.csv",
                                    5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c5.csv", 
                                    6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c6.csv", 
                                    7:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c7.csv", 
                                    8:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c8.csv"}
        else:
            args.train_file = {'glb':"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/server.csv", 
                                    1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c1.csv", 
                                    2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c2.csv", 
                                    3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c3.csv", 
                                    4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c4.csv",
                                    5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c5.csv", 
                                    6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c6.csv", 
                                    7:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c7.csv", 
                                    8:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c8.csv"}
    else:
        args.train_file = {'glb':"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/glb.csv", 
                1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c1.csv", 
                2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c2.csv", 
                3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c3.csv", 
                4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c4.csv",
                5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c5.csv", 
                6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/c6.csv"
                }

        args.valid_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/val.csv"
        args.test_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/FedMEMA_pure_code/split/18_c4_c6/test.csv"
    
    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = ['flair', 't1ce', 't1', 't2']
    logging.info(masks_torch.int())

    ########## setting seed for deterministic
    if args.deterministic:
        # cudnn.enabled = False
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.local_devices = []
    args.num_devices = torch.cuda.device_count()
    for i in range(args.client_num):
        args.local_devices.append(torch.device('cuda:{}'.format(i%4))) # use 4 gpus
    
    ########## setting global and local model
    server_model = model.E4D4Model(num_cls=args.num_class, is_lc=False)
    client_model = model.E4D4Model(num_cls=args.num_class, is_lc=True)

    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    ########## FL setting ##########
    # define dataset, model, optimizer for each clients 
    dataloader_clients, validloader_clients, testloader_clients = [], [], []
    model_clients = []
    optimizer_clients = []
    exp_avg_clients = []
    client_counts, client_weights = [], []     ### FedAvg Setting

    client_weights_per = []
    client_count_per = []
    params_dec = list(client_model.decoder_fuse.parameters())

    keys = list(client_model.decoder_fuse.state_dict().keys())

    for l in range(args.client_num):
        c_weight = []
        c_count = []
        for l2, param in enumerate(params_dec):
            if l2%2 and l2 < 172:
                c_weight.append(torch.ones_like(param.data))
                c_count.append(torch.ones_like(param.data) * args.patience)
            elif l2 < 172:
                c_weight.append(torch.ones(param.data.shape[0]))
                c_count.append(torch.ones(param.data.shape[0]) * args.patience)
                for l in range(len(param.data.shape) - len(c_weight[-1].shape)):
                    c_weight[-1] = c_weight[-1].unsqueeze(-1)
                    c_count[-1] = c_count[-1].unsqueeze(-1)
            else: # for linear layers, personlized
                c_weight.append(torch.zeros_like(param.data))
                c_count.append(torch.zeros_like(param.data))
        client_count_per.append(c_count)
        client_weights_per.append(c_weight)    

    
    modal_list = ['flair', 't1ce', 't1', 't2']
    logging.info(str(args))

    for client_idx in range(args.client_num):
        chose_modal = 'all'
        lc_train_file = args.train_file[client_idx+1]
        data_set = Brats_train(transforms=args.train_transforms, root=args.datapath, 
                                modal=chose_modal, num_cls=args.num_class, train_file=lc_train_file)
        data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
        valid_set = Brats_test(transforms=args.test_transforms, root=args.datapath, 
                                modal=chose_modal, test_file=lc_train_file)
        valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = valid_loader
        
        # Set Optimizer for the local model update
        net = copy.deepcopy(client_model)   
        model_clients.append(net)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_clients.append(optimizer)
        dataloader_clients.append(data_loader)
        validloader_clients.append(valid_loader)
        testloader_clients.append(test_loader)
        logging.info('Client-{} : Brats dataset with modal {}'.format(client_idx+1, masks[client_idx]))
        logging.info('the length of Brats dataset is {} : {}'.format(len(data_set), len(valid_set)))

        device = args.local_devices[client_idx]

    #### global dataset 
    glb_train_file = args.train_file['glb']
    glb_dataset = GLB_Brats_train(transforms=args.train_transforms, root=args.datapath, 
                                modal='all', num_cls=args.num_class, train_file=glb_train_file, aug = False) # , partial = 0.1)
    glb_trainloader = DataLoader(dataset=glb_dataset, batch_size=1, num_workers=8,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
    glb_validset = Brats_test(transforms=args.test_transforms, root=args.datapath, 
                                modal='all', test_file=glb_train_file)
    glb_validloader = DataLoader(dataset=glb_validset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    glb_testloader = glb_validloader
    logging.info('Global : Brats dataset with all modal')
    logging.info('the length of Brats dataset is {} : {}'.format(len(glb_dataset), len(glb_validset)))

        
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(server_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_server = optimizer
        
    best_dices = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_dice = 0.0    
        
    
    if args.resume != 0:
        
        ckpt = torch.load(args.modelfile_path + '/last.pth')
        
        for client_i in range(args.client_num):
            model_clients[i].load_state_dict(clients_dict[client_i])
            optimizer_clients[i].load_state_dict(clients_optim_dict[client_i])

        server_model.load_state_dict(ckpt["server"])
        optimizer_server.load_state_dict(ckpt["os"]), 


        args.start_round = ckpt['round']
        cls_glb_clusDict = ckpt['cls_glb_clusDict']
        global_Fs = ckpt['global_Fs'] # 'x1': [], 'x2':[], 'x3':[], 'x4':[]
        best_dice = ckpt['best_dice']
        best_dices = ckpt['best_dices']
        
        print("load best result: {}, {}, {}, {}.".format(best_dice, best_dices[0], best_dices[1], best_dices[2]))
        print(best_dices)
        ckpt = torch.load(args.modelfile_path + "/" + str(args.start_round-1) + '_weight.pth')
        client_weights_per = ckpt["weight"]
        client_count_per = ckpt["count"]

        exp_avg = ckpt["exp_avg"]
    

    Xscale_list = ['x1', 'x2', 'x3', 'x4', 'x1_4']
    ########## FL Training ##########
    for round in tqdm(range(args.start_round, args.c_rounds+1)):
        exp_avg_clients = []
        start = time.time()
        if round==0:
         
            ##### global training
            glb_w, glb_protos, glb_Pnames, exp_avg = global_training(args, args.device, glb_trainloader, server_model, None, round, optimizer_server)
            ## 按某一尺度特征 按类别进行聚类
            cls_glb_clusDict = {}
            global_Fs = { } # 'x1': [], 'x2':[], 'x3':[], 'x4':[]
            for scale in Xscale_list:

                if scale != Xscale_list[-1]:
                
                    clu_Fs, labels = cluster_Fs(glb_protos[scale], asCls=True)  # [k, cls*C]
                    clu_Fs = np.stack(clu_Fs, axis=1)   # 再按类别cat起来 [k, cls, C]
                    clu_Fs = torch.from_numpy(clu_Fs)
                    global_Fs[scale] = clu_Fs
                    
                else:
                    clu_Fs, labels = cluster_Fs([torch.cat([glb_protos[s_][l] for s_ in Xscale_list[:4]], -1) for l in range(len(glb_protos['x1']))], asCls=True)  # [k, cls*C]

                    print(labels)
                if scale=='x4':
                    for c in range(args.num_class):
                        glb_clusDict = getClusDict(glb_Pnames, labels[c])
                        cls_glb_clusDict[c] = glb_clusDict

        else: 
            ##### local training
            local_weights, local_losses, local_protos = [], [], {}
            logging.info(f'\n | Global Training Round : {round} |')
            
            start = time.time()
            
            result = []

            branch_num = args.client_num // args.num_devices
            if branch_num % args.num_devices:
                branch_num += 1
            
            for branch in range(branch_num):
                
                torch.cuda.empty_cache()
                ctx = torch.multiprocessing.get_context("spawn")
                
                pool = ctx.Pool(args.num_devices)    
                
                for client_i_ in range(args.num_devices):
                    client_i = client_i_ + branch * args.num_devices
                    
                    if client_i >= args.client_num:
                            break

                    result.append(pool.apply_async(local_training, args=(args, args.local_devices[client_i], masks_torch[client_i], dataloader_clients[client_i], model_clients[client_i], client_i, global_Fs, round, optimizer_clients[client_i], )))
                    
                
                pool.close()
                pool.join()
            
            logging.info("client training: {}".format(time.time() - start))
            for client_i, i in enumerate(result):
                w, loss, m, exp_avg_c, optim = i.get()
                local_weights.append(w)
                local_losses.append(loss['total'])
                model_clients[client_i].load_state_dict(m.state_dict())
                optimizer_clients[client_i].load_state_dict(optim)
                exp_avg_clients.append(exp_avg_c)
                # local_protos[client_i] = agg_protos
                writer.add_scalar('LocalTrain/total_Loss/client_' + str(client_i + 1), loss['total'], round)
                writer.add_scalar('LocalTrain/Loss_fuse/client_' + str(client_i + 1), loss['fuse'], round)
                writer.add_scalar('LocalTrain/Loss_prm/client_' + str(client_i + 1), loss['prm'], round)
                writer.add_scalar('LocalTrain/Loss_sep/client_' + str(client_i + 1), loss['sep'], round)


            # global Aggre and Fusion
            # local_protos - 4个客户端即4个模态的4个类别原型

            ### global training
            
            glb_w, glb_protos, glb_Pnames, client_weights_per, exp_avg = uploadLCweightsandGLBupdate(masks_torch, server_model, local_weights, glb_trainloader, None, round, client_weights_per, optimizer)   
            
            if round > args.pretrain:
                for l in range(len(client_weights_per)):
                    client_weights_per[l], client_count_per[l] = get_client_weights4(exp_avg, exp_avg_clients[l], client_weights_per[l], client_count_per[l], client_idx = l, patience = args.patience) # changed lh
            

            for scale in Xscale_list[:4]:
                scale_protos = torch.stack(glb_protos[scale], dim=0)    # (len, cls, C)
                cls_protos = []
                for c in range(args.num_class):
                    clu_Fs = EMA_cls_Fs(global_Fs[scale][:,c], scale_protos[:,c], glb_Pnames, cls_glb_clusDict[c], round)
                    cls_protos.append(clu_Fs)
                cls_protos = np.stack(cls_protos, axis=1)   # (k, cls, C)
                global_Fs[scale] = (torch.from_numpy(cls_protos)).float()
            

        ##### Eval the model after aggregation and 10 round
        if (round+1)%args.eval==0:
            logging.info('-'*20 + 'Test All the Models per 10 round'+ '-'*20)
            with torch.no_grad():
                # test clients
                results = []
                branch_num = args.client_num // args.num_devices
                if branch_num % args.num_devices:
                    branch_num += 1

                for branch in range(branch_num):
                
                    torch.cuda.empty_cache()
                    ctx = torch.multiprocessing.get_context("spawn")
                    
                    pool = ctx.Pool(args.num_devices)
                    
                    for c_ in range(args.num_devices):
                        c = branch * args.num_devices + c_
                        
                        if c >= args.client_num:
                            break
                        
                        results.append(pool.apply_async(local_test, (args, validloader_clients[c], model_clients[c], args.local_devices[c], 'BRATS2020', global_Fs, masks[c],)))
                
                    pool.close()
                    pool.join()
                
                for c, result in enumerate(results):
                    dice_score = result.get()
                    c_model = model_clients[c]
                    avgdice_score = sum(dice_score)/len(dice_score)
                    logging.info('--- Eval at round_{}, Avg_Scores: {:.4f}, cls_Dice: {}'
                                                        .format((round), avgdice_score*100, dice_score))
                    writer.add_scalar('Eval_AvgDice/client_'+str(c+1), avgdice_score*100, round)
                    
                    if best_dices[c] < avgdice_score:
                        best_dices[c] = avgdice_score
                        torch.save({
                            'round': round+1,
                            'dice': dice_score,
                            'state_dict': c_model.state_dict(),
                            'proto' :global_Fs
                        }, args.modelfile_path + '/client-%d_round_%d_model_best.pth'%(c+1, round))

                # test server
                logging.info('-'*15+' Test the Global Model '+'-'*15)
                mask = [True, True, True, True]
                glbdice = global_test(glb_validloader, server_model, args.device, 'BRATS2020', mask)
                avg_glbdice = sum(glbdice)/len(glbdice)
                logging.info('--- Eval at round_{}, Avg_Scores: {:.4f}, cls_Dice: {}'
                                .format((round), avg_glbdice*100, glbdice))
                writer.add_scalar('Eval_AvgDice/server', avg_glbdice, round)

                if best_dice < avg_glbdice:
                    best_dice = avg_glbdice
                    torch.save({
                        'round': round+1,
                        'dice': glbdice,
                        'state_dict': server_model.state_dict()
                    }, args.modelfile_path + '/sever_model_best.pth')

        
        downloadGLBweights(glb_w, model_clients, copy.deepcopy(client_count_per))
        logging.info('*'*10+'FL train a round total time: {:.4f} hours'.format((time.time() - start)/3600)+'*'*10)
        if (round+1)%args.eval == 0:
            torch.save({
            
            'round': round + 1,
            
            "clients_dict": [model_clients[client_i].state_dict() for client_i in range(args.client_num)],
            "clients_optim_dict": [optimizer_clients[client_i].state_dict() for client_i in range(args.client_num)],
            
            'server': server_model.state_dict(),
            'os': optimizer_server.state_dict(),

            'cls_glb_clusDict':cls_glb_clusDict,
            'global_Fs':global_Fs,
            'best_dice': best_dice,
            'best_dices': best_dices,
            'exp_avg': exp_avg
            }, args.modelfile_path + '/last.pth')

            torch.save({

            'weight': client_weights_per,
            'count': client_count_per,

            }, args.modelfile_path + '/' + str(round) + '_weight.pth')
                
        
        import json
        with open(args.save_path+"/glb_Pdict.json", 'w') as f1:
            f1.write(json.dumps(cls_glb_clusDict, indent=4, ensure_ascii=False))
        if round != 0:     
            for scale in Xscale_list[:4]:
                Fs = global_Fs[scale].numpy()
                np.save(args.save_path+'/glb_'+str(scale)+'.npy', Fs)

    writer.close()    
