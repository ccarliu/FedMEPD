import os
from datetime import datetime
import logging
import random
import time
import torch
import numpy as np
import copy

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils.lr_scheduler import LR_Scheduler

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