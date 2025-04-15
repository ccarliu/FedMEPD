import os
from datetime import datetime
import logging
import random
import time
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import sys
# sys.path.append('/disk3/qd/edFL/code')
from options_18_test import args_parser
from models import model
from utils.lr_scheduler import LR_Scheduler
from utils.predict import AverageMeter, softmax_output_dice_class4, softmax_output_dice_class5

from utils import criterions
from dataset.data_utils import init_fn
from dataset.datasets import Brats_test
import csv


def local_test(args, test_loader, model, device, dataname, glb_features, modal_mask, writer = None, store_path = None):
    model = model.to(device)
    H, W, T = 240, 240, 155
    patch_size = 80
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, 80).float().to(device)    # .cuda()
    glb_protos = []
    if glb_features== {}:
            Px1, Px2, Px3, Px4 = None, None, None, None
            logging.info('*'*10+'the local test without Global Features!'+'*'*10)
    else:        
        ##### 聚类特征图
        Px1, Px2, Px3, Px4 = glb_features['x1'].to(device), glb_features['x2'].to(device), glb_features['x3'].to(device), glb_features['x4'].to(device)
        Px1, Px2 = Px1.reshape(-1, Px1.shape[-1]), Px2.reshape(-1, Px2.shape[-1]), 
        Px3, Px4 = Px3.reshape(-1, Px3.shape[-1]), Px4.reshape(-1, Px4.shape[-1]) 

        logging.info('*'*10+'the local test with Global Features!'+'*'*10)

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'NET-ncr_net', 'ED-edema', 'ET-enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
    else:
        num_cls = 2
        class_evaluation= 'WT-whole', 'TC-core', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema'
        
    for i, data in enumerate(test_loader):
        
        target = data[1].to(device)
        x = data[0].to(device)
        # print(x.shape, x.dtype, x.max(), x.min())
        names = data[-1]
        if modal_mask is not None:
            mask = torch.from_numpy(np.array(modal_mask)).unsqueeze(0)     # torch.Size([1, 4])
            # mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.to(device)
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size)

        w_cnt = int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size)

        z_cnt = int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().to(device) #.cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device) #.cuda()
        model.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    with torch.no_grad():
                        x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+80]
                        # 1.传递GLb_Features, RFNet
                        # print(mask.shape, x_input.shape, x_input.dtype)
                        # pred_part, _, _ = model(x_input, mask, Px1, Px2, Px3, Px4)   # glb_protos
                        # 不传递
                        pred_part, _, _, _ = model(x_input[:, mask[0]])
                        # 2. HeMIS
                        # pred_part, _ = model(x_input.transpose(1,0).unsqueeze(2), mask, Px1, Px2, Px3, Px4)
                        # 3. mmmodel
                        # pred_part, _, _, _ = model(x_input, mask, Px1, Px2, Px3, Px4)
                        pred[:, :, h:h+patch_size, w:w+patch_size, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)    # torch.Size([1, H, W, D])

        if store_path is not None:
            np.save(os.path.join(store_path, data[2][0] + ".npy"), pred.cpu().numpy().astype(np.int8))

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            if writer is not None:
                writer.writerow([names[0], scores_evaluation[0,0], scores_evaluation[0,1],scores_evaluation[0,2]])
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)
    # logging.info(msg)
    model.train()
    model = model.cpu()
    return vals_evaluation.avg

def test_softmax(test_loader, model, device, dataname = 'BRATS2020', writer=None, store_path = None):
    
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)    # .cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'NET-ncr_net', 'ED-edema', 'ET-enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        

    for i, data in enumerate(test_loader):
        target = data[1].to(device)
        x = data[0].to(device)
        names = data[-1]
        # if feature_mask is not None:
        #     mask = torch.from_numpy(np.array(feature_mask))     # torch.Size([1, 4])
        #     mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        # else:
        #     mask = data[2]
        # mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int(np.ceil((H - 80) / (80 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int(80 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 80)

        w_cnt = np.int(np.ceil((W - 80) / (80 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int(80 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 80)

        z_cnt = np.int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().to(device) #.cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device) #.cuda()
        # model.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    pred_part, _, _ = model(x_input)
                    pred[:, :, h:h+80, w:w+80, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)    # torch.Size([1, H, W, D])

        if store_path is not None:
            np.save(os.path.join(store_path, data[2][0] + ".npy"), pred.cpu().numpy().astype(np.int8))

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            writer.writerow([names[0], *scores_separate])
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg +=  str(scores_evaluation[k].mean()) + ', '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    logging.info(msg)
    model.train()
    return vals_evaluation.avg

    
def test_softmax_temp(test_loader, model, device, dataname = 'BRATS2020', mask = None):
    
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)    # .cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'NET-ncr_net', 'ED-edema', 'ET-enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        

    for i, data in enumerate(test_loader):
        target = data[1].to(device)
        x = data[0].to(device)
        names = data[-1]
        # if feature_mask is not None:
        #     mask = torch.from_numpy(np.array(feature_mask))     # torch.Size([1, 4])
        #     mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        # else:
        #     mask = data[2]
        # mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int(np.ceil((H - 80) / (80 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int(80 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 80)

        w_cnt = np.int(np.ceil((W - 80) / (80 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int(80 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 80)

        z_cnt = np.int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().to(device) #.cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device) #.cuda()
        # model.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    pred_part, _, _ = model(x_input, mask, None, None, None, None)
                    pred[:, :, h:h+80, w:w+80, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)    # torch.Size([1, H, W, D])

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            print(scores_evaluation)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg +=  str(scores_evaluation[k].mean()) + ', '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    logging.info(msg)
    model.train()
    return vals_evaluation.avg

if __name__ == '__main__':
    args = args_parser()
    
    args.train_transforms = 'Compose([RandCrop3D((120,120,120)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    #masks = [[True, True, True, True], [True, True, False,False], [False, False, True, True], [True, False, True, False], [False, True, False, True], [True, False, False, True], [False, True, True, False]]
    mask_name = ['flairt1cet1t2', 'flairt1ce', 't1t2', 'flairt1', 't1cet2', 'flairt2', 't1cet1']
    masks = [[True, False, False,False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
    mask_name = ['flair', 't1ce', 't1', 't2']
    
    if "m3" in args.setting_options:
    
        masks = [[True, True, True, True], [True, True, True,False], [True, False, True, True], [True, True, False, True], [False, True, True, True]]
        mask_name = ['flairt1cet1t2', 'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1t2']
    elif "m2" == args.setting_options:
        masks = [[True, True, True, True], [True, True, False,False], [False, False, True, True], [True, False, True, False], 
             [False, True, False, True], [True, False, False, True], [False, True, True, False]]
        mask_name = ['flairt1cet1t2', 'flairt1ce', 't1t2', 'flairt1', 't1cet2', 'flairt2', 't1cet1']
    elif "m1" in args.setting_options:
        masks = [[True, True, True, True], [True, False, False,False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
        mask_name = ['flairt1cet1t2', 'flair', 't1ce', 't1', 't2']
    elif "c8" in args.setting_options:
        masks = [[True, True, True, True], [True,  True, True,  False], [True,  False, True,  True], [True, True, False, False], [False, False, True, True], 
             [False, True, False, False], [False, False, False, True], [True,  True,  True, True], [True,  True,  True, True]]
        mask_name = ['m1111', 'm1110', 'm1011', 'm1100', 'm0011', 'm0100', 'm0001', 'm1111', 'm1111']
    else:
        masks = [[True, True, True, True], [True, True, False,False], [False, True, False, True], [True, False, False, True], [False, True, True, False]]
        mask_name = ['flairt1cet1t2', 'flair', 't1ce', 't1', 't2']

    masks_torch = torch.from_numpy(np.array(masks))
    
    # print (masks_torch.int())
    
    args.save_root = 'test_results_2'
    partID = args.maskid
    # args.resume_path = "/data3/tp_qd/FedMEMA_96/results/cl_rf_cn3_c3_07052236/model_files/model_e740_0.6952.pth"
    args.version = args.resume_path.split('/')[6]
    name = args.resume_path.split('/')[-1].split('.')[0]
    # timestamp = datetime.now().strftime("%m%d%H%M")
    args.save_path = args.save_root + '/' + str(args.version)  + "_temp" # + '_%s'%(timestamp)
    print(args.save_path, name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=args.save_path + '/'+str(name)+'.txt')
                        
    logging.info(masks_torch.int())
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    
    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########## setting models
    args.num_class = 4   # for Brats2019
    # model = model.attEDModel(num_cls=args.num_class)
    # model = model.E4D4Model(num_cls=args.num_class, is_lc=args.fl)
    model = model.EDModel(num_cls=args.num_class, channel = masks_torch[partID].sum())
    # model = model.E4D4Model_LC(num_cls=args.num_class, is_lc = False, is_hc=args.fl)
    
    model = model.to(args.device)
        
    
    # logging.info('Valid Dice : ', checkpoint['dice'])
    args.test_file = "/apdcephfs_cq10/share_1290796/lh/dataset/BRATS2018_Training_none_npy/split_4c_1g_1v_1t/test.csv"
    args.train_file = {0:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/server.csv", 
                            1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c1.csv", 
                            2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c2.csv", 
                            3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c3.csv", 
                            4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c4.csv",
                            5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c5.csv", 
                            6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c6.csv", 
                            7:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c7.csv", 
                            8:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/18_c8_heter_modalnum/c8.csv"}
    
    '''
    args.train_file = {0:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/server.csv", 
                            1:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c1.csv", 
                            2:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c2.csv", 
                            3:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c3.csv", 
                            4:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c4.csv",
                            5:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c5.csv", 
                            6:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c6.csv", 
                            7:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c7.csv", 
                            8:"/apdcephfs_cq10/share_1290796/lh/FedMEMA/split/20_c8_heter_modalnum/c8.csv"}
    '''
    # args.test_file = '/apdcephfs/share_1290796/lh/BRATS2020_Training_none_npy/split_0/test.txt'
    # args.test_file = "/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_50/test.txt"
    args.chose_modal = 'all'
    if "m3" in args.setting_options or "m1" in args.setting_options:
        args.test_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/split_4c_1g_1v_1t/test.csv"
        test_set = Brats_test(transforms=args.test_transforms, root=args.datapath, modal=args.chose_modal, test_file=args.test_file, all_ = True)
    elif "m2" == args.setting_options:
        args.test_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/split_6c_1g_1v_1t/test.csv"
        test_set = Brats_test(transforms=args.test_transforms, root=args.datapath, modal=args.chose_modal, test_file=args.test_file, all_ = True)
    elif "c8" in args.setting_options:
        test_set = Brats_test(transforms=args.test_transforms, root=args.datapath, modal=args.chose_modal, test_file=args.train_file[partID])
    else:
        args.test_file = "/apdcephfs_cq10/share_1290796/lh/FedMEMA/split_6c_1g_1v_1t/test.csv"
        test_set = Brats_test(transforms=args.test_transforms, root=args.datapath, modal=args.chose_modal, test_file=args.test_file, all_ = True)

    modal_list = ['flair', 't1ce', 't1', 't2']
    
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    logging.info(str(args))
    logging.info(args.setting_options)
    logging.info('the Brats dataset with modal {}'.format(mask_name[partID]))
    logging.info('the length of Brats test dataset is {}'.format(len(test_set)))
    
    mask = masks_torch[partID]
    globalFS = { }
    scale_list = ['x1','x2','x3','x4']
    
    ########## resume model from checkpoint
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=args.device)
        if "last" in args.resume_path:
            if partID != 0:
                model.load_state_dict(checkpoint["c"+str(partID)])
                result_file = open(os.path.join(args.save_path, "last_" + str(partID) + ".csv"), "w") 
                csv_writer = csv.writer(result_file)
            else:
                model.load_state_dict(checkpoint["server"])
                result_file = open(os.path.join(args.save_path, "last_" + str(partID) + ".csv"), "w") 
                csv_writer = csv.writer(result_file)
        else:
            model.load_state_dict(checkpoint['state_dict'])
            result_file = open(os.path.join(args.save_path, name + ".csv"), "w") 
            csv_writer = csv.writer(result_file)
    

    result_store_path = os.path.join(args.save_path, str(partID))
    # os.mkdir(result_store_path, exist_ok=True)
    import pathlib
    pathlib.Path(result_store_path).mkdir(parents=True, exist_ok=True) 

    if args.fl != 0:
        try:
            globalFS = checkpoint["proto"]
        except:
            for i in range(4):
                path = args.resume_path.split('model')[0]
                glbX_path = os.path.join(path, 'glb_x'+str(i+1)+'.npy')
                glbX = np.load(glbX_path)   # [3, 4, C]
                globalFS[scale_list[i]] = torch.from_numpy(glbX)
        
    ### test the model
    model.eval()
    with torch.no_grad():
        logging.info(' ########## test the model ########## ')
        test_dice_score = local_test(args, test_loader, model, args.device, 'BRATS2018', globalFS, mask, csv_writer, result_store_path)
        #test_dice_score = test_softmax_temp(test_loader, model, args.device, 'BRATS2018', mask)
        result_file.close()
        test_avg_dice = sum(test_dice_score)/len(test_dice_score)
        logging.info('--- Test Avg_Scores: {:.4f}, cls_Dice: {}'.format(test_avg_dice*100, test_dice_score))