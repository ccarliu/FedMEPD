datapath=/apdcephfs_cq10/share_1290796/lh/dataset/BRATS2018_Training_none_npy
dataname=BRATS2018


python3 -u fl_train_clsPasData_async.py --client_num 8 --pretrain 90 --gpus 1,2,3,0 --c_rounds 1000 --eval 30 --datapath ${datapath} --dataname ${dataname} --setting_options c8 --version brats18_rf_c8 --resume 0