import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', default='/apdcephfs_cq10/share_1290796/lh/dataset/BRATS2018_Training_none_npy', type=str)
    parser.add_argument('--dataname', default='BRATS2018', type=str)
    parser.add_argument('--chose_modal', default='t1', type=str)
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--save_root', default='./', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--visualize', default=True)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')

    # FL Settings
    parser.add_argument('--setting_options', default="c8", type=str)
    parser.add_argument('--gpus', default='1,2,3,4', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--c_rounds', type=int, default=300, help="number of rounds of training and communication")
    parser.add_argument('--start_round', type=int, default=0, help="number of rounds of training and communication")
    parser.add_argument('--round_per_train', type=int, default=100, help="number of rounds of training and communication")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--global_ep', type=int, default=1, help="the number of global epochs: E")
    parser.add_argument('--client_num', type=int, default=4, help="number of users: K")
    parser.add_argument('--pretrain', type=int, default=20, help="the number of local epochs: E")
    parser.add_argument('--eval', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--patience', default=10, type=int)

    # 说明
    parser.add_argument('--version', type=str, default='debug', help='to explain the experiment set up')

    # inference
    parser.add_argument('--fl', default=1, type = int)
    parser.add_argument('--resume_path', default="", type=str)
    parser.add_argument('--maskid', default=0,  type=int)

    args = parser.parse_args()
    return args