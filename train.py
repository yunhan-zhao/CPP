import os, sys
import random, time, copy
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

# ======================= dataloaders ======================
from dataloader.CAM_interiorNet_depth_dataLoader import CAM_interiorNet_depth_dataLoader
from dataloader.CAM_ScanNet_depth_dataLoader import CAM_ScanNet_depth_dataLoader

# ================================================ comment / uncomment to choose the training model ============================================================= #
from training.CPP_training import CPP_training_model as train_model # CPP only
# from training.PDA_training import PDA_training_model as train_model # PDA only
# from training.CPP_PDA_joint_training import CPP_PDA_joint_training_model as train_model # PDA + CPP

import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print('pytorch version: {}'.format(torch.__version__))

################## set attributes for this project/experiment ##################

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=os.path.join(os.getcwd(), 'experiments'),
							 	help='place to store all experiments')
parser.add_argument('--project_name', type=str, help='Test Project')
parser.add_argument('--data_root', type=str, default='/home/yunhaz5/project/CAM/dataset/InteriorNet',
							 	help='absolute path to dir of all datasets')
parser.add_argument('--training_set_name', type=str, default='interiorNet_training_natural_10800',
							 	help='which dataset to use as training set')
parser.add_argument('--testing_set_name', type=str, default='interiorNet_testing_natural_1080',
							 	help='which dataset to use as testing set')
parser.add_argument('--is_train', action='store_true', help='whether this is training phase')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--sampleSize', type=list, default=[240, 320] , help='size of samples in experiments')
parser.add_argument('--total_epoch_num', type=int, default=200, help='total number of epoch')
parser.add_argument('--device', type=str, default='cpu', help='whether running on gpu')
parser.add_argument('--base_lr', type=int, default=0.001, help='basic learning rate')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataLoaders')
parser.add_argument('--eval_mode', type=int, default=-1, help='eval epoch')

args = parser.parse_args()

if torch.cuda.is_available(): 
	args.device='cuda'
	torch.cuda.empty_cache()

if 'interiorNet' in args.training_set_name:
	training_dataset = CAM_interiorNet_depth_dataLoader(root_dir=args.data_root, set_name=args.training_set_name, size=args.sampleSize)

elif 'ScanNet' in args.training_set_name:
	training_dataset = CAM_ScanNet_depth_dataLoader(root_dir=args.data_root, set_name=args.training_set_name, size=args.sampleSize)

else:
	raise RuntimeError('only support following training datasets: interiorNet and ScanNet!')

training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

if 'interiorNet' in args.testing_set_name:
	testing_dataset = CAM_interiorNet_depth_dataLoader(root_dir=args.data_root, set_name=args.testing_set_name, size=args.sampleSize)

elif 'ScanNet' in args.testing_set_name:
	if 'ScanNet' not in args.data_root:
		# for the case where we want to train on InteriorNet while test on ScanNet
		ScanNet_data_root = args.data_root[:-11] + 'ScanNet'
	else:
		ScanNet_data_root = args.data_root
	testing_dataset = CAM_ScanNet_depth_dataLoader(root_dir=ScanNet_data_root, set_name=args.testing_set_name, size=args.sampleSize)

else:
	raise RuntimeError('only support following testing datasets: interiorNet and ScanNet!')

testing_dataloader = DataLoader(testing_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
model = train_model(args, training_dataloader, testing_dataloader)

if args.is_train:
	model.train()
else:
	if args.eval_mode == -1:
		args.eval_mode = 'best'
	model.evaluate(mode=args.eval_mode)