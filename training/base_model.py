import os, copy
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
# from tensorboardX import SummaryWriter

from utils.metrics import *

try:    
	from apex import amp
except ImportError:
	raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run with apex.")

import torch.multiprocessing as mp

def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad

def apply_scheduler(optimizer, lr_policy, num_epoch=None, total_num_epoch=None):
	if lr_policy == 'linear':
		# num_epoch with initial lr
		# rest of epoch linearly decrease to 0 (the last epoch is not 0)
		def lambda_rule(epoch):
			# lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)
			lr_l = 1.0 - max(0, epoch + 1 - num_epoch) / float(total_num_epoch - num_epoch + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	elif lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
	return scheduler

class base_model(nn.Module):
	def __init__(self, args):
		super(base_model, self).__init__()
		self.device = args.device
		self.is_train = args.is_train
		self.training_set_name = args.training_set_name
		self.testing_set_name = args.testing_set_name
		self.project_name = args.project_name
		self.exp_dir = args.exp_dir

		self.use_tensorboardX = False
		self.use_apex = True
		
		self.sampleSize = args.sampleSize # patch size for training the model. Default: [240, 320]
		self.H, self.W = self.sampleSize[0], self.sampleSize[1]
		self.batch_size = args.batch_size
		self.total_epoch_num = args.total_epoch_num # total number of epoch in training
		self.base_lr = args.base_lr # base learning rate

		# self.save_result_npy = True # whether to save intermediate results as npy array

	def _initialize_training(self):
		if self.project_name is not None:
			self.save_dir = os.path.join(self.exp_dir, self.project_name)
		else:
			self.project_name = self._get_project_name()
			self.project_name = self.project_name + '_' + self.training_set_name
			self.save_dir = os.path.join(self.exp_dir, self.project_name)

		project_name_info = 'project name: {}'.format(self.project_name)
		save_dir_info = 'save dir: {}'.format(self.save_dir)
		dataset_info = 'training:  {}\ntesting: {}'.format(self.training_set_name, self.testing_set_name)
		if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

		if self.is_train:
			today = datetime.datetime.today()
			self.train_log = os.path.join(self.save_dir, 'train_{}.log'.format(today.strftime('%m-%d-%Y')))
			
			fn = open(self.train_log, 'w')
			fn.write(project_name_info + '\n')
			fn.write(save_dir_info + '\n')
			fn.write(dataset_info + '\n')
			fn.close()

			self.evaluate_log = os.path.join(self.save_dir, 'evaluate_{}.log'.format(today.strftime('%m-%d-%Y')))
		else:
			self.evaluate_log = os.path.join(self.save_dir, 'evaluate_sep.log')

		print(project_name_info)
		print(save_dir_info)
		print(dataset_info)

		if self.use_tensorboardX:
			self.tensorboard_train_dir = os.path.join(self.save_dir, 'tensorboardX_train_logs')
			self.train_SummaryWriter = SummaryWriter(self.tensorboard_train_dir)
			self.tensorboard_eval_dir = os.path.join(self.save_dir, 'tensorboardX_eval_logs')
			self.eval_SummaryWriter = SummaryWriter(self.tensorboard_eval_dir)
			self.tensorboard_num_display_per_epoch = 5
			self.val_display_freq = 10

	def _initialize_networks(self):
		for name, model in self.model_dict.items():
			model.train().to(self.device)
			# init_weights(model, net_name=name, init_type='normal', init_gain=0.02)
			init_weights(model, net_name=name, init_type='normal', gain=0.02)

	def _get_scheduler(self, optim_type='linear'):
		'''
			if type is None -> all optim use default scheduler
			if types is str -> all optim use this types of scheduler
			if type is list -> each optim use their own scheduler
		'''
		self.scheduler_list = []
		if isinstance(optim_type, str):
			for name in self.optim_name:
				self.scheduler_list.append(apply_scheduler(getattr(self, name), lr_policy=optim_type, num_epoch=0.6*self.total_epoch_num, 
					total_num_epoch=self.total_epoch_num))
		elif isinstance(optim_type, list):
			for name, optim in zip(self.optim_name, optim_type):
				self.scheduler_list.append(apply_scheduler(getattr(self, name), lr_policy=optim, num_epoch=0.6*self.total_epoch_num, 
					total_num_epoch=self.total_epoch_num))
		else:
			raise RuntimeError("optim type should be either string or list!")

	def _init_apex(self, Num_losses):
		# self.model_list, self.optim_list = amp.initialize(self.model_list, self.optim_list, opt_level="O1", num_losses=Num_losses)
		model_list = []
		optim_list = []
		for m in self.model_name:
			model_list.append(getattr(self, m))
		for o in self.optim_name:
			optim_list.append(getattr(self, o))

		# model_list, optim_list = amp.initialize(model_list, optim_list, opt_level="O1", keep_batchnorm_fp32=True, num_losses=Num_losses)
		model_list, optim_list = amp.initialize(model_list, optim_list, opt_level="O1", num_losses=Num_losses)

	def _check_parallel(self):
		if torch.cuda.device_count() > 1:
			for name in self.model_name:
				setattr(self, name, nn.DataParallel(getattr(self, name)))

	def _check_distribute(self):
		if torch.cuda.device_count() > 1:
			# world size is number of process participat in the job
			# torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
			# mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
			if use_apex:
				setattr(self, name, apex.parallel.DistributedDataParallel(getattr(self, name)))
			else:
				for name in self.model_name:
					setattr(self, name, nn.DistributedDataParallel(getattr(self, name)))

	def _set_models_train(self, model_name):
		for name in model_name:
			getattr(self, name).train()

	def _set_models_eval(self, model_name):
		for name in model_name:
			getattr(self, name).eval()

	def _set_models_float(self, model_name):
		for name in model_name:
			for layers in getattr(self, name).modules():
				layers.float()

	def save_models(self, model_list, mode):
		'''
			mode include best, latest, or a number (epoch)
			save as non-dataparallel state_dict
		'''
		for model_name in model_list:
			if mode == 'latest':
				path_to_save_paramOnly = os.path.join(self.save_dir, 'latest_{}.pth'.format(model_name))
			elif mode == 'best':
				path_to_save_paramOnly = os.path.join(self.save_dir, 'best_{}.pth'.format(model_name))
			elif isinstance(mode, int):
				path_to_save_paramOnly = os.path.join(self.save_dir, 'epoch-{}_{}.pth'.format(str(mode), model_name))

			try:
				state_dict = getattr(self, model_name).module.state_dict()
			except AttributeError:
				state_dict = getattr(self, model_name).state_dict()

			model_weights = copy.deepcopy(state_dict)
			torch.save(model_weights, path_to_save_paramOnly)

	def _load_models(self, model_list, mode, isTrain=False, model_path=None):
		if model_path is None:
			model_path = self.save_dir

		for model_name in model_list:
			if mode == 'latest':
				path = os.path.join(model_path, 'latest_{}.pth'.format(model_name))
			elif mode == 'best':
				path = os.path.join(model_path, 'best_{}.pth'.format(model_name))
			elif isinstance(mode, int):
				path = os.path.join(model_path, 'epoch-{}_{}.pth'.format(str(mode), model_name))
			else:
				raise RuntimeError("Mode not implemented")

			state_dict = torch.load(path)

			try:
				getattr(self, model_name).load_state_dict(state_dict)
			except RuntimeError as e:
				print(e.message)
				# in the case of parallel model loading non-parallel state_dict || add module to all keys
				new_state_dict = OrderedDict()
				for k, v in state_dict.items():
					# print(k)
					# print('module.' + k)
					name = 'module.' + k # add `module.`
					new_state_dict[name] = v

				getattr(self, model_name).load_state_dict(new_state_dict)

			if isTrain:
				getattr(self, model_name).to(self.device).train()
			else:	
				getattr(self, model_name).to(self.device).eval()

	def _load_models_with_different_name(self, model_list, name_list, mode, isTrain=False, model_path=None):
		if model_path is None:
			model_path = self.save_dir

		for model_name, name_here in zip(model_list, name_list):
			if mode == 'latest':
				path = os.path.join(model_path, 'latest_{}.pth'.format(model_name))
			elif mode == 'best':
				path = os.path.join(model_path, 'best_{}.pth'.format(model_name))
			elif isinstance(mode, int):
				path = os.path.join(model_path, 'epoch-{}_{}.pth'.format(str(mode), model_name))
			else:
				raise RuntimeError("Mode not implemented")

			state_dict = torch.load(path)

			try:
				getattr(self, name_here).load_state_dict(state_dict)
			except RuntimeError:
				# in the case of parallel model loading non-parallel state_dict || add module to all keys
				new_state_dict = OrderedDict()
				for k, v in state_dict.items():
					# print(k)
					# print('module.' + k)
					name = 'module.' + k # add `module.`
					new_state_dict[name] = v

				getattr(self, name_here).load_state_dict(new_state_dict)

			if isTrain:
				getattr(self, name_here).to(self.device).train()
			else:	
				getattr(self, name_here).to(self.device).eval()

	def print_and_write_loss_summary(self, iterCount, totalCount, name_list, value_list, log_file):
		if len(name_list) != len(value_list):
			min_len = min(len(name_list), len(value_list))
			name_list = name_list[:min_len]
			value_list = value_list[:min_len]

		loss_summary = '\t{}/{}'.format(iterCount, totalCount)
		for loss_name, loss_value in zip(name_list, value_list):
			loss_summary += ' {}: {:.4f}'.format(loss_name, loss_value)

		fn = open(log_file, 'a')
		print(loss_summary)
		fn.write(loss_summary + '\n')
		fn.close()

	# def save_tensor2np(self, tensor, name, epoch, path=None):
	# 	if path == None:
	# 		path = self.save_dir
	# 	# expect a 4D tensor
	# 	# count = 0
	# 	# for i in range(tensor.size(0)):
	# 	generated_sample = tensor.detach().cpu().numpy()
	# 	# section = 'rgb'
	# 	# if not rgb:
	# 	# 	section = 'depth'
	# 	generated_sample_save_path = os.path.join(path, 'tensor2np', 'Epoch-%s_%s.npy' % (epoch, name))
	# 	if not os.path.exists(os.path.join(path, 'tensor2np')):
	# 		os.makedirs(os.path.join(path, 'tensor2np')) #  # ./expriments/project_name/generated_samples/val/section/1_depth_generated.npy

	# 	# print(img_target.dtype, img_target.shape) torch.float32 torch.Size([3, 240, 320])
	# 	np.save(generated_sample_save_path, generated_sample)

	def write_2_tensorboardX(self, writer, input_tensor, name, mode, count, nrow=None, normalize=True, value_range=(-1.0, 1.0)):
		# assume dict:{'name': data}
		if mode == 'image':
			if not nrow:
				raise RuntimeError('tensorboardX: must specify number of rows in image mode')
			grid = make_grid(input_tensor, nrow=nrow, normalize=normalize, range=value_range)
			writer.add_image(name, grid, count)
		elif mode == 'scalar':
			if isinstance(input_tensor, list) and isinstance(name, list):
				assert len(input_tensor) == len(name)
				for n, t in zip(name, input_tensor):
					writer.add_scalar(n, t, count)
			else:
				writer.add_scalar(name, input_tensor, count)
		else:
			raise RuntimeError('tensorboardX: this mode is not yet implemented')