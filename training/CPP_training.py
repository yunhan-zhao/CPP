import os, time, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid

from models.T2Net import _UNetGenerator, init_weights

from utils.metrics import *
from utils.cpp_encoding import get_extrinsic_channel

from training.base_model import set_requires_grad, base_model

try:    
	from apex import amp
except ImportError:
	raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run with apex.")

class CPP_training_model(base_model):
	def __init__(self, args, training_dataloader, testing_dataloader):
		super(CPP_training_model, self).__init__(args)
		self._initialize_training()

		self.training_dataloader = training_dataloader
		self.testing_dataloader = testing_dataloader
		self.MIN_DEPTH_CLIP = 1.0
		self.MAX_DEPTH_CLIP = 10.0

		self.EVAL_DEPTH_MIN = 1.0
		self.EVAL_DEPTH_MAX = 10.0

		self.CEILING_HEIGHT = 3.0

		self.save_evaluate_steps = 10

		self.depthEstModel = _UNetGenerator(input_nc = 4, output_nc = 1)
		self.model_name = ['depthEstModel']

		if self.is_train:
			self.depth_optimizer = optim.Adam(self.depthEstModel.parameters(), lr=self.base_lr, betas=(0.5, 0.999))
			self.optim_name = ['depth_optimizer']
			self._get_scheduler()
			self.L1Loss = nn.L1Loss()
			self._initialize_networks(['depthEstModel'])

			# apex can only be applied to CUDA models
			if self.use_apex:
				self._init_apex(Num_losses=2)

		self.EVAL_best_loss = float('inf')
		self.EVAL_best_model_epoch = 0
		self.EVAL_all_results = {}

		self._check_parallel()

	def _get_project_name(self):
		return 'CPP_training_model'

	def _initialize_networks(self, model_name):
		for name in model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

	def compute_depth_loss(self, rgb, gt_depth, mask=None):
		predicted_depth = self.depthEstModel(rgb.detach())[-1]

		if mask is not None:
			loss = self.L1Loss(predicted_depth[mask], gt_depth[mask])

		else:
			loss = self.L1Loss(predicted_depth, gt_depth)
		
		return loss

	def train(self):
		phase = 'train'
		since = time.time()
		best_loss = float('inf')

		self.train_display_freq = len(self.training_dataloader)

		tensorboardX_iter_count = 0
		for epoch in range(self.total_epoch_num):
			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()
			
			self._set_models_train(['depthEstModel'])
			
			# Iterate over data.
			iterCount = 0

			for sample_dict in self.training_dataloader:
				imageTensor, depthGTTensor = sample_dict['rgb'], sample_dict['depth']
				extrinsic_para = sample_dict['extrinsic'].float() # otherwise mismatch data type double and float
				if "intrinsic" in sample_dict.keys():
					# for ScanNet only
					intrinsic_para = sample_dict['intrinsic'].float() # fx, fy, px, py
					focal_length = intrinsic_para[:, :2]
					p_pt = intrinsic_para[:, 2:]
				else:
					# for interiorNet
					focal_length = 300
					p_pt = (120, 160)
					
				extrinsic_channel = get_extrinsic_channel(imageTensor, focal_length, p_pt, extrinsic_para, self.CEILING_HEIGHT, augmentation=sample_dict['augmentation'])

				imageTensor_C = torch.cat((imageTensor, extrinsic_channel), dim=1)
				imageTensor_C = imageTensor_C.to(self.device)
				depthGTTensor = depthGTTensor.to(self.device) # [B_size, 1, 240, 320]
				valid_mask = (depthGTTensor >= -1.) & (depthGTTensor <= 1.)

				with torch.set_grad_enabled(phase=='train'):

					total_loss = 0.
					#############  train the depthEstimator
					self.depth_optimizer.zero_grad()
					depth_loss = self.compute_depth_loss(imageTensor_C, depthGTTensor, valid_mask)
					total_loss += depth_loss

					if self.use_apex:
						with amp.scale_loss(total_loss, self.depth_optimizer) as total_loss_scaled:
							total_loss_scaled.backward()
					else:
						total_loss.backward()

					self.depth_optimizer.step()

				iterCount += 1
				if iterCount % 20 == 0:
					loss_name = ['total_loss', 'depth_loss']
					loss_value = [total_loss, depth_loss]
					self.print_and_write_loss_summary(iterCount, len(self.training_dataloader), loss_name, loss_value, self.train_log)

			# take step in optimizer
			for scheduler in self.scheduler_list:
				scheduler.step()
				for optim in self.optim_name:				
					lr = getattr(self, optim).param_groups[0]['lr']
					lr_update = 'Epoch {}/{} finished: {} learning rate = {:7f}'.format(epoch+1, self.total_epoch_num, optim, lr)
					print(lr_update)
					
					fn = open(self.train_log,'a')
					fn.write(lr_update + '\n')
					fn.close()

			if (epoch+1) % self.save_evaluate_steps == 0:
				self.save_models(self.model_name, mode=epoch+1)
				_ = self.evaluate(epoch+1)

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		
		fn = open(self.train_log,'a')
		fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
		fn.close()

		best_model_summary = 'Overall best model is epoch {}'.format(self.EVAL_best_model_epoch)
		print(best_model_summary)
		print(self.EVAL_all_results[str(self.EVAL_best_model_epoch)])
		fn = open(self.evaluate_log, 'a')
		fn.write(best_model_summary + '\n')
		fn.write(self.EVAL_all_results[str(self.EVAL_best_model_epoch)] + '\n')
		fn.close()

	def evaluate(self, mode):
		'''
			mode choose from <int> or best
			<int> is the number of epoch, represents the number of epoch, used for in training evaluation
			'best' is used for after training mode
		'''

		set_name = 'test'
		eval_model_list = ['depthEstModel']

		if isinstance(mode, int) and self.is_train:
			self._set_models_eval(eval_model_list)
			if self.EVAL_best_loss == float('inf'):
				fn = open(self.evaluate_log, 'w')
			else:
				fn = open(self.evaluate_log, 'a')

			fn.write('Evaluating with mode: {} | dataset: {} \n'.format(mode, self.testing_set_name))
			fn.write('\tEvaluation range min: {} | max: {} \n'.format(self.EVAL_DEPTH_MIN, self.EVAL_DEPTH_MAX))
			fn.close()

		else:
			self._load_models(eval_model_list, mode)

		print('Evaluating with mode: {} | dataset: {}'.format(mode, self.testing_set_name))
		print('\tEvaluation range min: {} | max: {}'.format(self.EVAL_DEPTH_MIN, self.EVAL_DEPTH_MAX))

		total_loss = 0.
		count = 0

		predTensor = torch.zeros((1, 1, self.H, self.W)).to('cpu')
		grndTensor = torch.zeros((1, 1, self.H, self.W)).to('cpu')
		imgTensor = torch.zeros((1, 3, self.H, self.W)).to('cpu')
		extTensor = torch.zeros((1, 6)).to('cpu')
		idx = 0

		with torch.no_grad():
			for sample_dict in self.testing_dataloader:
				imageTensor, depthGTTensor = sample_dict['rgb'], sample_dict['depth']
				extrinsic_para = sample_dict['extrinsic'].float() # otherwise mismatch data type double and float

				if "intrinsic" in sample_dict.keys():
					# for ScanNet only
					intrinsic_para = sample_dict['intrinsic'].float() # fx, fy, px, py
					focal_length = intrinsic_para[:, :2]
					p_pt = intrinsic_para[:, 2:]
				else:
					# for interiorNet
					focal_length = 300
					p_pt = (120, 160)

				extrinsic_channel = get_extrinsic_channel(imageTensor, focal_length, p_pt, extrinsic_para, self.CEILING_HEIGHT)
				imageTensor_C = torch.cat((imageTensor, extrinsic_channel), dim=1)
				valid_mask = np.logical_and(depthGTTensor >= self.EVAL_DEPTH_MIN, depthGTTensor <= self.EVAL_DEPTH_MAX)

				idx += imageTensor.shape[0]
				print('epoch {}: have processed {} number samples in {} set'.format(mode, str(idx), set_name))
				imageTensor_C = imageTensor_C.to(self.device)
				depthGTTensor = depthGTTensor.to(self.device)	# real depth

				if self.is_train and self.use_apex:
					with amp.disable_casts():
						predDepth = self.depthEstModel(imageTensor_C)[-1].detach().to('cpu')
				else:
					predDepth = self.depthEstModel(imageTensor_C)[-1].detach().to('cpu')

				# recover real depth
				predDepth = ((predDepth + 1.0) * 0.5 * (self.MAX_DEPTH_CLIP - self.MIN_DEPTH_CLIP)) + self.MIN_DEPTH_CLIP

				depthGTTensor = depthGTTensor.detach().to('cpu')
				predTensor = torch.cat((predTensor, predDepth), dim=0)
				grndTensor = torch.cat((grndTensor, depthGTTensor), dim=0)
				imgTensor = torch.cat((imgTensor, imageTensor.to('cpu')), dim=0)
				extTensor = torch.cat((extTensor, extrinsic_para), dim=0)

				if isinstance(mode, int) and self.is_train:
					eval_depth_loss = self.L1Loss(predDepth[valid_mask], depthGTTensor[valid_mask])
					total_loss += eval_depth_loss.detach().cpu()

				count += 1

			if isinstance(mode, int) and self.is_train:
				validation_loss = (total_loss / count)

			results_nyu = Result(mask_min=self.EVAL_DEPTH_MIN, mask_max=self.EVAL_DEPTH_MAX)
			results_nyu.evaluate(predTensor[1:], grndTensor[1:])
			individual_results = results_nyu.individual_results(predTensor[1:], grndTensor[1:])

			result1 = '\tabs_rel:{:.3f}, sq_rel:{:.3f}, rmse:{:.3f}, rmse_log:{:.3f}, mae:{:.3f} '.format(
					results_nyu.absrel,results_nyu.sqrel,results_nyu.rmse,results_nyu.rmselog,results_nyu.mae)
			result2 = '\t[<1.25]:{:.3f}, [<1.25^2]:{:.3f}, [<1.25^3]::{:.3f}'.format(results_nyu.delta1,results_nyu.delta2,results_nyu.delta3)

			print(result1)
			print(result2)

			if isinstance(mode, int) and self.is_train:
				self.EVAL_all_results[str(mode)] = result1 + '\t' + result2

				if validation_loss.item() < self.EVAL_best_loss:
					self.EVAL_best_loss = validation_loss.item()
					self.EVAL_best_model_epoch = mode
					self.save_models(self.model_name, mode='best')

				best_model_summary = '\tCurrent eval loss {:.4f}, current best loss {:.4f}, current best model {}\n'.format(validation_loss.item(), self.EVAL_best_loss, self.EVAL_best_model_epoch)
				print(best_model_summary)

				fn = open(self.evaluate_log, 'a')
				fn.write(result1 + '\n')
				fn.write(result2 + '\n')
				fn.write(best_model_summary + '\n')
				fn.close()

			return_dict = {}
			return_dict['rgb'] = imgTensor[1:]
			return_dict['depth_pred'] = predTensor[1:]
			return_dict['depth_gt'] = grndTensor[1:]
			return_dict['extrinsic'] = extTensor[1:]
			return_dict['ind_results'] = individual_results
			
			return return_dict