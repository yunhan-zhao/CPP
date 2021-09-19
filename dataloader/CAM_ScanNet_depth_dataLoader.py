import os, sys
import random, time, copy
from skimage import io, transform
import numpy as np
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import cv2

import skimage.transform 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms


IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.bin'
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class CAM_ScanNet_depth_dataLoader(Dataset):
	def __init__(self, root_dir, set_name, size=[240, 320], downsampleDepthFactor=1, surface_normal=False, augmentation=True):
		self.root_dir = root_dir
		self.size = size
		self.MIN_DEPTH_CLIP = 1.0
		self.MAX_DEPTH_CLIP = 10.0

		self.set_name = set_name # e.g., ScanNet_all_uniform_1080/
		self.include_surface_normal = surface_normal
		self.set_len = 0
		self.path2rgbFiles = []
		self.downsampleDepthFactor = downsampleDepthFactor
		self.augmentation = augmentation # whether to augment each batch data
		self.extrinsic_angle = 'radian' # radian or degree

		self.return_keys = ['rgb', 'depth', 'intrinsic', 'extrinsic', 'augmentation']
		if self.include_surface_normal:
			self.return_keys.append('surface_normal')
		self.return_values = []

		self.color_original_size = (968, 1296)
		
		rgbFileNameList = os.listdir(os.path.join(self.root_dir, self.set_name, 'rgb'))
		for fName in rgbFileNameList:
			if is_image_file(fName):
				path = os.path.join(self.root_dir, self.set_name, 'rgb', fName)
				self.path2rgbFiles.append(path)

		self.set_len = len(self.path2rgbFiles)

		# read cam parameter file
		camFileName = os.path.join(self.root_dir, self.set_name, 'cam_parameter.txt')
		self.extrinsicParaDict = {}
		self.intrinsicParaDict = {}
		with open(camFileName, 'r') as f:
			for l in f:
				fileName, intrinsic, extrinsic = l.rstrip('\n').split('|')
				p_x, p_y, p_z, roll, pitch, yaw = extrinsic.split(' ')
				intrinsic_list = intrinsic.split(' ')
				f_x, p_ptx, f_y, p_pty = intrinsic_list[0], intrinsic_list[2], intrinsic_list[5], intrinsic_list[6]

				self.intrinsicParaDict[fileName] = np.array((float(f_x), float(f_y), float(p_ptx), float(p_pty)))

				# being consistent with interiorNet dataLoader
				if self.extrinsic_angle == 'degree':
					self.extrinsicParaDict[fileName] = np.array((float(p_x), float(p_y), float(p_z), float(pitch), float(roll), float(yaw)))
				elif self.extrinsic_angle == 'radian':
					self.extrinsicParaDict[fileName] = np.array((float(p_x), float(p_y), float(p_z), np.deg2rad(float(pitch)), np.deg2rad(float(roll)),
																np.deg2rad(float(yaw))))
				else:
					raise RuntimeError('choose angle representation between radian or degree')

		self.TF2tensor = transforms.ToTensor()
		self.TF2PIL = transforms.ToPILImage()
		self.TFNormalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		self.funcResizeTensor = nn.Upsample(size=self.size, mode='nearest', align_corners=None)
		self.funcResizeDepth = nn.Upsample(size=[int(self.size[0]*self.downsampleDepthFactor),
												 int(self.size[1]*self.downsampleDepthFactor)], 
												 mode='nearest', align_corners=None)
		
	def __len__(self):
		return self.set_len
	
	def __getitem__(self, idx):
		rgbFileName = self.path2rgbFiles[idx]
		return_dict = {}
		return_dict.fromkeys(self.return_keys)
		return_dict = self.fetch_img_and_corresponding_labels(rgbFileName, return_dict)
		return_dict = self.fetch_corresponding_cam_parameters(rgbFileName, return_dict)

		return return_dict

	def get_dataset_name(self):
		return self.set_name

	def distance_2_depth(self, distance_map):
		H, W = distance_map.shape[0], distance_map.shape[1]
		y_grid, x_grid = np.mgrid[0:H, 0:W]
		y_vector, x_vector = y_grid.astype(np.float32).reshape(1, H*W), x_grid.astype(np.float32).reshape(1, H*W)
		
		y = (y_vector - self.original_p_pt[0]) / self.original_focal_length
		x = (x_vector - self.original_p_pt[1]) / self.original_focal_length

		depth_map = distance_map.flatten() / np.sqrt(x**2 + y**2 + 1)
		depth_map = depth_map.reshape(H, W)
		
		return depth_map

	def fetch_img_and_corresponding_labels(self, rgbFileName, return_dict):
		if 'training' in self.set_name and self.augmentation:
			if np.random.random(1) > 0.5:
				augmentation = True
			else:
				augmentation = False
		else:
			augmentation = False
		return_dict['augmentation'] = augmentation

		image = PIL.Image.open(rgbFileName).convert('RGB')
		image = np.array(image, dtype=np.float32) / 255.

		if augmentation:
			image = np.fliplr(image).copy()
		imageT = self.TF2tensor(image)
		try:
			imageT = self.TFNormalize(imageT)
		except RuntimeError:
			print('image shape missmatch error')
			print(rgbFileName)			
		imageT = imageT.unsqueeze(0) # need 4D data to resize tensor
		imageT = self.funcResizeTensor(imageT)
		imageT = imageT.squeeze(0)

		return_dict['rgb'] = imageT

		fileName = rgbFileName.split('/')[-1].split('.')[0]
		depthFileName = os.path.join(self.root_dir, self.set_name, 'depth', fileName + '.pgm')
		depth = cv2.imread(depthFileName, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.
		depth = np.expand_dims(depth, 2)
		if augmentation:
			depth = np.fliplr(depth).copy()
		depthT = self.TF2tensor(depth)
		depthT = self.preprocess_depth(depthT, mode='tanh')
		depthT = depthT.unsqueeze(0) # need 4D data to resize tensor
		depthT = self.funcResizeTensor(depthT)
		depthT = depthT.squeeze(0)

		return_dict['depth'] = depthT

		if self.include_surface_normal:
			normalFileName = os.path.join(self.root_dir, self.set_name, 'surface_normal', fileName)
			normal = PIL.Image.open(normalFileName)
			normal = np.array(normal, dtype=np.float32) # shape (H, W, 3), [0, 255]
			if augmentation:
				normal = np.fliplr(normal).copy()
			normalT = self.TF2tensor(normal)
			return_dict['surface_normal'] = normalT

		return return_dict

	def fetch_corresponding_cam_parameters(self, rgbFileName, return_dict):
		fileName = rgbFileName.split('/')[-1].split('.')[0]

		return_dict['extrinsic'] = self.extrinsicParaDict[fileName]
		original_intrinsics = self.intrinsicParaDict[fileName]

		updated_intrinsic = self.updated_intrinsic_parameters(original_intrinsics)
		return_dict['intrinsic'] = updated_intrinsic

		return return_dict

	def updated_intrinsic_parameters(self, original_intrinsics):
		x_scale = self.color_original_size[1] / self.size[1]
		y_scale = self.color_original_size[0] / self.size[0]

		updated_f_x = original_intrinsics[0] / x_scale
		updated_f_y = original_intrinsics[1] / y_scale

		p_ptx = int(original_intrinsics[2] / x_scale)
		p_pty = int(original_intrinsics[3] / y_scale)

		return np.array((updated_f_x, updated_f_y, p_ptx, p_pty), dtype=np.float32)

	def preprocess_depth(self, depthT, mode='tanh'):
		'''
			preprocess depth tensor before feed into the network
			mode: choose from depth [0, max_depth], disparity [0, 1], tanh [-1.0, 1.0] 
		'''
		if 'training' in self.set_name:
			if mode == 'tanh':
				return (((depthT - self.MIN_DEPTH_CLIP) / (self.MAX_DEPTH_CLIP - self.MIN_DEPTH_CLIP)) - 0.5) * 2.0 # mask out depth over 
			elif mode == 'depth':
				return depthT
		else:
			return depthT