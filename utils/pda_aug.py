import numpy as np
import torch
import torch.nn.functional as F

def _get_intrinsic_matrix(focal_length, p_pt, batch_size):
	K = torch.zeros((batch_size, 9), dtype=torch.float32)
	K[:, -1] = 1.
	if isinstance(focal_length, (int, float)):
		# suggest fx = fy for all samples
		K[:, 0] = focal_length
		K[:, 4] = focal_length
	elif isinstance(focal_length, (list, tuple)):
		# suggest fx, fy for all samples
		K[:, 0] = focal_length[0]
		K[:, 4] = focal_length[1]
	elif torch.is_tensor(focal_length):
		if focal_length.dim() == 1:
			# suggest fx = fy for indivdual sample
			K[:, 0] = focal_length
			K[:, 4] = focal_length
		elif focal_length.dim() == 2:
			# suggest fx, fy for indivdual sample
			K[:, 0] = focal_length[:, 0]
			K[:, 4] = focal_length[:, 1]
		else:
			raise ValueError('focal length tensor has to have shape of [B, ] or [B, 2]')
	else:
		raise ValueError('focal length variable should be either int/float, list/tuple or tensor of size [B, ]/[B, 2]')

	if isinstance(p_pt, (list, tuple)):
		K[:, 2] = p_pt[1]
		K[:, 5] = p_pt[0]
	elif torch.is_tensor(p_pt):
		assert p_pt.dim() == 2
		K[:, 2] = p_pt[:, 1]
		K[:, 5] = p_pt[:, 0]
	else:
		raise ValueError('principle point variable should be either list/tuple or tensor of size [B, 2]')
	return K.reshape(batch_size, 3, 3)

def _get_inverse_intrinsic_matrix(K):
	'''
		K is a tensor with shape [B, 3, 3]
	'''
	K_inv = torch.zeros_like(K, dtype=torch.float32)
	K_inv[:, 0, 0] = 1. / K[:, 0, 0]
	K_inv[:, 1, 1] = 1. / K[:, 1, 1]
	K_inv[:, 0, 2] = -K[:, 0, 2] / K[:, 0, 0]
	K_inv[:, 1, 2] = -K[:, 1, 2] / K[:, 1, 1]
	K_inv[:, -1, -1] = 1.
	return K_inv

def _convert_depth_for_projection(depthGT, dataset, method, is_train=True, MAX_DEPTH_CLIP_GIVEN=None, MIN_DEPTH_CLIP_GIVEN=None):
	'''
		default setting: dataset: interiorNet, method: vanilla
	'''
	# setting MAX_DEPTH_CLIP, MIN_DEPTH_CLIP
	if dataset == 'interiorNet':
		MAX_DEPTH_CLIP=10.0
		MIN_DEPTH_CLIP=1.0
	elif dataset == 'ScanNet':
		MAX_DEPTH_CLIP=10.0
		MIN_DEPTH_CLIP=1.0
	else:
		raise RuntimeError('Current only support interiorNet | ScanNet!')

	# overwrite if MAX_DEPTH_CLIP or MIN_DEPTH_CLIP is given
	if MAX_DEPTH_CLIP_GIVEN is not None:
		MAX_DEPTH_CLIP = MAX_DEPTH_CLIP_GIVEN
	if MIN_DEPTH_CLIP_GIVEN is not None:
		MIN_DEPTH_CLIP = MIN_DEPTH_CLIP_GIVEN

	if method == 'vanilla':
		depthGT = ((depthGT * 0.5) + 0.5) * (MAX_DEPTH_CLIP - MIN_DEPTH_CLIP) + MIN_DEPTH_CLIP
	else:
		raise RuntimeError('Current only support vanilla')

	return depthGT

def _rescale_depth_for_training(depthGT, dataset, method, is_train=True, MAX_DEPTH_CLIP_GIVEN=None, MIN_DEPTH_CLIP_GIVEN=None):
	# setting MAX_DEPTH_CLIP, MIN_DEPTH_CLIP
	if dataset == 'interiorNet':
		MAX_DEPTH_CLIP=10.0
		MIN_DEPTH_CLIP=1.0
	elif dataset == 'ScanNet':
		MAX_DEPTH_CLIP=10.0
		MIN_DEPTH_CLIP=1.0
	else:
		raise RuntimeError('Current only support interiorNet | ScanNet!')

	# overwrite if MAX_DEPTH_CLIP or MIN_DEPTH_CLIP is given
	if MAX_DEPTH_CLIP_GIVEN is not None:
		MAX_DEPTH_CLIP = MAX_DEPTH_CLIP_GIVEN
	if MIN_DEPTH_CLIP_GIVEN is not None:
		MIN_DEPTH_CLIP = MIN_DEPTH_CLIP_GIVEN

	if method == 'vanilla':
		depthGT = (((depthGT - MIN_DEPTH_CLIP) / (MAX_DEPTH_CLIP - MIN_DEPTH_CLIP)) - 0.5) * 2.0
	else:
		raise RuntimeError('Current only support vanilla!')

	return depthGT

def _compute_distance_map(pc, H, W):
	B = pc.shape[0]
	return torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2 + pc[:, 2]**2).reshape(B, 1, H, W)

def _distance_2_depth(distance_map, K_inv):
	B, H, W = distance_map.shape[0], distance_map.shape[2], distance_map.shape[3]
	
	grid_y, grid_x = np.mgrid[0:H, 0:W]
	grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
	q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(0).expand(B, 3, H*W)

	pc = torch.bmm(K_inv, q)

	denom = torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2 + 1) # [B, N]
	depth_map = distance_map.reshape(B, -1) / denom
	depth_map = depth_map.reshape(B, 1, H, W)

	return depth_map

def warp_image_depth_with_pose_augmentation(image, depthGT, pose, focal_length, p_pt, training_dataset_name,
		pose_perturbed=None, MAX_DEPTH_CLIP=10.0, MIN_DEPTH_CLIP=1.0, method='vanilla',
		is_train=True, include_depth_warp=True, augmentation=None, pose_sample_mode='uniform'):
	'''
		depthGT is the depth tensor with shape [B, 1, H, W] -- need real depth !
	'''
	return_dict = {}
	# first flip back augmented sample since the pose is still for the original sample
	if augmentation is not None:
		image[augmentation] = torch.flip(image[augmentation], [1, 3])
		depthGT[augmentation] = torch.flip(depthGT[augmentation], [1, 3])

	dataset = training_dataset_name.split('_')[0]
	# depth is preprocess in the range [-1., 1.], we need abs scale depth for reprojection
	depthGT = _convert_depth_for_projection(depthGT, dataset, method, is_train)
	depthGT[depthGT < 1e-6] = 1e6 # to filter close to 0 depth

	B, H, W = image.shape[0], image.shape[2], image.shape[3]
	K = _get_intrinsic_matrix(focal_length, p_pt, B)
	K_inv = _get_inverse_intrinsic_matrix(K)

	translation_sampler, euler_sampler = sample_pose_perturbance(pose, dataset, mode=pose_sample_mode)
	pc = image_to_pointcloud(depthGT, K_inv, homogeneous_coord=True)
	dist_map = _compute_distance_map(pc, H, W)

	if pose_perturbed is None:
		# random perturb pose if the new pose is not given 
		pose_perturbed = pose.clone()
		pose_perturbed[:, :3] += translation_sampler.squeeze(1)
		pose_perturbed[:, 3:] += euler_sampler.squeeze(1)

	pc_perturbed_depth, pc_perturbed_image  = _transform_pc_with_poses(pc, pose, pose_perturbed)
	pixel_coords_perturbed = pointcloud_to_pixel_coords(pc_perturbed_image, K, image)
	image_warped = F.grid_sample(image, pixel_coords_perturbed, padding_mode="border")
	return_dict['image_warped'] = image_warped
	if include_depth_warp:
		# depth_warped is the augmented depth we used in all training
		dist_warped = F.grid_sample(dist_map, pixel_coords_perturbed)
		depth_warped = _distance_2_depth(dist_warped, K_inv) # no translation in our case, convert depth to distance to get rid of grid artifacts
		depth_warped[depth_warped > 1e3] = 0.

	depth_reproject = pointcloud_to_depth_maps(pc_perturbed_depth, K, image)
	depth_reproject[depth_reproject > 1e3] = 0.

	# now convert the reproject depth back to [-1., 1.] to continue training
	depth_reproject = _rescale_depth_for_training(depth_reproject, dataset, method, is_train)
	if include_depth_warp:
		depth_warped = _rescale_depth_for_training(depth_warped, dataset, method, is_train)
		return_dict['depth_warped'] = depth_warped

	return_dict['depth_reproject'] = depth_reproject
	return_dict['pose_perturbed'] = pose_perturbed

	# make lrflip for augmented samples
	if augmentation is not None:
		return_dict['image_warped'][augmentation] = torch.flip(return_dict['image_warped'][augmentation], [1, 3])
		return_dict['depth_reproject'][augmentation] = torch.flip(return_dict['depth_reproject'][augmentation], [1, 3])
		if include_depth_warp:
			return_dict['depth_warped'][augmentation] = torch.flip(return_dict['depth_warped'][augmentation], [1, 3])

	return return_dict

def sample_pose_perturbance(pose, dataset, upper=0.1, lower=-0.1, mode='uniform'):
	'''
		input pose: tensor with shape [B, 6]
		return pose_perturbed: pose with shape [B, 6]
	'''
	B = pose.shape[0]
	if mode == 'uniform':
		translation_sampler = torch.rand((B, 1, 3)) * 0.
		euler_sampler = torch.zeros((B, 1, 3), dtype=torch.float32).uniform_(lower, upper)
	# add more sampling mode as you like~
	return translation_sampler, euler_sampler

def image_to_pointcloud(depth, K_inv, homogeneous_coord=False):
	assert depth.dim() == 4
	assert depth.size(1) == 1

	B, H, W = depth.shape[0], depth.shape[2], depth.shape[3]
	depth_v = depth.reshape(B, 1, -1)

	grid_y, grid_x = np.mgrid[0:H, 0:W]
	grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
	q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(0).expand(B, 3, H*W)

	pc = torch.bmm(K_inv, q) * depth_v
	if homogeneous_coord:
		pc = torch.cat((pc, torch.ones((B, 1, depth_v.shape[-1]), dtype=pc.dtype)), dim=1)
	return pc

def pointcloud_to_pixel_coords(pc, K, image, normalization=True, eps=1e-8):
	B, H, W = image.shape[0], image.shape[2], image.shape[3]
	pc = pc[:, :3, :]
	pc = pc / (pc[:, -1, :].unsqueeze(1) + eps)
	p_coords = torch.bmm(K, pc)
	p_coords = p_coords[:, :2, :]
	if normalization:
		p_coords_n = torch.zeros_like(p_coords, dtype=torch.float32)
		p_coords_n[:, 0, :] = p_coords[:, 0, :] / (W - 1.)
		p_coords_n[:, 1, :] = p_coords[:, 1, :] / (H - 1.)
		p_coords_n = (p_coords_n - 0.5) * 2.
		u_proj_mask = ((p_coords_n[:, 0, :] > 1) + (p_coords_n[:, 0, :] < -1))
		p_coords_n[:, 0, :][u_proj_mask] = 2
		v_proj_mask = ((p_coords_n[:, 1, :] > 1) + (p_coords_n[:, 1, :] < -1))
		p_coords_n[:, 1, :][v_proj_mask] = 2

		p_coords_n = p_coords_n.reshape(B, 2, H, W).permute(0, 2, 3, 1)
		return p_coords_n
	else:
		return p_coords

def pointcloud_to_depth_maps(pc, K, image, eps=1e-7):
	B, H, W = image.shape[0], image.shape[2], image.shape[3]
	pc = pc[:, :3, :]
	zt = pc[:, 2, :]
	pc = pc / (pc[:, -1, :].unsqueeze(1) + eps)
	p_coords = torch.bmm(K, pc)
	p_coords = p_coords[:, :2, :]
	xt, yt = p_coords[:, 0, :].type(torch.long), p_coords[:, 1, :].type(torch.long)
	keep = (yt < H) & (yt >= 0) & (xt < W) & (xt >= 0)
	depth_map = torch.zeros((B, H, W), dtype=torch.float32)
	depth_map[:, yt[keep], xt[keep]] = zt[keep]
	return depth_map.unsqueeze(1)

def _transform_pc_with_poses(pc, pose1, pose2):
	B = pose1.shape[0]

	R_1 = torch.bmm(Rotz(pose1[:, 5]), torch.bmm(Roty(pose1[:, 4]), Rotx(pose1[:, 3])))
	t_1 = pose1[:, :3].reshape(-1, 3, 1)


	R_2 = torch.bmm(Rotz(pose2[:, 5]), torch.bmm(Roty(pose2[:, 4]), Rotx(pose2[:, 3])))
	t_2 = pose2[:, :3].reshape(-1, 3, 1)
	
	R0 = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
	R0[:, 0, 0] = -1 # handedness is different than our camera model 
	R_1 = torch.bmm(R_1, R0)
	R_2 = torch.bmm(R_2, R0)

	cam_coord = pc[:, :3, :]

	cam_coord_depth = R_2.transpose(1, 2)@R_1@cam_coord + R_2.transpose(1, 2)@(t_2-t_1)
	cam_coord_rgb = R_1.transpose(1, 2)@R_2@cam_coord + R_1.transpose(1, 2)@(t_1-t_2)
	
	return cam_coord_depth, cam_coord_rgb

def Rotx(t):
	"""
		Rotation about the x-axis.
		np.array([[1,  0,  0], [0,  c, -s], [0,  s,  c]])

		-- input t shape B x 1
		-- return B x 3 x 3
	"""
	B = t.shape[0]
	Rx = torch.zeros((B, 9, 1), dtype=torch.float)

	c = torch.cos(t)
	s = torch.sin(t)
	ones = torch.ones(B)

	Rx[:, 0, 0] = ones
	Rx[:, 4, 0] = c
	Rx[:, 5, 0] = -s
	Rx[:, 7, 0] = s
	Rx[:, 8, 0] = c

	Rx = Rx.reshape(B, 3, 3)

	return Rx


def Roty(t):
	"""
		Rotation about the x-axis.
		np.array([[c,  0,  s], [0,  1,  0], [-s, 0,  c]])

		-- input t shape B x 1
		-- return B x 3 x 3
	"""
	B = t.shape[0]
	Ry = torch.zeros((B, 9, 1), dtype=torch.float)

	c = torch.cos(t)
	s = torch.sin(t)
	ones = torch.ones(B)

	Ry[:, 0, 0] = c
	Ry[:, 2, 0] = s
	Ry[:, 4, 0] = ones
	Ry[:, 6, 0] = -s
	Ry[:, 8, 0] = c

	Ry = Ry.reshape(B, 3, 3)

	return Ry

def Rotz(t):
	"""
		Rotation about the z-axis.
		np.array([[c, -s,  0], [s,  c,  0], [0,  0,  1]])

		-- input t shape B x 1
		-- return B x 3 x 3
	"""
	B = t.shape[0]
	Rz = torch.zeros((B, 9, 1), dtype=torch.float)

	c = torch.cos(t)
	s = torch.sin(t)
	ones = torch.ones(B)

	Rz[:, 0, 0] = c
	Rz[:, 1, 0] = -s
	Rz[:, 3, 0] = s
	Rz[:, 4, 0] = c
	Rz[:, 8, 0] = ones

	Rz = Rz.reshape(B, 3, 3)

	return Rz