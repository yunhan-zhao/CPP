import numpy as np
import torch

def get_extrinsic_channel(imageTensor, focal_length, p_pt, extrinsic_para, CEILING_HEIGHT, inverse_tangent=True, augmentation=None):
	B, H, W = imageTensor.shape[0], imageTensor.shape[2], imageTensor.shape[3]
	K = _get_intrinsic_matrix(focal_length, p_pt, B)

	# make sure to adapt to your coordinate system 
	cam_height, roll = extrinsic_para[:, 2], extrinsic_para[:, 4] # all with size: [B]
	pitch = extrinsic_para[:, 3] - np.pi/2
	R = torch.bmm(Rotx(pitch), Rotz(roll)) # B x 3 x 3
	
	translation_v = torch.zeros((B, 3, 1), dtype=torch.float)
	translation_v[:, 1, 0] = -cam_height

	normal = torch.tensor((0, -1, 0), dtype=torch.float).reshape(-1, 1)
	normal_t = torch.transpose(normal, 0, 1)

	# convert normal and normal_t(transpose) to batches
	normal = normal.unsqueeze(0).expand(B, -1, 1)
	normal_t = normal_t.unsqueeze(0).expand(B, 1, -1)

	grid_y, grid_x = np.mgrid[0:H, 0:W]
	grid_y, grid_x = torch.tensor(grid_y, dtype=torch.float32), torch.tensor(grid_x, dtype=torch.float32)
	q = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x.reshape(-1))), dim=0).unsqueeze(0).expand(B, 3, H*W)
	
	# computing points intersecting ground plane
	scale_f = - torch.bmm(normal_t, translation_v) / torch.bmm(torch.bmm(torch.bmm(normal_t, R), torch.inverse(K)), q)
	p_f = torch.bmm(torch.bmm(R, torch.inverse(K)), q)
	p_f = p_f * scale_f.expand_as(p_f) + translation_v
	k_vec = torch.tensor((0, 0, 1), dtype=torch.float).reshape(-1, 1)
	k_vec_t = torch.transpose(k_vec, 0, 1)
	k_vec_t = k_vec_t.unsqueeze(0).expand(B, 1, -1)
	z_f = scale_f * torch.bmm(k_vec_t, q)
	
	z_f_channel = z_f.reshape(B, 1, H, W)
	
	# computing points intersecting celing plane
	scale_c = (CEILING_HEIGHT- torch.bmm(normal_t, translation_v)) / torch.bmm(torch.bmm(torch.bmm(normal_t, R), torch.inverse(K)), q)
	p_c = torch.bmm(torch.bmm(R, torch.inverse(K)), q)
	p_c = p_c * scale_c.expand_as(p_c) + translation_v
	z_c = scale_c * torch.bmm(k_vec_t, q)

	z_c_channel = z_c.reshape(B, 1, H, W)
	
	extrinsic_channel = torch.zeros(B, 1, H, W)
	extrinsic_channel[z_f_channel > 0.] =  z_f_channel[z_f_channel > 0.]
	extrinsic_channel[z_c_channel > 0.] =  z_c_channel[z_c_channel > 0.]    

	if inverse_tangent:
		extrinsic_channel = torch.atan(extrinsic_channel)

	if augmentation is not None:
		# augmentation is a bool tensor with size B, 1 means lrflip aug and 0 means original
		assert extrinsic_channel.dim() == 4
		extrinsic_channel_aug = torch.zeros_like(extrinsic_channel)
		extrinsic_channel_aug[augmentation] = torch.flip(extrinsic_channel[augmentation], [1, 3])
		extrinsic_channel_aug[~augmentation] = extrinsic_channel[~augmentation]
		extrinsic_channel = extrinsic_channel_aug

	return extrinsic_channel

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
	# print(t)
	# print(c.shape, c)
	# print(ones.shape, ones)

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