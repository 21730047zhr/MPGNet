import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from utils import load_state_dict
from networks_phase import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class amp_simulator(nn.Module): 
	def __init__(self, Dr0, img_size, thre, corr = -2, data_path = './TPG-FP', device = 'cuda:0', scale = 1.0, use_temp = False):
		super().__init__()

		self.img_size = img_size
		self.initial_grid = 16
		self.Dr0 = torch.tensor(Dr0)
		self.device = torch.device(device)
		self.Dr0 = torch.tensor(Dr0).to(self.device,dtype=torch.float32)
		self.mapping = amp_P2S()
		self.mapping = load_state_dict(self.mapping, './TPG-FP/TPG-FP_amp_model.pth')
		self.dict_psf = np.load('./TPG-FP/dictionary.npy', allow_pickle = True)

		self.mu = torch.tensor(self.dict_psf.item()['mu']).reshape((1,1,33,33)).to(self.device,dtype=torch.float32)
		self.dict_psf = torch.tensor(self.dict_psf.item()['dictionary'][:100,:]).reshape((100,1,33,33))
		self.dict_psf = self.dict_psf.to(self.device,dtype=torch.float32)

		self.R = np.load(os.path.join(data_path,'R-corr_{}.npy'.format(corr)))
		self.R = torch.tensor(self.R).to(self.device,dtype=torch.float32)
		self.offset = torch.tensor([31,31]).to(self.device,dtype=torch.float32)

		if use_temp: 
			self.S_half = np.load(os.path.join(data_path,'S_half-temp.npy'.format(img_size,Dr0)), allow_pickle=True)
		else:
			self.S_half = np.load(os.path.join(data_path,'S_half-size_{}-D_r0_{:.4f}_thre_{}.npy'.format(img_size,Dr0, thre)), allow_pickle=True)
		self.const = self.S_half.item()['const']
		self.S_half = torch.tensor(self.S_half.item()['s_half']).to(self.device,dtype=torch.float32)

		xx = torch.arange(0, img_size).view(1,-1).repeat(img_size,1)
		yy = torch.arange(0, img_size).view(-1,1).repeat(1,img_size)
		xx = xx.view(1,1,img_size,img_size).repeat(1,1,1,1)
		yy = yy.view(1,1,img_size,img_size).repeat(1,1,1,1)
		self.grid = torch.cat((xx,yy),1).permute(0,2,3,1).to(self.device,dtype=torch.float32)
		self.scale=scale

	def forward(self, img): 

		img_pad = F.pad(img.view((-1,1,self.img_size,self.img_size)), (16,16,16,16), mode = 'reflect')

		img_mean = F.conv2d(img_pad, self.mu).squeeze()

		dict_img = F.conv2d(img_pad, self.dict_psf)

		random_ = torch.sqrt(self.Dr0**(5/3)) * torch.randn((self.initial_grid**2 * 36),1,device=self.device)

		zer = torch.matmul(self.R,random_).view(self.initial_grid,self.initial_grid,36).permute(2,0,1).unsqueeze(0)

		zer = F.interpolate(zer,size=(self.img_size,self.img_size),mode='bilinear', align_corners=False)

		zer = zer * self.scale

		weight = self.mapping(zer.squeeze().permute(1,2,0).view(self.img_size**2,-1))

		weight = weight.view((self.img_size,self.img_size,100)).permute(2,0,1)

		out = weight.unsqueeze(0) * dict_img

		out = torch.sum(out,1) + img_mean

		pos = torch.fft.irfft2((self.S_half.permute(1, 2, 0).unsqueeze(0) * torch.randn(1, self.img_size,
								self.img_size, 2, device=self.device)), s=(self.img_size,self.img_size), dim=(1,2)) * self.const

		flow = 2.0*(self.grid+pos) / (self.img_size-1) - 1.0

		out = F.grid_sample(out.view((1,-1,self.img_size,self.img_size)), flow, 'bilinear', padding_mode='border', align_corners=False).squeeze()

		return out


class amp_P2S(nn.Module): 
	def __init__(self, input_dim = 36, hidden_dim = 100, output_dim = 100): 
		super().__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x): 

		y = F.relu(self.fc1(x))

		y = F.relu(self.fc2(y))

		y = F.relu(self.fc2(y))

		out = self.fc3(y)

		return out


class phase_simulator(nn.Module): 
	def __init__(self, Dr0, img_size, thre, corr = -2, data_path = './TPG-FP', device = 'cuda:0', scale = 1.0, use_temp = False):
		super().__init__()

		self.img_size = img_size
		self.Dr0 = torch.tensor(Dr0)
		self.device = torch.device(device)
		self.Dr0 = torch.tensor(Dr0).to(self.device,dtype=torch.float32)
		self.mapping = phase_P2S()
		self.mapping = load_state_dict(self.mapping, './TPG-FP/TPG-FP_phase_model.pth')

		self.scale=scale

	def forward(self, img, CN2, Humidity, Temperature, Visibility, Wind): 

		out = self.mapping(img.permute(0,2,3,1), CN2, Humidity, Temperature, Visibility, Wind).permute(0,3,1,2)

		return out * self.scale


class phase_P2S(nn.Module): 
	def __init__(self, input_dim = 1, hidden_dim = 256, output_dim = 1): 
		super().__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=1/12)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
				
	def forward(self, x, CN2, Humidity, Temperature, Visibility, Wind): 
		
		y = x * ((x*0.5*torch.pi).sin() + 1e-8).pow(CN2)
		
		y = F.relu(self.fc1(y))
		
		y = F.relu(self.fc2(y))* Humidity
		
		y = F.relu(self.fc2(y))* Temperature
		
		y = F.relu(self.fc2(y))* Visibility
		
		y = F.relu(self.fc2(y))* Wind
		
		out = torch.sin(self.fc3(y)) + x

		return torch.clamp_((out-torch.min(out))/(torch.max(out)-torch.min(out)), 0.0, 1.0)

