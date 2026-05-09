import argparse
import os
from math import log10, cos, pi
import torchvision.transforms as transforms
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
import cv2
import torch
import pytorch_ssim
from model_TPGNet import *
import numpy as np
import os
import random
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

parser = argparse.ArgumentParser(description='Test Parameters')
parser.add_argument('--test_root', default='./dataset/600/1/', type=str, help='test data_root')
parser.add_argument('--save_root', default='./results/', type=str, help='save data_root')
parser.add_argument('--model_name', default='./epochs/0511_TPGNet_epoch_900_tiny.pth', type=str, help='test data_root')
torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()
# 计算一下总内存有多少。
#total_memory = torch.cuda.get_device_properties(0).total_memory#
#print(torch.cuda.memory_summary())

def load_state_dict(
        model: nn.Module,
        model_weights_path: str,
) -> nn.Module:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    pretrained_dict = checkpoint["state_dict"]
    # Load model state dict. Extract the fitted model weights
    model_state_dict = model.state_dict()
    #state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
    #               k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    # Overwrite the model weights to the current model
    keys=[]
    for k,v in pretrained_dict.items():
      keys.append(k)
    i=0
    for k, v in model_state_dict.items():
      if v.size()== pretrained_dict[keys[i]].size():
        model_state_dict[k] = pretrained_dict[keys[i]]
        i = i+1
  
    #model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)

if __name__ == '__main__':
	opt = parser.parse_args()

	net = TPGNet(depths=[3, 3, 6]).cuda()
	net.load_state_dict(torch.load(opt.model_name, map_location="cuda:0"))
	print('# generator parameters:', sum(param.numel() for param in net.parameters()))
	lr_in = []
	for jj in range(4):
		image_filename_lr = opt.test_root + str(jj+1) + '.png'
		lr = cv2.imread(image_filename_lr,-1)
		lr = np.float32(lr)
		lr = (lr - np.min(lr)) / (np.max(lr)-np.min(lr))
		lr = torch.FloatTensor(lr)
		lr_in.append(lr)

	yaml_file = opt.test_root + 'turbulence_params.yml'
	turb_opt = load_config(yaml_file)
		
	lr_in = torch.stack(lr_in, axis=0).unsqueeze(0)
	input = lr_in.cuda()
	net.eval()
	with torch.no_grad():
		output = net(input, turb_opt)
		output = torch.clamp(output,0,1)

		enhanced_img = transforms.ToPILImage()(output.squeeze(0))
		enhanced_img.save(os.path.join(opt.save_root,'output.png'))
		del net
