import numpy as np
import cv2
import os
from turbSimulator import amp_simulator, phase_simulator
from turbMeasurement import tilt_mat, corr_mat
import torch
import cv2
import numpy as np
from os import listdir
import math

def random_add_gaussian_noise(img, sigma_range=(0, 10), clip=True, rounds=False, bitWidth=255):
	#Input image, shape (h, w, c), range [0, 1], float32
	size = np.shape(img)
	rr = np.random.rand(size[0], size[1])
	sigma = np.random.uniform(sigma_range[0], sigma_range[1])
	noise = rr * 1.0 * sigma / bitWidth
	out = img + noise

	if clip and rounds:
		out = np.clip((out * bitWidth).round(), 0, bitWidth) / bitWidth
	elif clip:
		out = np.clip(out, 0, 1)
	elif rounds:
		out = (out * bitWidth).round() / bitWidth

	return out


def random_add_poisson_noise(img, scale_range=(0, 1.0), clip=True, rounds=False, bitWidth=255):
	#Input image, shape (h, w, c), range [0, 1], float32
	img = np.clip((img * bitWidth).round(), 0, bitWidth) / bitWidth
	vals = len(np.unique(img))
	vals = 2**np.ceil(np.log2(vals))
	lam = img*vals
	tmp = np.float32(np.random.poisson(lam)) / float(vals)
	noise = tmp - img
	scale = np.random.uniform(scale_range[0], scale_range[1])
	out = img + noise*scale

	if clip and rounds:
		out = np.clip((out * bitWidth).round(), 0, bitWidth) / bitWidth
	elif clip:
		out = np.clip(out, 0, 1)
	elif rounds:
		out = (out * bitWidth).round() / bitWidth

	return out


def TPG_FP(ori_path, img_name):
	N = 2
	padLeft = 0
	padTop = 0
	bitWidth = 65535
	D = 100
	L = 1000
	r0 = 5
	Corr = -2
	TH = 0.002
	lr_size = 256
	FP_size = lr_size*3
	turb_size = lr_size*2
	apDia = 60
	CN2 = 8.62E-16
	Humidity = 87.2
	Temperature = 21.0
	Visibility = 19.23
	Wind = 0.46
	
	
	cv2imreadFlag = 0

	data_dir = ori_path + img_name
	hr_path = './data_simulation/hr/'
	lr_path = './data_simulation/lr/'

	if not os.path.exists(hr_path):
		os.makedirs(hr_path)
	if not os.path.exists(lr_path):
		os.makedirs(lr_path)


	
	CN2 = (math.log10(CN2) + 30) / 10
	Humidity = 0.1 + (Humidity - 0)/100
	Temperature = 0.1 + (Temperature +40)/100
	Visibility = 0.1 + (Visibility - 0)/100
	Wind = 0.1 + (Wind - 0)/100
	
	imgOri = cv2.imread(data_dir, cv2imreadFlag)
	imgHR = cv2.resize(imgOri, dsize=(turb_size, turb_size), interpolation=cv2.INTER_LANCZOS4)
	imgHR = imgHR * (imgHR>0)
	imgHR = np.array(imgHR, dtype=float)
	imgHR = (imgHR-np.min(imgHR)) / (np.max(imgHR)-np.min(imgHR)) * bitWidth
	cv2.imwrite(hr_path+img_name, (imgHR).astype("uint16"))

	x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, apDia, endpoint=True), np.linspace(-1, 1, apDia, endpoint=True))
	pupil = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1

	imgFP = cv2.resize(imgOri, dsize=(FP_size, FP_size), interpolation=cv2.INTER_LANCZOS4)
	imgFP = imgFP * (imgFP>0)
	imgFP = np.array(imgFP, dtype=float)
	rr = np.random.rand(FP_size, FP_size)
	phase = rr/ np.max(rr)

	#tilt_mat(FP_size, D, r0, L, thre = TH, use_temp = False, save_path = './TPG-FP/')
	#corr_mat(Corr, D, r0, save_path = './TPG-FP/')

	for group_idx in range(11, 12):
		simulator_amp = amp_simulator(D/r0, FP_size, thre=TH, corr=Corr, use_temp=False, data_path='./TPG-FP/').to(device, dtype=torch.float32)
		x = torch.tensor((imgFP / np.max(imgFP)), device = device, dtype = torch.float32)
		out = simulator_amp(x).squeeze(0).squeeze(0)
		out_amp_img = out.clamp_(0, 1).detach().cpu().numpy()
		out_amp_img = np.sqrt(out_amp_img * bitWidth)
		
		simulator_phase = phase_simulator(D/r0, FP_size, thre=TH, corr=Corr, use_temp=False, data_path='./TPG-FP/').to(device, dtype=torch.float32)
		x = torch.tensor(phase, device = device, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
		out = simulator_phase(x, CN2, Humidity, Temperature, Visibility, Wind).squeeze(0).squeeze(0)
		out_phase_img = out.detach().cpu().numpy()

		complex_img = out_amp_img * np.exp(1j*out_phase_img * np.pi * 2)
		UinSpec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_img)))

		for i in range(N):
			for j in range(N):
				padLeft = (FP_size - round(1.2*apDia) - lr_size)//2 + round(apDia*1.2)*i
				padTop = (FP_size - round(1.2*apDia) - lr_size)//2 + round(apDia*1.2)*j
				pupil_left = (lr_size-apDia)//2

				pad_pupil = np.pad(pupil, ((pupil_left, lr_size-apDia-pupil_left), (pupil_left, lr_size-apDia-pupil_left)), 'constant', constant_values=0)
				UinSpec1 = UinSpec[padLeft:padLeft+lr_size, padTop:padTop+lr_size]
				Uout = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(UinSpec1 * pad_pupil))))
				Uout = Uout**2

				img = Uout/np.max(Uout)
				img = random_add_gaussian_noise(img, sigma_range=(0, 10), clip=True, rounds=False, bitWidth = bitWidth)
				Uout = random_add_poisson_noise(img, scale_range=(0, 1), clip=True, rounds=False, bitWidth = bitWidth)
				Uout = (Uout-np.min(Uout))/(np.max(Uout)-np.min(Uout))*bitWidth
				cv2.imwrite(lr_path+img_name[:-4]+'-'+str(group_idx)+'-'+str(i*2+j+1)+'.png',Uout.astype("uint16"))




device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('CPU')
ori_path = './data/hr/'
images_list = listdir(ori_path)
print(images_list[0])
for idx in range(len(images_list)):
	TPG_FP(ori_path, images_list[idx])

