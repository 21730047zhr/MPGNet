import torch
import torch.nn as nn
from transformer_utils import *
from einops import rearrange
import triton
import triton.language as tl
import math
from yaml_config import *

class SRCA_without_triton(nn.Module):
	def __init__(self, dim, num_heads, pool_size):
		super(SRCA_without_triton, self).__init__()
		self.num_heads = num_heads
		#self.temperature = (dim/num_heads)**-0.5#nn.Parameter(torch.ones(num_heads, 1, 1))

		self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
		self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
		#self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
		self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
		self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
		self.sr = nn.Conv2d(dim*3, dim*2, kernel_size=1, stride=1)
		self.norm = RMSNorm(dim*2)
		#self.norm = nn.LayerNorm(dim*2)
		self.act = nn.GELU()
		self.min_pool = nn.MaxPool2d(kernel_size=pool_size)#池化大小待调整
		self.max_pool = nn.MaxPool2d(kernel_size=pool_size)
		self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)
		self.minus_one = torch.tensor(-1)
		
	def forward(self, x, y):
		b, c, h, w = x.shape

		q = self.q_dwconv(self.q(x))
		#加入池化kv
		y_min = self.min_pool(y*self.minus_one)*self.minus_one
		y_max = self.max_pool(y)
		y_avg = self.avg_pool(y)
		y_new = torch.cat([y_min,y_max,y_avg], dim=1)
		y_new = self.sr(y_new)
		b_tmp, c_tmp, h_tmp, w_tmp = y_new.shape
		y_new = y_new.reshape(b_tmp, c_tmp, -1).permute(0, 2, 1).contiguous()#变成b(hw)c
		y = self.act(self.norm(y_new)).reshape(b_tmp, h_tmp, w_tmp, c_tmp).permute(0, 3, 1, 2).contiguous()#变成bchw
		kv = self.kv_dwconv(y)
		#kv = self.kv_dwconv(self.kv(y))
		k, v = kv.chunk(2, dim=1)

		q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).permute(0, 1, 3, 2).contiguous()#b head hw c
		k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)					#b head c hw'
		v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).permute(0, 1, 3, 2).contiguous()#b head hw' c

		#q = torch.nn.functional.normalize(q, dim=-1)
		#k = torch.nn.functional.normalize(k, dim=-1)

		attn = (q @ k) * self.temperature
		attn = nn.functional.softmax(attn,dim=-1)

		out = (attn @ v)#b head hw c

		out = out.transpose(1, 2).reshape(b, c, h, w)

		out = self.project_out(out)#b dim h w
		return out

@triton.jit
def attention_kernel(
	# 输入/输出指针
	q_ptr, k_ptr, v_ptr, output_ptr,
	# 张量元数据
	B, H, h, w, D, # 批大小, 头数, 高度, 宽度, 特征维度
	kh, kw,	 # 缩减后序列长度kh*kw
	# 步长信息
	q_stride_b, q_stride_h, q_stride_n,
	k_stride_b, k_stride_h, k_stride_n,
	v_stride_b, v_stride_h, v_stride_n,
	out_stride_b, out_stride_h, out_stride_n,
	# 块大小参数
	BLOCK_SIZE: tl.constexpr,
):
	# 获取当前处理的元素索引
	pid_b = tl.program_id(0)  # B*H索引
	pid_h = tl.program_id(1)  # height索引
	pid_w = tl.program_id(2)  # width索引
	batch_idx = pid_b // H
	head_idx = pid_b % H
	# 计算Q指针
	q_offset = batch_idx * q_stride_b + head_idx * q_stride_h + pid_h * w * q_stride_n + pid_w * q_stride_n
	mask_q = tl.arange(0, BLOCK_SIZE) < D
	q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE), mask = mask_q, other=0)
	
	# 初始化注意力分数和最大值
	max_val = tl.zeros([1], dtype=tl.float32) - float('inf')
	exp_sum = tl.zeros([1], dtype=tl.float32)
	output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
	
	# 遍历缩减后的序列
	for k_hStart in range(0, kh):
		for k_wStart in range(0, kw):
			red_offset = batch_idx * k_stride_b + head_idx * k_stride_h + k_hStart * kw * k_stride_n + k_wStart * k_stride_n
			# 加载缩减后的K和V
			mask_red = tl.arange(0, BLOCK_SIZE) < D
			
			k_red = tl.load(k_ptr + red_offset + tl.arange(0, BLOCK_SIZE), mask=mask_red)
			v_red = tl.load(v_ptr + red_offset + tl.arange(0, BLOCK_SIZE), mask=mask_red)
			
			# 计算Q与K的点积
			attn_scores = tl.sum(q * k_red, axis = -1, keep_dims = True)
			#print("attn_scores: ",attn_scores)
			#attn_scores = attn_scores / tl.sqrt(tl.float32(D))
			
			# 在线Softmax计算
			# 1. 查找当前块的最大值
			block_max = tl.max(attn_scores, axis=0)
			new_max = tl.maximum(max_val, block_max)
			
			# 2. 更新指数和
			exp_vals = tl.exp(attn_scores - new_max)
			exp_sum = exp_sum * tl.exp(max_val - new_max) + tl.sum(exp_vals, axis=0)
			
			# 3. 更新输出
			scale = tl.exp(max_val - new_max)
			output = output * scale + tl.sum(exp_vals[:, None] * v_red, axis=0)
			
			max_val = new_max
		
		# 最终归一化
	output = output / exp_sum
		
	# 存储结果
	out_offset = (
		batch_idx * out_stride_b + 
		head_idx * out_stride_h + 
		pid_h * w * out_stride_n + 
		pid_w * out_stride_n
		)
	tl.store(output_ptr + out_offset + tl.arange(0, BLOCK_SIZE), output, mask=mask_q)

def qkv_attention(
	q: torch.Tensor, 
	k: torch.Tensor, 
	v: torch.Tensor, 
	h: int,
	w: int,
	kh: int,
	kw: int
):
	"""PVTv2 空间缩减注意力实现
	
	Args:
		q: 查询张量 (B, H, D, N)
		k: 键张量 (B, H, D, N)
		v: 值张量 (B, H, D, N)
	
	Returns:
		注意力输出 (B, H, D, N)
	"""
	B, H, D, N = q.shape
	kB, kH, kD, kN = k.shape
	#维度对齐
	q_red = q.permute(0, 1, 3, 2).contiguous()#BHND
	k_red = k.permute(0, 1, 3, 2).contiguous()#BHND
	v_red = v.permute(0, 1, 3, 2).contiguous()#BHND
	
	# 分配输出张量
	output = torch.empty_like(q_red)
	
	# 配置Triton内核参数
	BLOCK_SIZE = D
	B_new = B*H
	grid = (B_new, h, w)
	# 调用Triton内核
	attention_kernel[grid](
		q_red, k_red, v_red, output,
		B, H, h, w, D,
		kh, kw,
		q_red.stride(0), q_red.stride(1), q_red.stride(2),
		k_red.stride(0), k_red.stride(1), k_red.stride(2),
		v_red.stride(0), v_red.stride(1), v_red.stride(2),
		output.stride(0), output.stride(1), output.stride(2),
		BLOCK_SIZE=BLOCK_SIZE,
	)#输出output是BHND
	output = output.permute(0,1,3,2).contiguous()
	return output

	
# Space Reduction Cross Attention
class SRCA(nn.Module):
	def __init__(self, dim, num_heads, pool_size):
		super(SRCA, self).__init__()
		self.num_heads = num_heads
		#self.temperature = (dim/num_heads)**-0.5#nn.Parameter(torch.ones(num_heads, 1, 1))

		self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)#卷积的输入输出tensor是BCHW
		self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
		self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=False)
		self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
		self.sr = nn.Conv2d(dim*3, dim*2, kernel_size=1, stride=1)
		self.norm = RMSNorm(dim*2)
		#self.norm = nn.LayerNorm(dim*2)
		self.act = nn.GELU()
		self.min_pool = nn.MaxPool2d(kernel_size=pool_size)#池化大小待调整,输入输出tensor排布是BCHW
		self.max_pool = nn.MaxPool2d(kernel_size=pool_size)
		self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)
		self.pool_size = pool_size
		self.minus_one = torch.tensor(-1)
		
	def forward(self, x, y):
		b, c, h, w = x.shape

		q = self.q_dwconv(self.q(x))
		#加入池化kv
		y_min = self.min_pool(y*self.minus_one)*self.minus_one
		
		y_max = self.max_pool(y)
		y_avg = self.avg_pool(y)
		y_new = torch.cat([y_min,y_max,y_avg], dim=1)
		y_new = self.sr(y_new)
		b_tmp, c_tmp, h_tmp, w_tmp = y_new.shape
		y_new = y_new.reshape(b_tmp, c_tmp, -1).permute(0, 2, 1)#变成b(hw)c
		y = self.act(self.norm(y_new)).reshape(b_tmp, h_tmp, w_tmp, c_tmp).permute(0, 3, 1, 2)#变成bchw
		kv = self.kv_dwconv(y)
		#kv = self.kv_dwconv(self.kv(y))
		k, v = kv.chunk(2, dim=1)
		Q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		K = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h_tmp, w=w_tmp)
		V = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h_tmp, w=w_tmp)

		triton_out = qkv_attention(Q, K, V, h, w, h_tmp, w_tmp)
		out = rearrange(triton_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
		out = self.project_out(out)
		return out
		
# Intensity Enhancement Layer
class IEL(nn.Module):
	def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
		super(IEL, self).__init__()

		hidden_features = int(dim*ffn_expansion_factor)

		self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
		
		self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
		self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
		self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
	   
		self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

		self.Tanh = nn.Tanh()
	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x1 = self.Tanh(self.dwconv1(x1)) + x1
		x2 = self.Tanh(self.dwconv2(x2)) + x2
		x = x1 * x2
		x = self.project_out(x)
		return x
  
  
class Transformer_SRCA(nn.Module):
	def __init__(self, dim, num_heads, pool_size):
		super(Transformer_SRCA, self).__init__()
		self.gdfn = IEL(dim)
		self.norm = LayerNorm(dim)
		self.ffn = SRCA(dim, num_heads, pool_size)
		
	def forward(self, x, y):
		x = x + self.ffn(self.norm(x),self.norm(y))
		x = x + self.gdfn(self.norm(x))
		y = y + self.ffn(self.norm(y),self.norm(x))
		y = y + self.gdfn(self.norm(y)) 
		return x, y
		
class MPGNet(nn.Module):
	def __init__(self, depths=[1, 2, 4],
				 channels=[16, 64, 256, 1024],
				 heads=[2, 4, 8],
				 pool_size=[4, 2, 2],
				 norm=False
		):
		super(MPGNet, self).__init__()
		
		self.channels = channels
		self.heads = heads
		self.depths = depths

		self.IE_block = nn.PixelUnshuffle(2)
		self.E_block0 = nn.Conv2d(4, channels[0], 3, stride=1, padding=1)
		self.E_block1 = NormDownsample(channels[0], channels[1], use_norm = norm)
		self.E_block2 = NormDownsample(channels[1], channels[2], use_norm = norm)
		self.E_block3 = NormDownsample(channels[2], channels[3], use_norm = norm)
		
		self.D_block3 = NormUpsample(channels[3], channels[2], use_norm = norm)
		self.D_block2 = NormUpsample(channels[2], channels[1], use_norm = norm)
		self.D_block1 = NormUpsample(channels[1], channels[0], use_norm = norm)
		self.upD_block0 = NormUpsample(channels[0], 4, use_norm = norm)
		self.upD_block1 = nn.Conv2d(4, 1, 3, stride=1, padding=1)
		self.ID_block = nn.PixelShuffle(2)
		self.up_scale = nn.Sequential(
			nn.Conv2d(channels[0],4,kernel_size=3,stride=1, padding=1, bias=False),
			nn.UpsamplingBilinear2d(scale_factor=2))
		self.block1 = nn.ModuleList([Transformer_SRCA(dim=channels[1], num_heads=heads[0], pool_size=pool_size[0])
									for i in range(depths[0])])
		self.block2 = nn.ModuleList([Transformer_SRCA(dim=channels[2], num_heads=heads[1], pool_size=pool_size[1])
									for i in range(depths[1])])
		self.block3 = nn.ModuleList([Transformer_SRCA(dim=channels[3], num_heads=heads[2], pool_size=pool_size[2])
									for i in range(depths[2])])

	def forward(self, x, opt):
		CN2_value = torch.tensor(opt['CN2']['value'], device='cuda', dtype=torch.float32)
		self.CN2 = (torch.log10(CN2_value) + 30) / 10
		
		H_min = opt['humidity']['min']
		H_max = opt['humidity']['max']
		H_value = opt['humidity']['value']
		self.Humidity = 0.1 + torch.tensor((H_value - H_min)/(H_max - H_min), device='cuda',dtype=torch.float32)
		
		T_min = opt['temperature']['min']
		T_max = opt['temperature']['max']
		T_value = opt['temperature']['value']
		self.Temperature = 0.1 + torch.tensor((T_value - T_min)/(T_max - T_min), device='cuda',dtype=torch.float32)
		
		V_min = opt['visibility']['min']
		V_max = opt['visibility']['max']
		V_value = opt['visibility']['value']
		self.Visibility = 0.1 + torch.tensor((V_value - V_min)/(V_max - V_min), device='cuda',dtype=torch.float32)
		
		W_min = opt['wind']['min']
		W_max = opt['wind']['max']
		W_value = opt['wind']['value']
		self.Wind = 0.1 + torch.tensor((W_value - W_min)/(W_max - W_min), device='cuda',dtype=torch.float32)
		
		cn2_x = x * ((x*0.5*torch.pi).sin() + 1e-8).pow(self.CN2) 
		x_0 = self.E_block0(x)
		i_0 = self.E_block0(cn2_x)
		
		x_1 = self.E_block1(x_0)
		i_0 = torch.mul(i_0, self.Humidity)
		i_enc0 = self.IE_block(i_0)
		i_enc1 = torch.mul(i_enc0, self.Temperature)
		
		#i_jump0 = i_0
		x_jump0 = x_0
		for i, blk in enumerate(self.block1):
			i_enc1, x_1 = blk(i_enc1, x_1)
		#i_jump1 = i_enc1
		x_jump1 = x_1
		i_enc2 = self.IE_block(i_enc1)
		i_enc2 = torch.mul(i_enc2, self.Visibility)
		x_2 = self.E_block2(x_1)
		for i, blk in enumerate(self.block2):
			i_enc2, x_2 = blk(i_enc2, x_2)
		#i_jump2 = i_enc2
		x_jump2 = x_2
		i_enc2 = self.IE_block(i_enc2)
		i_enc3 = torch.mul(i_enc2, self.Wind)
		x_3 = self.E_block3(x_2)
		for i, blk in enumerate(self.block3):
			i_enc3, x_3 = blk(i_enc3, x_3)
		x_dec2 = self.D_block3(x_3, x_jump2)
		i_dec2 = self.ID_block(i_enc3)
		for i, blk in enumerate(self.block2):
			i_dec3, x_dec2 = blk(i_dec2, x_dec2)
		x_dec1 = self.D_block2(x_dec2, x_jump1)
		i_dec1 = self.ID_block(i_dec2)
		for i, blk in enumerate(self.block1):
			i_dec1, x_dec1 = blk(i_dec1, x_dec1)
		i_dec0 = self.ID_block(i_dec1)
		i_dec0 = self.up_scale(i_dec0)
		x_dec0 = self.D_block1(x_dec1, x_jump0)
		
		output_x = self.upD_block0(x_dec0, i_dec0)
		output_x = self.upD_block1(output_x)
		return output_x

