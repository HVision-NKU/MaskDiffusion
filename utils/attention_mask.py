#######
# Code refactored to implement the mask mechanism.
# The original implementation was created in April 2023 for a conference submission.
# The implementation reuses code from "attend-and-excite" and "densediffusion" to achieve a more simple implementation on 2023.9.20.
#######


from select import select
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc

from utils import ptp_utils
from utils.ptp_utils import AttentionStore  as AttentionStore 
from PIL import Image
device = torch.device('cuda')
import math


@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel








def combine_color_layers(bool_layers, color_map=  [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    """
    Combines several boolean layers each representing a color into a single RGB image.
    
    Parameters:
        bool_layers (numpy.ndarray): A boolean array of shape (n, h, w), each layer n representing a different color.
        color_map (list): List of tuples, each representing the RGB values for the corresponding layer.
    
    Returns:
        PIL.Image: The combined RGB image.
    """
    bool_layers = bool_layers.cpu().numpy().astype(np.int32)

    if bool_layers.shape[0] > len(color_map):
        raise ValueError(f"Number of layers {bool_layers.shape[0]} and number of colors {len(color_map)} must match.")
    
    # Initialize an empty image array with zeros.
    h, w = bool_layers.shape[1], bool_layers.shape[2]
    n = bool_layers.shape[0]
    image_array = np.zeros((h, w, 3), dtype=np.int32)
    
    # Add each color layer to the image.
    for i in range(n):
        color = color_map[i]
        layer = bool_layers[i]
        # print(layer)
        # Only add where the boolean layer is True
        for c in range(3):  # RGB channels
            image_array[:, :, c] += (color[c] * layer)
    
    # Clip values to be in valid range (0-255) after combining colors
    image_array = np.clip(image_array, 0, 255).astype(dtype=np.int8)
    
    return Image.fromarray(image_array, 'RGB')


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def point_to_line_distance(point, endpoint1, endpoint2):
    """
    Calculate the distance from a point to a line defined by two endpoints.
    
    Args:
    point (tuple): The coordinates of the point (x, y).
    endpoint1 (tuple): The coordinates of the first endpoint (x1, y1).
    endpoint2 (tuple): The coordinates of the second endpoint (x2, y2).
    
    Returns:
    float: The distance from the point to the line.
    """
    x1, y1 = endpoint1
    x2, y2 = endpoint2
    x, y = point
    a = np.sqrt((x2-x1)**2+ (y2-y1)**2)/2
    b = a/2.5
    # Numerator of the distance formula
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    # Denominator of the distance formula
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if denominator == 0:
        raise ValueError("The two endpoints cannot be the same point.")
    
    # Calculate the distance
    ya = numerator / denominator

    #
    l = np.sqrt((y - (y1+y2)/2)**2 + (x - (x2 + x1)/2)**2)
    #
    if l**2-y**2 <=0:
        xa = 0
    else:
        xa = np.sqrt(l**2-y**2)
    return (xa*xa) / (a*a) + (ya*ya) / (b*b)
    



def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1)) 

def judge_region(x,y,h,w,object_nums):
    band_width = 16// object_nums
    if (x ) // band_width == (x  + h)//band_width or (y  )// band_width == (y  + w) //band_width:
        return True
    else:
        return False

def find_max_sum_submatrix(matrix, area,object_nums):
    max_sum = 0

    num_rows, num_cols = len(matrix), len(matrix[0])
    # print(num_rows,num_cols)
    hmax = int(area**0.5) 
    # hmin = int(area / (16 * 0.7))
    possiblehws = []
    for h_pos in range(4,hmax):
        w_pos = int(area/h_pos)
        possiblehws.append((h_pos,w_pos))
        possiblehws.append((w_pos,h_pos))
    xm = 0
    ym = 0
    hwm =  possiblehws[0]
    # 遍历每一个可能的矩形框起点位置
    for i in range(0,num_rows):
        for j in range(0,num_cols):            
            for hw_now in possiblehws:
                h,w = hw_now
                if i+h >= num_rows-1 or j+w >= num_cols-1:
                    continue
                # if not judge_region(i,j,h,w,object_nums):
                #     continue
                current_sum = 0
                
                # 计算这个矩形框白所有元素的和
                for ki in range(h):
                    for kj in range(w):
                        current_sum += matrix[i + ki][j + kj]
                # 更新最大和
                if max_sum < current_sum/(h*w):
                    hwm = hw_now
                    xm = i
                    ym = j
                    max_sum = current_sum/(h*w)
    
    mask = torch.zeros((64,64)).to(matrix.device)
    real_mask = torch.zeros((64,64)).to(matrix.device)
    forbid_mask = torch.zeros_like(matrix)
    hm,wm = hwm
    forbid_mask[xm:xm+hm,ym:ym+wm] = 1
    # forbid_mask[
    # (xm - 1 if xm - 1 > 0 else xm) : (xm + hm + 1 if xm + hm + 1 < num_rows else xm + hm),
    # (ym - 1 if ym - 1 > 0 else ym) : (ym + wm + 1 if ym + wm + 1 < num_cols else ym + wm)
    # ] = 1
    hm *= 4
    wm *= 4
    xm *= 4
    ym *= 4
    mask[xm:xm+hm,ym:ym+wm] = 1
    for iterh in range(hm+4):
        for iterw in range(wm+4):
            if xm+iterh-2 >= num_rows*4-1 or ym+iterw-2 >= num_cols*4-1:
                continue
            if xm+iterh-2 <0 or ym+iterw-2 < 0:
                continue
            if hm > wm:
                if point_to_line_distance((xm+iterh-2,ym+iterw-2), (xm,ym), (xm+hm,ym+wm))<1:
                    real_mask[xm+iterh-2,ym+iterw-2] =1
            else:
                if point_to_line_distance((xm+iterh-2,ym+iterw-2), (xm,ym), (xm+hm,ym+wm))<1:
                    real_mask[xm+iterh-2,ym+iterw-2] =1
    return mask,mask,forbid_mask #mask,real_mask,forbid_mask

def resize_attn(attn_map,th,tw):
    h = int(math.sqrt(attn_map.shape[1]))
    w = h
    bz, _, c = attn_map.shape
    attn_map = attn_map.permute(0,2,1).reshape(bz,c,h,w)
    attn_map = torch.mean(attn_map,dim = 0,keepdim = True)
    resized_attn_map = nnf.interpolate(attn_map, size=(th,tw), mode='bilinear').squeeze(0)
    return resized_attn_map 
def text_list(text):
    text =  text.replace(' ','')
    text =  text.replace('\n','')
    text =  text.replace('\t','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    return tuple(result)
class MaskdiffusionStore:

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    @property
    def num_uncond_att_layers(self):
        return 0
    def forward(self, attn,is_cross: bool, place_in_unet: str):
        if not is_cross:
            if self.cur_step < 15:
                return self.add_self_mask(attn)
            else:
                return attn
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_step > 1:
            attn = self.add_mask(attn)
        if attn.shape[1] == 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn[int(attn.size(0)/2):].softmax(-1).clone())
        return attn


    def add_mask(self,sim):
        reduction_factor= (self.timesteps[self.cur_step]/1000) ** 5
        batch_size = int(sim.shape[0]/2)
        if self.cur_step == 0:
            return sim
        heads_nums = sim.shape[0]//2
        mask = self.mask[sim.size(1)].repeat(heads_nums,1,1)
        ## Calculate the minimum and maximum values along the token dimension.
        min_value = sim[batch_size:].min(-1)[0].unsqueeze(-1)
        max_value = sim[batch_size:].max(-1)[0].unsqueeze(-1)  
        sim[batch_size:] += reduction_factor*1*(mask>0.05)*mask*(max_value-sim[batch_size:])
        if self.cur_step <15:
            sim[batch_size:] = sim[batch_size:]  - (mask==0)*reduction_factor*(sim[batch_size:]-min_value) +  sim[batch_size:] * (mask>= 0.05) 


        return sim
    def add_self_mask(self,sim):
        reduction_factor = 0.3*(self.timesteps[self.cur_step]/1000) ** 5
        batch_size = int(sim.shape[0]/2)
        if self.cur_step == 0:
            return sim
        
        heads_nums = sim.shape[0]//2
        mask = self.self_mask[sim.size(1)].repeat(heads_nums,1,1)

        min_value = sim[batch_size:].min(-1)[0].unsqueeze(-1)
        max_value = sim[batch_size:].max(-1)[0].unsqueeze(-1)  
        sim[batch_size:] += (mask>0)*reduction_factor*(max_value-sim[batch_size:])
        sim[batch_size:] -= (mask == 0)*reduction_factor*(sim[batch_size:]-min_value)
        return sim


    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_step >= self.end_step:
            return attn
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
            if self.cur_step <5:
                self.generated_mask2()
        return attn
    def reset(self):

        self.cur_step = 0
        self.cur_att_layer = 0
        ####
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def get_input_mask(self,token_id,device):
        y0,x0,y1,x1 = self.input_mask[token_id]
        x0 = x0 // 8
        x1 = x1 // 8
        y0 = y0 // 8
        y1 = y1 // 8
        num_rows = 16
        num_cols = 16
        hm = x1-x0
        wm = y1-y0
        
        mask = torch.zeros((64,64)).to(device)
        real_mask = torch.zeros((64,64)).to(device)
        forbid_mask = torch.zeros((16,16)).to(device)
        forbid_mask[x0//8:x1//8,y0//8:y1//8] = 1
        mask[x0:x1,y0:y1] = 1

        for iterh in range(hm+4):
            for iterw in range(wm+4):
                if x0 + iterh-2 >= num_rows * 4 - 1 or y0 + iterw-2 >= num_cols * 4 - 1:
                    continue
                if x0 + iterh-2 <0 or y0 + iterw-2 < 0:
                    continue
                if hm > wm:
                    if point_to_line_distance((x0+iterh-2,y0+iterw-2), (x0,y0), (x0+hm,y0+wm))<1:
                        real_mask[x0+iterh-2,y0+iterw-2] =1
                else:
                    if point_to_line_distance((x0+iterh-2,y0+iterw-2), (x0,y0), (x0+hm,y0+wm))<1:
                        real_mask[x0+iterh-2,y0+iterw-2] =1

        return mask,mask,forbid_mask

    def generated_mask2(self):
        maskres = 16
        ##### Strength = 5
        w = 5
        average_attention = self.get_average_global_attention()
        extract_attentions = average_attention["down_cross"] + average_attention["up_cross"]
        print(len(extract_attentions))
        ## Bz*head, h*w, 77  BZ=1
        extract_attentions = [resize_attn(extract_attention,maskres,maskres) for extract_attention in extract_attentions]
        mean_attentions = torch.mean(torch.stack(extract_attentions),dim = 0,keepdim = False).reshape(77,-1)  
        # print(mean_attentions.shape)  # 77 H,W
        num_pixels = mean_attentions.shape[-1]
        token_ids = list(self.token_dict.keys())
        protect_indexs = {}
        protect_attentions = mean_attentions.clone()
        ## new zeros mask_maps 
        mask_maps = torch.zeros(1,77,64,64).to(device)
        mask_maps[:,0] = 0.01 
        mask_maps[:,self.text_length-2:,:,:] += 0.20
        self_maps = torch.zeros(1,maskres*maskres,maskres*maskres).to(device)
        negative_mask_map = torch.zeros(1,77,maskres,maskres).to(device)
        ## perform protection
        for token_id in token_ids:
            protect_indexs[token_id] = torch.topk(mean_attentions[token_id],int(num_pixels*0.15),0).indices
            sub_prompt_ids = self.token_dict[token_id]
            minus_mask = torch.zeros_like(mean_attentions, dtype=torch.bool)

            minus_mask[:,protect_indexs[token_id]] = True
            minus_mask[sub_prompt_ids] = False
            mean_attentions[minus_mask] /= 2
        save_mask_maps = []
        save_real_maps = []
        def get_other_ids(token_id):
            id_pools = []
            for iter_id in list(self.token_dict.keys()):
                if token_id != iter_id:
                    id_pools += self.token_dict[iter_id]
            return id_pools
        ## select the max rectangle
        object_nums = len(token_ids)
        for token_id in token_ids:
            sub_prompt_ids = self.token_dict[token_id]
            curmap = mean_attentions[token_id].clone()
            if self.input_mask is None:
                real_mask,mask_map,forbid_mask = find_max_sum_submatrix(curmap.reshape(maskres,maskres),maskres**2*0.2,object_nums)
            else:
                real_mask,mask_map,forbid_mask = self.get_input_mask(token_id,curmap.device)
            # mask_maps[0,get_other_ids(token_id)] -= 1* mask_map
            mask_maps[0,sub_prompt_ids] = 0.20 * mask_map
            mask_maps[0,token_id] = 0.20 * mask_map * w
            # negative_mask_map[0,sub_prompt_ids] = 1-mask_map
            save_mask_maps.append(mask_maps[0,token_id].clone())
            save_real_maps.append(real_mask.clone())
            # self_maps[0,:] += mask_map.reshape(-1,1) * (1 - mask_map.reshape(1,-1))
            # negative_self_mask_map.append(mask_maps[0,token_id].clone())
            mean_attentions -=  (forbid_mask.reshape(1,-1)>0) * mean_attentions * 3
            # save_mask_maps[:,token_id] = mask_maps[:,token_id]
        save_mask_maps = torch.stack(save_mask_maps)  # N x H x W


        maskdict = {}
        sreg_maps = {}
        negative_mask_maps = {}
        sreg_maps = {}
        negative_self_maps = {}
        for r in range(4):
            res = int(64/np.power(2,r))
            layouts_s = nnf.interpolate(save_mask_maps.unsqueeze(1),(res, res),mode='nearest')
            layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0)
            sreg_maps[np.power(res, 2)] = layouts_s
            layout_c = nnf.interpolate(mask_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1)
            maskdict[np.power(res, 2)] = layout_c


        self.mask = maskdict 
        self.self_mask = sreg_maps
        self.negative_mask_maps = negative_mask_maps
        self.negative_self_maps = negative_self_maps


        return None
    def __init__(self, save_global_store=True,end_step = 15,token_dict = None):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        # super(AttentionStore, self).__init__()
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        ####
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        #### maskdiffusion parameter
        self.text_cond = None
        self.mask = None
        self.end_step = end_step
        self.token_dict = token_dict
        #### 
        self.timesteps = None
        self.self_mask = None
        self.input_mask = None

