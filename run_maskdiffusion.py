#######
# Code refactored to implement the maskdiffusion.
# The original implementation was created in April 2023 for a conference submission.
# The new implementation reuses code from "attend-and-excite" and "densediffusion" to achieve a more simple implementation on 2023.9.20.
#######
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline

from utils import ptp_utils
from utils.attention_mask import MaskdiffusionStore
from utils.ptp_utils import AttentionStore


MY_TOKEN = '<replace with your token>'
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
SAVE_PATH = "./generated_images/test/"


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载 Stable Diffusion 模型
stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
stable = StableDiffusionPipeline.from_pretrained(
    stable_diffusion_version, safety_checker=None
).to(device)

# 配置 Stable Diffusion 的调度器
tokenizer = stable.tokenizer
stable.scheduler = DDIMScheduler.from_config(stable.scheduler.config)
stable.scheduler.set_timesteps(50)



@torch.no_grad()
def run_on_prompt(prompts: List[str],
                  pipe: StableDiffusionPipeline,
                  controller: AttentionStore,
                  seed: torch.Generator,
                  mask_dict=None) -> Image.Image:
    """
    Generate an image based on a list of prompts using a Stable Diffusion pipeline.

    Args:
        prompts (List[str]): List of prompts. The first is the main prompt, and the rest are sub-prompts.
        pipe (StableDiffusionPipeline): Pre-trained Stable Diffusion pipeline.
        controller (AttentionStore): Controller to manage attention control.
        seed (torch.Generator): Random seed for reproducibility.
        mask_dict (dict, optional): Masking dictionary, if any.

    Returns:
        Image.Image: Generated image.
    """

    # Step 1: Register attention control, if applicable
    if controller is not None:
        ptp_utils.register_MA_attention_control(pipe, controller)

    # Step 2: Encode text embeddings for prompts
    text_input = pipe.tokenizer(
        prompts,
        padding="max_length",
        return_length=True,
        return_overflowing_tokens=False,
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    cond_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]

    # Encode unconditional (blank) embedding
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]

    # Step 3: Swap sub-prompt embeddings into the main prompt
    def find_and_replace_embedding(main_ids, sub_ids, cond_embeddings, i):
        """Find and replace sub-prompt embeddings in the main prompt."""
        wlen = len(sub_ids)  # Length of sub-prompt tokens
        for j in range(len(main_ids) - wlen + 1):  # Search within the main prompt tokens
            if torch.all(main_ids[j:j + wlen] == sub_ids):
                # Replace the embedding for the matched tokens
                cond_embeddings[0][j:j + wlen] = cond_embeddings[i][1:1 + wlen]
                return [k for k in range(j, j + wlen)]  # Return matched indices
        raise ValueError(f"Sub-prompt '{prompts[i]}' not found in the main prompt!")

    # Extract token IDs for the main prompt and initialize lists
    main_ids = text_input['input_ids'][0]
    indice_lists = []

    # Process each sub-prompt
    for i in range(1, len(prompts)):
        sub_ids = text_input['input_ids'][i][1:text_input['length'][i] - 1]  # Remove special tokens
        indice_lists.append(find_and_replace_embedding(main_ids, sub_ids, cond_embeddings, i))

    # Step 4: Build token dictionary for tracking
    token_dict = {indices[-1]: indices for indices in indice_lists}
    controller.token_dict = token_dict
    controller.text_cond = torch.cat([uncond_embeddings, cond_embeddings[0].unsqueeze(0)])
    controller.timesteps = pipe.scheduler.timesteps
    controller.text_length = text_input['length'][0]

    print(f"Token Dictionary: {token_dict}, Main Prompt: {prompts[0]}")


    # Step 5: Generate image using the pipeline
    outputs = pipe(
        prompt=prompts[0],
        generator=seed,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=512,
        width=512
    )
    return outputs.images[0]



promptsarr = []

# 读取文件并将每行添加到列表中
with open("prompts/demo_prompt.txt", "r") as f:
    for line in f:
        promptsarr.append(line.strip())  # 使用 strip() 去除行首尾的换行符和多余空格


# 遍历 promptsarr
for i in range(len(promptsarr)):
    target_text = promptsarr[i]
    
    g = torch.Generator('cuda').manual_seed(0)

    # 分割文本
    before_and, and_word, after_and = target_text.partition(' and ')
    
    # 创建保存路径
    controller = MaskdiffusionStore()
    save_path = os.path.join(SAVE_PATH, target_text)  # 用 os.path.join 拼接路径
    
    # 创建目录
    os.makedirs(save_path, exist_ok=True)
    controller.save_path = save_path

    # 生成图像
    image = run_on_prompt(prompts=[target_text, before_and, after_and],
                            pipe=stable,
                            controller=controller,
                            seed=g)


    # 保存生成的图像
    img_mask_diff = image
    img_mask_diff.save(os.path.join(save_path, f"save_img.png"))
