MaskDiffusion: Boosting Text-to-Image Consistency with Conditional Mask




## Setup

### Environment
Our code builds on the requirement of the official [Stable Diffusion repository](https://github.com/CompVis/stable-diffusion). To set up their environment, please run:

```
conda env create -f environment/environment.yaml
conda activate maskdiffusion

```

### Test with the mini-testset

```
python run_maskdiffusion.py
```




## Acknowledgements 
This orignal code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library and [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/). When reorganizing the code to be public, I also reused code from [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite) and [Densediffusion](https://github.com/naver-ai/DenseDiffusion), to achieve a more concise implementation. 

