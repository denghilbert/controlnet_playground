from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import clip
from PIL import Image

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        # Full length features
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, :77, :]
        # Dump last 30
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, :47, :]
        # Dump last 70
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, :7, :]
        # Dump last 74
        # This is the threshold for dog case
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, :3, :]
        # Dump last 76
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, :1, :]
        # Dump top1 and generate with last 76
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, 1:, :]


        # Random noise
        #cond['c_crossattn'][0] = torch.normal(0, 1, cond['c_crossattn'][0].shape).cuda()
        # Duplicate top1 for 77 times
        # This is identical to only w/ top1, because we will do cross attention (5, 77, 768)^T * (5, 77, 768) == (5, 1, 768)^T * (5, 1, 768)
        #cond['c_crossattn'][0] = cond['c_crossattn'][0][:, :1 :].repeat(1, 77, 1)

        # zero out last 30
        #cond['c_crossattn'][0][:, 47:77, :] = 0
        # zero out last 70
        #cond['c_crossattn'][0][:, 7:77, :] = 0
        # zero out last 74
        #cond['c_crossattn'][0][:, 3:77, :] = 0


        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_clip, preprocess = clip.load("ViT-L/14", device=device)

        # I modify the return value of clip.encode_text, return n*77*768 features
        # use this features to generate images
        tokens = clip.tokenize(['fluffy white dog'], context_length=77).cuda()
        text_features, full_length_features = model_clip.encode_text(tokens)
        #cond['c_crossattn'][0] = full_length_features.repeat(5, 1, 1).float()

        # interpolation of red and white from text
        cond['c_crossattn'][0] = (1 * full_length_features.repeat(5, 1, 1).float() + 5 * cond['c_crossattn'][0]) / 6


        image = preprocess(Image.open("/home/youming/Desktop/controlnet_playground/test_i2i/white_dog2.png")).unsqueeze(0).to(device)
        image_features = model_clip.encode_image(image)
        image_features = image_features.unsqueeze(0).repeat(5, 1, 1)


        ## cond['c_crossattn'][0] = torch.cat((cond['c_crossattn'][0], image_features.float()), 1)
        #cond['c_crossattn'][0][:, 2:3, :] = image_features.float()
        #cond['c_crossattn'][0][:, 3:4, :] = image_features.float()
        #cond['c_crossattn'][0][:, 4:5, :] = image_features.float()

        print(cond['c_crossattn'][0].shape)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

from PIL import Image
import os

test_dir = 'test/interpolation_red_white_1white5red'
test_prompt = 'fluffy red dog'

img = process(np.uint8(Image.open('./test_imgs/dog.png')), test_prompt, '', '', 5, 512, 50, True, 1, 7, 971431670, 0.0, 100, 200)
os.makedirs(test_dir, exist_ok=True)

for i in range(len(img)):
    Image.fromarray(img[i]).save('./' + test_dir + '/{}.png'.format(i))

Image.open('./test_imgs/dog.png').save('./' + test_dir + '/dog_gt.png')
