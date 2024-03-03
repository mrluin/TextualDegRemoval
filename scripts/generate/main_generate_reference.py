import os
import torch
import numpy as np
import torch.nn as nn

import cv2

from PIL import Image
from typing import Optional, Tuple
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from scripts.train.main_train_i2t_mapping import Mapper, th2image
from scripts.train.main_train_tr_mapping import CleanMapper, inj_forward_text, inj_forward_crossattention, validation, reshape_batch_dim_to_heads, reshape_heads_to_batch_dim
from data.guidance_generation_dataset import ReferenceGenerationDataset


def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images

def pww_load_tools(
        device: str = "cuda:0",
        scheduler_type = DDIMScheduler,
        global_mapper_path: Optional[str] = None,
        clean_mapper_path: Optional[str] = None,
        diffusion_model_path: Optional[str] = None,
        model_token: Optional[str] = None,
        num_words: Optional[int] = 20,
) -> Tuple[
    UNet2DConditionModel,
    CLIPTextModel,
    CLIPTokenizer,
    AutoencoderKL,
    CLIPVisionModel,
    Mapper,
    CleanMapper,
    DDIMScheduler,
]:

    local_path_only = diffusion_model_path is not None
    vae = AutoencoderKL.from_pretrained(
        diffusion_model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only
    )

    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_path, torch_dtype=torch.float16, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_path, torch_dtype=torch.float16, subfolder="text_encoder")
    image_encoder = CLIPVisionModel.from_pretrained(diffusion_model_path, torch_dtype=torch.float16)

    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    mapper = Mapper(input_dim=1280, output_dim=1024, num_words=num_words)
    clean_mapper = CleanMapper(input_dim=1024, output_dim=1024, num_words=num_words)

    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "Attention":

            _module.__class__.reshape_heads_to_batch_dim = reshape_heads_to_batch_dim
            _module.__class__.reshape_batch_dim_to_heads = reshape_batch_dim_to_heads

            if 'attn1' in _name: continue

            print("Attention added successfully")

            _module.__class__.__call__ = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

    mapper.load_state_dict(torch.load(global_mapper_path, map_location="cpu"))
    mapper.half()

    clean_mapper.load_state_dict(torch.load(clean_mapper_path, map_location="cpu"))
    clean_mapper.half()

    for _name, _module in unet.named_modules():

        _module.__class__.reshape_heads_to_batch_dim = reshape_heads_to_batch_dim
        _module.__class__.reshape_batch_dim_to_heads = reshape_batch_dim_to_heads

        if 'attn1' in _name: continue

        print("Attention added successfully")
        if _module.__class__.__name__ == "Attention":
            _module.add_module('to_k_global', getattr(mapper, f'{_name.replace(".", "_")}_to_k'))
            _module.add_module('to_v_global', getattr(mapper, f'{_name.replace(".", "_")}_to_v'))

    vae.to(device), unet.to(device), text_encoder.to(device), image_encoder.to(device), mapper.to(device), clean_mapper.to(device)

    scheduler = DDIMScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")

    vae.eval()
    unet.eval()
    image_encoder.eval()
    text_encoder.eval()
    mapper.eval()
    clean_mapper.eval()

    return vae, unet, text_encoder, tokenizer, image_encoder, mapper, clean_mapper, scheduler


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_index", type=str, default="full")

    parser.add_argument("--inference_data_dir", type=str, default=None, required=True)

    parser.add_argument("--i2t_mapper_path", type=str)
    parser.add_argument("--tr_mapper_path", type=str)
    parser.add_argument("--num_words", type=int)
    parser.add_argument("--range_index_left", type=int, default=None)
    parser.add_argument("--range_index_right", type=int, default=None)

    parser.add_argument("--pretrained_stable_diffusion_path", type=str)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--placeholder_token", type=str, default="S")
    parser.add_argument("--template", type=str, default="a photo of a {}")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vae, unet, text_encoder, tokenizer, image_encoder, mapper, clean_mapper, scheduler = pww_load_tools(
        "cuda:0",
        DDIMScheduler,
        diffusion_model_path=args.pretrained_stable_diffusion_path,
        global_mapper_path=args.i2t_mapper_path,
        clean_mapper_path=args.tr_mapper_path
    )

    dataset = ReferenceGenerationDataset(
        dataroot=args.inference_data_dir,
        range=None if args.range_index_left is None and args.range_index_right is None else [args.range_index_left, args.range_index_right],
        tokenizer=tokenizer,
        size=512,
        placeholder_token=args.placeholder_token,
        template=args.template,
    )

    print(len(dataset))

    dataloder = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for step, batch in enumerate(dataloder):

        batch["pixel_values"] = batch["pixel_values"].to("cuda:0")
        batch["pixel_values_clip"] = batch["pixel_values_clip"].to("cuda:0").half()
        batch["input_ids"] = batch["input_ids"].to("cuda:0")
        batch["index"] = batch["index"].to("cuda:0").long()

        print(step, batch["text"])

        syn_images = validation(batch, tokenizer, image_encoder, text_encoder, unet, mapper, clean_mapper,
                                vae, batch["pixel_values_clip"].device, 5, token_index=args.token_index,
                                seed=args.seed, pretrained_path=args.stable_diffusion_path)

        for index, syn in enumerate(syn_images):
            Image.fromarray(np.array(syn)).save(os.path.join(args.output_dir, f"{batch['image_name'][index]}.png"))





