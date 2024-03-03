import glob

import numpy as np
import torch
import os
import argparse

from models.archs.network_restormer_guided_arch import RestormerRefFusion
from models.archs.network_nafnet_guided_arch import NAFNetRefFusion
from models.archs.network_drsformer_guided_arch import DRSformerRefFusion
from models.archs.network_drsformer_guided_arch_200L_SPA import DRSformer200L_SPA_RefFusion
from models.archs.network_sfnet_guided_arch import SFNetRefFusion
from models.archs.network_promptir_guided_arch import PromptIRRefFusion
from utils.utils_image import img2tensor, tensor2img, imfrombytes
from metrics.psnr_ssim import calculate_ssim, calculate_psnr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)

    args = parser.parse_args()

    if args.task == "promptir_derain":
        model = PromptIRRefFusion(
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            decoder=False,

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )
    elif args.task == "promptir_dehaze":
        model = PromptIRRefFusion(
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            decoder=False,

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )
    elif args.task == "nafnet_deblur":
        model = NAFNetRefFusion(
            img_channel=3,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1],
            nf=64,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )

    elif args.task == "restormer_deblur":
        model = RestormerRefFusion(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )

    elif args.task == "restormer_dp_8bit":
        model = RestormerRefFusion(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )
    elif args.task == "restormer_dp_16bit":
        model = RestormerRefFusion(
            inp_channels=6,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=True,

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )

    elif args.task == "sfnet_dehaze":
        mmodel = SFNetRefFusion(
            mode="train",
            num_res=16,

            nf=32,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],

        )

    elif args.task == "drsformer_derain_200L":

        model = DRSformer200L_SPA_RefFusion(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )

    elif args.task == "drsformer_derain_200h_did_ddn":

        model = DRSformerRefFusion(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",

            nf=48,
            ext_n_blocks=[4, 4, 4, 4],
            reffusion_n_blocks=[2, 2, 2, 2],
            reffusion_n_blocks_middle=1,
            scale=1,
            num_nbr=1,
            psize=3,
            lr_block_size=8,
            ref_down_block_size=1.5,
            dilations=[1, 2, 3],
        )

    model.load_state_dict(torch.load(args.checkpoint_path)["params"])
    model = model.cuda()
    model.eval()

    inp_img_paths = sorted(glob.glob(os.path.join(args.dataroot, "lq", "*")))
    gt_img_paths = sorted(glob.glob(os.path.join(args.dataroot, "hq", "*")))
    ref_img_paths = sorted(glob.glob(os.path.join(args.dataroot, "ref", "*")))

    psnr_list = []
    ssim_list = []

    for inp_path, gt_path, ref_path in zip(inp_img_paths, gt_img_paths, ref_img_paths):

        f = open(inp_path, "rb")
        img_bytes = f.read()
        lq_img = imfrombytes(img_bytes, flag="color", float32=True)

        f = open(gt_path, "rb")
        img_bytes = f.read()
        gt_img = imfrombytes(img_bytes, flag="color", float32=True)

        f = open(ref_path, "rb")
        img_bytes = f.read()
        ref_img = imfrombytes(img_bytes, flag="color", float32=True)

        with torch.no_grad():
            pred = model(lq_img, ref_img)

        pred = tensor2img(pred)
        gt_img = tensor2img(gt_img)

        psnr_ = calculate_psnr(pred, gt_img, crop_border=0)
        ssim_ = calculate_ssim(pred, gt_img, crop_border=0)

        psnr_list.append(psnr_)
        ssim_list.append(ssim_)

    print(np.array(psnr_list).mean(), np.array(ssim_list).mean())