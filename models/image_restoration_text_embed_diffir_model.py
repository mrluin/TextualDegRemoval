import importlib
import torch
import math
import torch.nn as nn

from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from utils.logger import get_root_logger
from models.archs import define_network
from models.base_model import BaseModel
from utils.utils_image import basicsr_imwrite, tensor2img
from models.dino.vision_transformers import vit_base


from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


loss_module = importlib.import_module('losses')
metric_module = importlib.import_module('metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class Mapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
    ):
        super(Mapper, self).__init__()

        self.num_words = num_words

        for i in range(self.num_words):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                              nn.LayerNorm(1280),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1280, 1280),
                                                              nn.LayerNorm(1280),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1280, 1280),
                                                              nn.LayerNorm(1280),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1280, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        embs = embs[0]

        for i in range(self.num_words):
            hidden_state = getattr(self, f'mapping_{i}')(embs[:, :1]) + getattr(self, f'mapping_patch_{i}')(embs[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class DiffIRRefGuidedImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(DiffIRRefGuidedImageCleanModel, self).__init__(opt)

        # define network

        # self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        # if self.mixing_flag:
        #     mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
        #     use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
        #     self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # feature extractor, DINOv2
        # self.net_ext = vit_base(img_size=518,
        #                         patch_size=14,
        #                         init_values=1.0,
        #                         ffn_layer='mlp',
        #                         block_chunks=0)
        #
        # self.net_ext.load_state_dict(
        #     torch.load(self.opt['path']['pretrain_network_dino'], map_location='cpu'),
        #     strict=True
        # )
        # self.net_ext = self.net_ext.to(self.device)

        ###################
        self.clip_vision_model = CLIPVisionModel.from_pretrained("/hdd/linjingbo/CLIP-ViT-H-14-laion2b_s32b_b79k")
        for param in self.clip_vision_model.parameters():
            param.requires_grad_(False)

        self.clip_vision_model = self.clip_vision_model.to(self.device)
        self.clip_vision_model.eval()

        self.mapper = Mapper(input_dim=1280, output_dim=1024, num_words=20)
        self.mapper.load_state_dict(torch.load("/hdd/linjingbo/TextualDegRemoval/experiments/it2_tr_mapping/mapper_018000.pt"), strict=False)
        for param in self.mapper.parameters():
            param.requires_grad_(False)

        self.mapper = self.mapper.to(self.device)
        self.mapper.eval()

        #######################

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', False),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        #################################

    def init_training_settings(self):
        self.net_g.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # self.param_fix_iterations = self.opt['train']['fix_iterations'] if 'fix_iterations' in train_opt else None

        optim_params = []
        # optim_ref_params = []

        for k, v in self.net_g.named_parameters():

            # if "masa" in k:
            #     optim_ref_params.append(v)
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} appended to Ref Params.')
            # else:
            #     optim_params.append(v)
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} appended to Normal Params.')

            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # G_optim_params = [
        #     {
        #         "params": optim_params,
        #         "lr": train_opt["optim_g"]["lr"]
        #     },
        #     {
        #         "params": optim_ref_params,
        #         "lr": train_opt["optim_g"]["ref_lr"]
        #     }
        # ]

        optim_type = train_opt['optim_g'].pop('type')

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):

        self.lq = data['lq'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'ref' in data:
            self.ref = data['ref'].to(self.device)

        if 'clip_inp' in data:
            self.clip_inp = data['clip_inp'].to(self.device)

        clip_inp = F.interpolate(self.clip_inp, (224, 224), mode='bilinear')
        image_features = self.clip_vision_model(clip_inp, output_hidden_states=True)
        image_embeddings = [image_features[0]]
        image_embeddings = [emb.detach() for emb in image_embeddings]
        inj_embedding = self.mapper(image_embeddings)

        self.embed = inj_embedding

        # if self.mixing_flag:
        #     self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):

        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'ref' in data:
            self.ref = data['ref'].to(self.device)
        if 'clip_inp' in data:
            self.clip_inp = data['clip_inp'].to(self.device)

    def optimize_parameters(self, current_iter):

        ###############
        # for name, param in self.net_ext.named_parameters():
        #     param.requires_grad_(False)
        #
        # if self.param_fix_iterations is not None:
        #     if current_iter < self.param_fix_iterations:
        #         for name, param in self.net_g.named_parameters():
        #             if "masa" in name:
        #                 param.requires_grad_(False)
        # else:
        #     for name, param in self.net_g.named_parameters():
        #         param.requires_grad_(True)
        ###############

        # B, C, train_h, train_w = self.lq.shape
        # train_patch_size = train_h
        # stride = int(train_h // 4)
        #
        # with torch.no_grad():
        #     unfold_ref = torch.nn.functional.unfold(self.ref.clone(),
        #                                             kernel_size=(train_patch_size, train_patch_size),
        #                                             stride=(stride, stride))
        #     _, L, N = unfold_ref.shape
        #     unfold_ref = unfold_ref.transpose(-1, -2).contiguous().view(B * N, C, train_patch_size, train_patch_size)
        #     lq_match_in = F.interpolate(self.lq.clone(), size=(int(math.ceil(train_h / 14) * 14), int(math.ceil(train_w / 14) * 14)), mode="bilinear")
        #     feature_L = self.net_ext(lq_match_in)
        #     ref_match_in = F.interpolate(unfold_ref.clone(), size=(int(math.ceil(train_h / 14) * 14), int(math.ceil(train_w / 14) * 14)), mode="bilinear")
        #
        #     feature_ref = self.net_ext(ref_match_in)
        #     feature_L = feature_L.view(B, 1, -1)
        #     feature_ref = feature_ref.view(B, N, -1)
        #     corr = torch.matmul(torch.nn.functional.normalize(feature_L, dim=-1),
        #                         torch.nn.functional.normalize(feature_ref, dim=-1).transpose(-1, -2))
        #
        #     topk_value, topk_indices = torch.topk(corr, k=1, dim=-1)
        #
        #     topk_indices = topk_indices[:, :, 0]
        #     topk_indices = topk_indices[:, :, None, None, None].expand(-1, -1, C, train_patch_size, train_patch_size)
        #
        #     unfold_ref = unfold_ref.view(B, N, C, train_patch_size, train_patch_size)
        #     ref_in = torch.gather(unfold_ref, dim=1, index=topk_indices).squeeze(1)
        #
        #     self.ref_in = ref_in
        #
        #     del lq_match_in
        #     del ref_match_in
        #     del unfold_ref

        ####################################



        # print(inj_embedding.shape)

        #####################################

        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq, self.embed.detach())

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        # if current_iter % 2000 == 0:
        #     L = [tensor2img(img_tensor.detach()) for img_tensor in self.lq][0]
        #     H = [tensor2img(img_tensor.detach()) for img_tensor in self.gt][0]
        #     E = [tensor2img(img_tensor.detach()) for img_tensor in self.output][0]
        #     R = [tensor2img(img_tensor.detach()) for img_tensor in self.ref_in][0]
        #
        #     LHER = np.concatenate([L, H, E, R], axis=1)
        #
        #     basicsr_imwrite(LHER, os.path.join("./intermediate_results", f"{current_iter:06d}.png"), rgb2bgr=False)

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img, self.embed)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, self.embed)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_train_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            # print("1", self.output.shape)
            # print(sr_img.shape)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                basicsr_imwrite(sr_img, save_img_path)
                basicsr_imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
