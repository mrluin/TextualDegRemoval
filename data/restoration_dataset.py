from packaging import version
import glob
import os
import torchvision
import random
import torch
import PIL
import numpy as np

from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.utils_data import (paired_paths_from_folder,
                             tri_DP_paths_from_folder,
                             paired_DP_paths_from_folder,
                             paired_paths_from_lmdb,
                             paired_paths_from_meta_info_file,
                             tri_paths_from_folder)

from data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation

from utils.file_client import FileClient
from utils.utils_image import imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP, uint2single, single2uint


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['hqroot'], opt['lqroot']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


# derain, deblur
class Dataset_PairedImageWithRef(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImageWithRef, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.ref_folder = opt['hqroot'], opt['lqroot'], opt['refroot']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = tri_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.ref_folder], ['lq', 'gt', 'ref'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        ref_path = self.paths[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        try:
            img_ref = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("ref path {} not working".format(ref_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_ref = img2tensor([img_gt, img_lq, img_ref], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'ref': img_ref,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


# deblur dualpixel 16bit
class Dataset_PairedImageWithRef_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImageWithRef_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder_L, self.lq_folder_R, self.ref_folder = opt['hqroot'], opt['lqroot_L'], \
                                                                              opt['lqroot_R'], opt['refroot']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # if self.io_backend_opt['type'] == 'lmdb':
        #     self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        #     self.io_backend_opt['client_keys'] = ['lq', 'gt']
        #     self.paths = paired_paths_from_lmdb(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:

        self.paths = tri_DP_paths_from_folder(
            [self.lq_folder_L, self.lq_folder_R, self.gt_folder, self.ref_folder], ['lqL', 'lqR', 'gt', 'ref'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqL_path, 'lqR')
        try:
            img_lqR = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lqR_path))

        ref_path = self.paths[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        try:
            img_ref = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("ref path {} not working".format(ref_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lqL, img_lqR, img_ref = img2tensor([img_gt, img_lqL, img_lqR, img_ref], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'ref': img_ref,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


# gaussian denoising
class Dataset_GaussianDenoisingWithRef(data.Dataset):
    def __init__(self, opt):
        super(Dataset_GaussianDenoisingWithRef, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.sigma_type = opt['sigma_type']
        self.sigma_range = opt['sigma_range']
        self.sigma_test = opt['sigma_test'] if 'sigma_test' in opt else None
        self.in_ch = opt['in_ch']

        self.gt_folder, self.lq_folder, self.ref_folder = opt['hqroot'], None, opt['refroot']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.gt_folder, self.ref_folder], ['gt', 'ref'], self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)

        lq_path = None
        img_lq = img_gt.copy()

        ref_path = self.paths[index]['ref_path']
        img_bytes = self.file_client.get(ref_path, 'ref')
        if self.in_ch == 3:
            try:
                img_ref = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("ref path {} not working".format(ref_path))
        else:
            try:
                img_ref = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(ref_path))

            img_ref = np.expand_dims(img_ref, axis=2)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq, img_ref = img2tensor([img_gt, img_lq, img_ref], bgr2rgb=True, float32=True)

            # add noise
            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)
            else:
                raise NotImplementedError

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)
        else:

            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            img_gt, img_lq, img_ref = img2tensor([img_gt, img_lq, img_ref], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        if lq_path is None:
            lq_path = gt_path

        return {
            'lq': img_lq,
            'gt': img_gt,
            'ref': img_ref,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


# dehaze
class Dataset_PairedDehazeWithRef(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedDehazeWithRef, self).__init__()

        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.ref_folder = opt['hqroot'], opt['lqroot'], opt['refroot']
        self.meta_info_file_path = opt['meta_info_file']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.lq_paths = []
        self.hq_paths = []
        self.ref_paths = []

        with open(self.meta_info_file_path) as f:
            contents = f.readlines()
            haze_names_indoor = [i.strip() for i in contents]
            gt_names_indoor = [i.split('_')[0] for i in haze_names_indoor]
            ref_names_indoor = [i.split("_")[0] for i in haze_names_indoor]

        self.lq_paths.extend([os.path.join(self.lq_folder, i) for i in haze_names_indoor])
        self.hq_paths.extend([os.path.join(self.gt_folder, i) for i in gt_names_indoor])
        self.ref_paths.extend([os.path.join(self.ref_folder, i) for i in ref_names_indoor])

        # if self.io_backend_opt['type'] == 'lmdb':
        #     self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        #     self.io_backend_opt['client_keys'] = ['lq', 'gt']
        #     self.paths = paired_paths_from_lmdb(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = tri_paths_from_folder(
        #         [self.lq_folder, self.gt_folder, self.ref_folder], ['lq', 'gt', 'ref'],
        #         self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):

        # if self.file_client is None:
        #     self.file_client = FileClient(
        #         self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.lq_paths)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.hq_paths[index]
        f = open(gt_path, "rb")
        img_bytes = f.read()
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.lq_paths[index]
        f = open(lq_path, "rb")
        img_bytes = f.read()
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        ref_path = self.ref_paths[index]
        f = open(ref_path, "rb")
        img_bytes = f.read()
        try:
            img_ref = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("ref path {} not working".format(ref_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_ref = img2tensor([img_gt, img_lq, img_ref], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'ref': img_ref,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.lq_paths)


# all-in-one
class Dataset_PairedUnifiedWithRef(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedUnifiedWithRef, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.image_paths_hq_denoise = []
        self.image_paths_hq_derain = []
        self.image_paths_hq_dehaze = []

        self.image_paths_lq_derain = []
        self.image_paths_lq_dehaze = []

        self.image_paths_ref_denoise = []
        self.image_paths_ref_derain = []
        self.image_paths_ref_dehaze = []

        for dataroot in self.opt['denoise_root']:
            self.image_paths_hq_denoise.extend(sorted(glob.glob(os.path.join(dataroot, "hq", "*"))))
            self.image_paths_ref_denoise.extend(sorted(glob.glob(os.path.join(dataroot, "ref", "*"))))

        for dataroot in self.opt['derain_root']:
            self.image_paths_hq_derain.extend(sorted(glob.glob(os.path.join(dataroot, "hq", "*"))))
            self.image_paths_lq_derain.extend(sorted(glob.glob(os.path.join(dataroot, "lq", "*"))))
            self.image_paths_ref_derain.extend(sorted(glob.glob(os.path.join(dataroot, "ref", "*"))))

        for dataroot in self.opt['dehaze_root']:
            meta_info_file = os.path.join(dataroot, "meta_info.txt")

            with open(meta_info_file) as f:
                contents = f.readlines()
                haze_names = [i.strip() for i in contents]
                gt_names = [i.split('_')[0] for i in haze_names]
                ref_names = [i.split("_")[0] for i in haze_names]

            self.image_paths_lq_dehaze.extend([os.path.join(dataroot, "lq", i) for i in haze_names])
            self.image_paths_hq_dehaze.extend([os.path.join(dataroot, "hq", i) for i in gt_names])
            self.image_paths_ref_dehaze.extend([os.path.join(dataroot, "ref", i) for i in ref_names])

        # self.gt_folder, self.lq_folder, self.ref_folder = opt['hqroot'], opt['lqroot'], opt['refroot']
        # if 'filename_tmpl' in opt:
        #     self.filename_tmpl = opt['filename_tmpl']
        # else:
        #     self.filename_tmpl = '{}'
        #
        # if self.io_backend_opt['type'] == 'lmdb':
        #     self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        #     self.io_backend_opt['client_keys'] = ['lq', 'gt']
        #     self.paths = paired_paths_from_lmdb(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = tri_paths_from_folder(
        #         [self.lq_folder, self.gt_folder, self.ref_folder], ['lq', 'gt', 'ref'],
        #         self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        # if self.file_client is None:
        #     self.file_client = FileClient(
        #         self.io_backend_opt.pop('type'), **self.io_backend_opt)

        task_choice = random.choice([0, 1, 2, 3, 4])

        # 0 denoise 15
        # 1 denoise 25
        # 2 denoise 50
        # 3 derain
        # 4 dehaze

        if task_choice == 0 or task_choice == 1 or task_choice == 2:
            gt_path = self.image_paths_hq_denoise[index % len(self.image_paths_hq_denoise)]
            lq_path = gt_path
            ref_path = self.image_paths_ref_denoise[index % len(self.image_paths_ref_denoise)]
        elif task_choice == 3:
            lq_path = self.image_paths_lq_derain[index % len(self.image_paths_lq_derain)]
            gt_path = self.image_paths_hq_derain[index % len(self.image_paths_hq_derain)]
            ref_path = self.image_paths_ref_derain[index % len(self.image_paths_ref_derain)]
        elif task_choice == 4:
            lq_path = self.image_paths_lq_dehaze[index % len(self.image_paths_lq_dehaze)]
            gt_path = self.image_paths_hq_dehaze[index % len(self.image_paths_hq_dehaze)]
            ref_path = self.image_paths_ref_dehaze[index % len(self.image_paths_ref_dehaze)]

        else:
            raise NotImplementedError

        scale = self.opt['scale']
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        f = open(gt_path, "rb")
        img_bytes = f.read()
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        f = open(lq_path, "rb")
        img_bytes = f.read()
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        ref_path = open(ref_path, "rb")
        img_bytes = f.read()
        try:
            img_ref = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("ref path {} not working".format(ref_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_ref = img2tensor([img_gt, img_lq, img_ref], bgr2rgb=True, float32=True)

        # add noise when denoising
        if task_choice == 0:
            noise_level = torch.FloatTensor([15]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)
        elif task_choice == 1:
            noise_level = torch.FloatTensor([25]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)
        elif task_choice == 2:
            noise_level = torch.FloatTensor([50]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'ref': img_ref,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.image_paths_hq_denoise) + len(self.image_paths_hq_derain) + len(self.image_paths_hq_dehaze)



# class Dataset_PairedImageWithClipInput(data.Dataset):
#     def __init__(self, opt):
#         super(Dataset_PairedImageWithClipInput, self).__init__()
#
#         self.opt = opt
#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None
#
#         self.gt_folder, self.lq_folder = opt['hqroot'], opt['lqroot']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'
#
#         if self.io_backend_opt['type'] == 'lmdb':
#             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
#             self.io_backend_opt['client_keys'] = ['lq', 'gt']
#             self.paths = paired_paths_from_lmdb(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'])
#         elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
#             self.paths = paired_paths_from_meta_info_file(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 self.opt['meta_info_file'], self.filename_tmpl)
#         else:
#             self.paths = paired_paths_from_folder(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 self.filename_tmpl)
#
#         if self.opt['phase'] == 'train':
#             self.geometric_augs = opt['geometric_augs']
#
#         interpolation = "bicubic"
#
#         self.interpolation = {
#             "linear": PIL_INTERPOLATION["linear"],
#             "bilinear": PIL_INTERPOLATION['bilinear'],
#             "bicubic": PIL_INTERPOLATION["bicubic"],
#             "lanczos": PIL_INTERPOLATION["lanczos"]
#         }[interpolation]
#
#     def get_tensor_clip(self, normalize=True, toTensor=True):
#         transform_list = []
#         if toTensor:
#             transform_list += [torchvision.transforms.ToTensor()]
#
#         if normalize:
#             transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
#                                                                 (0.26862954, 0.26130258, 0.27577711))]
#
#         return torchvision.transforms.Compose(transform_list)
#
#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)
#
#         scale = self.opt['scale']
#         index = index % len(self.paths)
#         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
#         # image range: [0, 1], float32.
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         try:
#             img_gt = imfrombytes(img_bytes, float32=True)
#         except:
#             raise Exception("gt path {} not working".format(gt_path))
#
#         lq_path = self.paths[index]['lq_path']
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         try:
#             img_lq = imfrombytes(img_bytes, float32=True)
#         except:
#             raise Exception("lq path {} not working".format(lq_path))
#
#         ####### clip input
#         lq_image = Image.open(lq_path)
#
#         if not lq_image.mode == "RGB":
#             lq_image = lq_image.convert("RGB")
#
#         H, W = lq_image.size
#
#         # center crop
#         if H < W:
#             croper = torchvision.transforms.CenterCrop(H)
#         else:
#             croper = torchvision.transforms.CenterCrop(W)
#
#         lq_image = croper(lq_image)
#         lq_image = np.array(lq_image)
#
#         lq_image = uint2single(lq_image)
#         lq_image = single2uint(lq_image)
#
#         lq_image_tensor = Image.fromarray(lq_image.astype('uint8')).resize((224, 224), resample=self.interpolation)
#         clip_inp = self.get_tensor_clip()(lq_image_tensor)
#         ###############################
#         # ref_path = self.paths[index]['ref_path']
#         # img_bytes = self.file_client.get(ref_path, 'ref')
#         # try:
#         #     img_ref = imfrombytes(img_bytes, float32=True)
#         # except:
#         #     raise Exception("ref path {} not working".format(ref_path))
#
#         # augmentation for training
#         if self.opt['phase'] == 'train':
#             gt_size = self.opt['gt_size']
#             # padding
#             img_gt, img_lq = padding(img_gt, img_lq, gt_size)
#
#             # random crop
#             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)  # BGR, [H, W, C]
#
#             # flip, rotation augmentations
#             if self.geometric_augs:
#                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
#
#         # BGR to RGB, HWC to CHW, numpy to tensor
#         img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
#         # normalize
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_gt, self.mean, self.std, inplace=True)
#
#         return {
#             'lq': img_lq,
#             'gt': img_gt,
#             'clip_inp': clip_inp,
#             'lq_path': lq_path,
#             'gt_path': gt_path
#         }
#
#     def __len__(self):
#         return len(self.paths)