from typing import List, Optional

import albumentations as albu


def get_transforms(size: int, crop='random'):
    aug_fn = albu.ShiftScaleRotate(always_apply=True, scale_limit=.5, rotate_limit=30)
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]
    pad = albu.PadIfNeeded(size, size)

    pipeline = albu.Compose([
        aug_fn,
        pad,
        crop_fn
    ], keypoint_params={'format': 'xy', 'remove_invisible': False, })

    def process(img, kpts):
        r = pipeline(image=img, keypoints=kpts)
        return r['image'], r['keypoints']

    return process


def get_normalize():
    normalize = albu.Normalize()

    def process(a):
        r = normalize(image=a)
        return r['image']

    return process


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.Cutout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.IAASharpen,
        'jpeg': albu.JpegCompression,
        'gray': albu.ToGray,
        'channel_shuffle': albu.ChannelShuffle,
        'grid_distortion': albu.GridDistortion,
    }
    return d[name]


def get_corrupt_function(config: Optional[List[dict]]):
    if config is None:
        return
    augs = []
    for aug_params in config:
        name = aug_params.pop('name')
        cls = _resolve_aug_fn(name)
        prob = aug_params.pop('prob') if 'prob' in aug_params else .5
        augs.append(cls(p=prob, **aug_params))

    augs = albu.OneOf(augs)

    def process(x):
        return augs(image=x)['image']

    return process
