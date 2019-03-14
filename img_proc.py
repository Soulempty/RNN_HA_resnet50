# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            self.PIL2Numpy = True
        for a in self.augmentations:
            img = a(img)
        if self.PIL2Numpy:
            img = np.array(img)
        return img


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):   
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
class Self_Scale(object):
    """Rescales the input PIL.Image to the given equal '(size,size)'.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

class Scale(object):
    """Rescales the input PIL.Image to the given equal '(size,size)'.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR)
                
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR)

class Invert_Normalize(object):
    def __init__(self, mean = None):
        self.mean = mean
        self.s = 1. / 255
    def __call__(self, tensor):
        for t, m in zip(tensor, self.mean):
            t.div(self.s).sub_(m)
        return tensor
