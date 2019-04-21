import cv2
import torch
import random
import numpy as np

cv2.setNumThreads(0)


def image_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def gauss_noise(image, sigma_sq):
    image = image.astype(np.uint32)
    h, w, c = image.shape
    gauss = np.random.normal(0, sigma_sq, (h, w))
    gauss = gauss.reshape(h, w)
    image = image + np.stack([gauss for _ in range(c)], axis=2)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox=None):
        if bbox is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, bbox = t(image, bbox)
            return image, bbox


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, bbox=None):
        if bbox is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, bbox = self.transform(image, bbox)
            return image, bbox


class OneOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox=None):
        transform = np.random.choice(self.transforms)
        if bbox is None:
            image = transform(image)
            return image
        else:
            image, bbox = transform(image, bbox)
            return image, bbox


class Flip:
    def __init__(self, flip_code):
        assert flip_code == 0 or flip_code == 1
        self.flip_code = flip_code

    def __call__(self, image):
        image = cv2.flip(image, self.flip_code)
        return image


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class VerticalFlip(Flip):
    def __init__(self):
        super().__init__(0)


class GaussNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, image):
        if self.sigma_sq > 0.0:
            image = gauss_noise(image,
                                np.random.uniform(0, self.sigma_sq))
        return image


class RandomGaussianBlur:
    '''Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    '''
    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image):
        kernel_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernel_size, self.sigma_x)
        return blured_image


class ImageToTensor:
    def __call__(self, image):
        image = np.stack([image, image, image], axis=0)
        image = image.astype(np.float32) / 100
        image = torch.from_numpy(image)
        return image


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = random.randint(0, signal.shape[1] - self.size)
        return signal[:, start: start + self.size]


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = (signal.shape[1] - self.size) // 2
        return signal[:, start: start + self.size]


def get_transforms(train, size):
    if train:
        transforms = Compose([
            RandomCrop(size),
            ImageToTensor()
        ])
    else:
        transforms = Compose([
            CenterCrop(size),
            ImageToTensor()
        ])
    return transforms
