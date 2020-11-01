import random
import numpy as np
from PIL import Image

class RLAugmentation:
    def __init__(self, transform, *args, p=0.5):
        self.p = p
        self.transform = transform
        self.args = args
    def __call__(self, img_batch):
        "Take a batch of images and apply the same transform across framestacks"
        img_batch = img_batch.copy()
        for i, im in enumerate(img_batch):
            if random.random() < self.p:
                img_batch[i] = self.transform(img_batch[i], *[
                    (random.uniform(*a) if isinstance(a, tuple) else random.choice(a)) for a in self.args])
        return img_batch

def rotate_image(img, angle):
    if img.ndim == 4:
#         img = img.copy()
        for i in range(len(img)):
            img[i] = rotate_image(img[i], angle)
        return img
    elif img.ndim == 3:
        return np.asarray(Image.fromarray(img).rotate(angle, Image.BILINEAR))
    else:
        raise ValueError("Ooops")

def translate(img, x, y):
    x,y = round(x), round(y)
    img = np.roll(img, x, axis=-2)
    img = np.roll(img, y, axis=-3)
    return img

def crop(img, crop_size, top_left=None):
    crop_size = round(crop_size)
    if img.ndim == 4:
        x,y = img.shape[-3:-1]
        x1 = random.randint(0,x-crop_size)
        y1 = random.randint(0,y-crop_size)
        for i in range(len(img)):
            img[i] = crop(img[i], crop_size, top_left=(x1,y1))
        return img
    elif img.ndim == 3:
        img = Image.fromarray(img)
        x,y = img.size
        if top_left is None:
            x1 = random.randint(0,x-crop_size)
            y1 = random.randint(0,y-crop_size)
        else: x1,y1 = top_left
        coords = (x1,y1,x1+crop_size, y1+crop_size)
        img = img.crop(coords).resize((x,y), Image.BILINEAR)
        return np.asarray(img)

def pad_crop(img, crop_size, top_left=None):
    crop_size = round(crop_size)
    if img.ndim == 4:
        x,y = img.shape[-3:-1]
        x1 = random.randint(0,x-crop_size)
        y1 = random.randint(0,y-crop_size)
        for i in range(len(img)):
            img[i] = crop(img[i], crop_size, top_left=(x1,y1))
        return img
    elif img.ndim == 3:
        img = np.pad(img, ((0,), (10,), (10,)), mode='reflect')
        img = Image.fromarray(img)
        x,y = img.size
        if top_left is None:
            x1 = random.randint(0,x-crop_size)
            y1 = random.randint(0,y-crop_size)
        else: x1,y1 = top_left
        coords = (x1,y1,x1+crop_size, y1+crop_size)
        img = img.crop(coords) #.resize((x,y), Image.BILINEAR)
        return np.asarray(img)

def flip(img, lr, ud):
    if ud:
        img = img[..., ::-1, :]
    if lr:
        img = img[..., ::-1, :, :]
    return img


def get_tfms(p=0.5):
    Rotate = RLAugmentation(rotate_image, (-20,20), p=p)
    Translate = RLAugmentation(translate, (-8, 8), (-8, 8), p=p)
    Crop = RLAugmentation(crop, (48, 64), p=p)
    Flip = RLAugmentation(flip, [True,False], [True,False], p=p)
    PadCrop = RLAugmentation(pad_crop, (64, 64), p=p)

    def transform_obs(obs):
        # return Flip(Rotate(Crop(Translate(obs))))
        # return Flip(obs)
        # return Rotate(obs)
        # return Crop(obs)
        # return Translate(obs)
        # return PadCrop(obs)
        return Flip(Translate(obs))

    return transform_obs
