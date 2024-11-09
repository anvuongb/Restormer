## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import cv2
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        super(TestDataset, self).__init__()
        self.img_paths = img_paths

    def __getitem__(self, index):
        with open(self.img_paths[index],'rb') as f:
            img_bytes = f.read()

        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        return img_lq

    def __len__(self):
        return len(self.img_paths)

batch_size = 4
noise_level = 100
weights_path = f"../experiments/RealDenoising_Restormer_t{noise_level}/models/net_g_latest.pth"
config_file = f"./Options/RealDenoising_Restormer_t{noise_level}.yml"
opt = yaml.load(open(config_file, mode='r'), Loader=Loader)

input_dir = f"/media/anvuong/Shared/datasets/celeba_prepared/img_celeba_test_noisy_t{noise_level}"
img_paths = os.listdir(input_dir)
img_paths = [os.path.join(input_dir, f"{idx}.png") for idx in range(2096)] # there are 2096 images in test set, in that order

output_dir = f"./results/celeba/t{noise_level}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load dataset
dataset = TestDataset(img_paths)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
s = opt['network_g'].pop('type')

##########################
model_restoration = Restormer(**opt['network_g'])

checkpoint = torch.load(weights_path)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights_path)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

idx = 0
with torch.no_grad():
    for noisy_batch in tqdm(dataloader) :
        restored_patch = model_restoration(noisy_batch)
        #for img in noisy_batch:
        #    torchvision.utils.save_image(img, os.path.join(output_dir, f"{idx}_noisy.png"))
        #    idx += 1
        for img in restored_patch:
            torchvision.utils.save_image(img, os.path.join(output_dir, f"{idx}.png"))
            idx += 1
