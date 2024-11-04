import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx


# src = 'Datasets/Downloads/SIDD'
split = "test"
tar = f'/home/anvuong/datasets/celeba_prepared/mpr/{split}'

lr_tar = os.path.join(tar, 'target_crops')
hr_tar_t100 = os.path.join(tar, 'input_crops_t100')
hr_tar_t200 = os.path.join(tar, 'input_crops_t200')
hr_tar_t300 = os.path.join(tar, 'input_crops_t300')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar_t100, exist_ok=True)
os.makedirs(hr_tar_t200, exist_ok=True)
os.makedirs(hr_tar_t300, exist_ok=True)

# files = natsorted(glob(os.path.join(src, '*', '*.PNG')))

# lr_files, hr_files = [], []
# for file_ in files:
#     filename = os.path.split(file_)[-1]
#     if 'GT' in filename:
#         hr_files.append(file_)
#     if 'NOISY' in filename:
#         lr_files.append(file_)

clean_path = f"/home/anvuong/datasets/celeba_prepared/img_celeba_{split}_clean"
noise_path_t100 = f"/home/anvuong/datasets/celeba_prepared/img_celeba_{split}_noisy_t100"
noise_path_t200 = f"/home/anvuong/datasets/celeba_prepared/img_celeba_{split}_noisy_t200"
noise_path_t300 = f"/home/anvuong/datasets/celeba_prepared/img_celeba_{split}_noisy_t300"
if split == "train":
    lr_files = [os.path.join(clean_path, str(i) + ".png") for i in range(100000)]
    hr_files_t100 = [os.path.join(noise_path_t100, str(i) + ".png") for i in range(100000)]
    hr_files_t200 = [os.path.join(noise_path_t200, str(i) + ".png") for i in range(100000)]
    hr_files_t300 = [os.path.join(noise_path_t300, str(i) + ".png") for i in range(100000)]
else:
    lr_files = [os.path.join(clean_path, str(i) + ".png") for i in range(2095)]
    hr_files_t100 = [os.path.join(noise_path_t100, str(i) + ".png") for i in range(2095)]
    hr_files_t200 = [os.path.join(noise_path_t200, str(i) + ".png") for i in range(2095)]
    hr_files_t300 = [os.path.join(noise_path_t300, str(i) + ".png") for i in range(2095)]

files = [(i, j, k, z) for i, j, k, z in zip(lr_files, hr_files_t100, hr_files_t200, hr_files_t300)]

patch_size = 128
overlap = 32
p_max = 0

def save_files(file_):
    lr_file, hr_file_t100, hr_file_t200, hr_file_t300 = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img_t100 = cv2.imread(hr_file_t100)
    hr_img_t200 = cv2.imread(hr_file_t200)
    hr_img_t300 = cv2.imread(hr_file_t300)
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch_t100 = hr_img_t100[i:i+patch_size, j:j+patch_size,:]
                hr_patch_t200 = hr_img_t200[i:i+patch_size, j:j+patch_size,:]
                hr_patch_t300 = hr_img_t300[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.png')
                hr_savename_t100 = os.path.join(hr_tar_t100, filename + '-' + str(num_patch) + '.png')
                hr_savename_t200 = os.path.join(hr_tar_t200, filename + '-' + str(num_patch) + '.png')
                hr_savename_t300 = os.path.join(hr_tar_t300, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename_t100, hr_patch_t100)
                cv2.imwrite(hr_savename_t200, hr_patch_t200)
                cv2.imwrite(hr_savename_t300, hr_patch_t300)

    else:
        lr_savename = os.path.join(lr_tar, filename + '.png')
        hr_savename_t100 = os.path.join(hr_tar_t100, filename + '.png')
        hr_savename_t200 = os.path.join(hr_tar_t200, filename + '.png')
        hr_savename_t300 = os.path.join(hr_tar_t300, filename + '.png')
        
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename_t100, hr_img_t100)
        cv2.imwrite(hr_savename_t200, hr_img_t200)
        cv2.imwrite(hr_savename_t300, hr_img_t300)

from joblib import Parallel, delayed
import multiprocessing
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))
