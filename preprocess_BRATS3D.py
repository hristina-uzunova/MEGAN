import os, sys
from glob import glob
import argparse
import numpy as np
import random
from scipy.ndimage import sobel, generic_gradient_magnitude
import matlab.engine
import json
from MEGAN.image_utils import read_image, save_image, normalize_image

"""Script for preprocessig the BRATS training data. Extractzing edges, cropping images, svaing list files."""

parser = argparse.ArgumentParser(description='Preprocess BRATS data')
parser.add_argument('--in_img_dir', type=str, default='../Data/BRATS2015_Training',help='Image dir of the BRATS images')
parser.add_argument('--out_img_dir', type=str, default='../Data/BRATST2_3D',help='Output dir of preprocessed images and sketches')
parser.add_argument('--out_list_dir', type=str, default='../Data/SKETCH2BRATST23D',
                    help='Output dir of images and sketches lists (training and validation)')
parser.add_argument('--valid_pr', type=float, default=0.1, help='[0,1] percentage of images for validation')
opt = parser.parse_args()


###python MEGAN/preprocess_BRATS3D.py --in_img_dir /imagedata/BRATS2016/BRATS2015_Training --out_img_dir /share/data_tiffy2/uzunova/GAN/Data/BRATST1_3DTmp --out_list_dir /share/data_tiffy2/uzunova/GAN/Data/SKETCH2BRATST13D/


def generate_sketch(img, low_val=0.1, high_val=0.2):
    """Generate an image sketch, using a Canny filter with a lower threshold and a high threshold. This implementation
    uses the MATLAB implementation and thus a MATLAB engine, since no 3D python implementation is available. If you dont
    have MATLAB simply use the gradient image or a sobel filter (method available in numpy and scipy)
     :param img: input image
     :param low_val: lower threshold of Canny filter
     :param high_val: higher threshold of Canny filter
     :return: sketch of image (canny edges weighted by gradient magnitude)
     """
    norm_img = normalize_image(img, 0, 1, 'float32')

    # start MATLAB engine
    eng = matlab.engine.start_matlab()
    img_list = matlab.double(norm_img.tolist())

    # apply canny
    edges = eng.edge3(img_list, 'approxcanny', matlab.double([low_val, high_val]))
    # form MATLAB to numpy
    edges_np = edges._data
    edges_np = np.reshape(edges_np, (img.shape[2], img.shape[1], img.shape[0]))
    edges_np = np.transpose(edges_np, (2, 1, 0))
    # we want a magnitude weighted edge  image
    magnitudes = generic_gradient_magnitude(normalize_image(norm_img, 0, 1, 'float32'), sobel)
    norm_magnitudes = normalize_image(magnitudes, 0, 1, 'float32')
    return edges_np * norm_magnitudes * 255


random.seed(2017)
valid_pr = opt.valid_pr

in_dir = opt.in_img_dir
in_HGG_dir = os.path.join(in_dir, 'HGG')  # both: high grade
in_LGG_dir = os.path.join(in_dir, 'LGG')  # and low grade cases
out_dir = opt.out_img_dir
out_list_dir = opt.out_list_dir
sketch_list_dir = os.path.join(out_list_dir, 'A/train')
img_list_dir = os.path.join(out_list_dir, 'B/train')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(sketch_list_dir):
    os.makedirs(sketch_list_dir)
if not os.path.exists(img_list_dir):
    os.makedirs(img_list_dir)

out_img_list = open(img_list_dir + '/data_list.txt', 'w')  # training img list
out_sketch_list = open(sketch_list_dir + '/data_list.txt', 'w')  # training sketches list

out_img_list_valid = open(img_list_dir + '/data_list_valid.txt', 'w')  # validation img list
out_sketch_list_valid = open(sketch_list_dir + '/data_list_valid.txt', 'w')  # validation sketch list

img_nr = 0
pats = glob(in_HGG_dir + '/brats_*') + glob(in_LGG_dir + '/brats_*')

for pat in pats:
    img_nr += 1
    pat_dir = in_HGG_dir + str(pat)
    print("Processing patient nr.: " + str(img_nr))
    try:
        T2_dir = glob(pat + '/VSD.Brain.XX.O.MR_T2*')[0]
        mask_dir = glob(pat + '/VSD.Brain_*more*')[0]
        file_name_3D = glob(T2_dir + '/*.mha')[0]
        mask_name_3D = glob(mask_dir + '/*.mha')[0]
    except:
        continue

    image_np = np.flip(np.flip(np.transpose(read_image(file_name_3D), (2, 1, 0)), 0), 1)
    mask_np = np.flip(np.flip(np.transpose(read_image(mask_name_3D), (2, 1, 0)), 0), 1)
    bb_coords = np.where(image_np > 0)
    offset = 3
    x_min = np.min(bb_coords[0])
    x_min = max(x_min - offset, 0)
    x_max = np.max(bb_coords[0])
    x_max = min(x_max + offset, image_np.shape[0])
    y_min = np.min(bb_coords[1])
    y_min = max(y_min - offset, 0)
    y_max = np.max(bb_coords[1])
    y_max = min(y_max + offset, image_np.shape[1])
    z_min = np.min(bb_coords[2])
    z_min = max(z_min - offset, 0)
    z_max = np.max(bb_coords[2])
    z_max = min(z_max + offset, image_np.shape[2])

    info_path = os.path.join(out_dir, 'shape_info_' + str(img_nr) + '.json')
    info_json = {}
    info_json['pat'] = file_name_3D
    info_json['imgSizeWhole'] = image_np.shape
    info_json['imgSizeCropped'] = image_np[x_min:x_max, y_min:y_max, z_min:z_max].shape
    info_json['affine'] = None
    info_json['coords'] = []
    info_json['coords'].append({'x_min': int(x_min), 'y_min': int(y_min), 'z_min': int(z_min)})
    with open(info_path, 'w') as outfile:
        json.dump(info_json, outfile)
    # crop
    image_np = image_np[x_min:x_max, y_min:y_max, z_min:z_max]
    image_np = normalize_image(image_np)
    mask_np = mask_np[x_min:x_max, y_min:y_max, z_min:z_max]

    image_path = os.path.join(out_dir, 'pat_' + str(img_nr) + '.nii')
    mask_path = os.path.join(out_dir, 'pat_' + str(img_nr) + '_mask.nii')
    save_image(image_np, image_path)
    save_image(mask_np, mask_path)

    #### Sketch
    sketch_img = generate_sketch(image_np)
    # combine with tumor mask
    mask_np = np.maximum(np.zeros_like(mask_np), (64 * mask_np.astype('int32') - 1))
    sketch_img = np.maximum(sketch_img, mask_np)

    sketch_path = os.path.join(out_dir, 'sketch_pat_' + str(img_nr) + '.nii')
    save_image(sketch_img.astype('uint8'), sketch_path)

    # img for validation?
    if random.random() <= valid_pr:
        out_img_list_valid.write(image_path + '\n')
        out_sketch_list_valid.write(sketch_path + '\n')
    else:
        out_img_list.write(image_path + '\n')
        out_sketch_list.write(sketch_path + '\n')
