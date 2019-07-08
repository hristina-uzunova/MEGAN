import os, sys
import json
import argparse
import numpy as np
from scipy.ndimage import sobel, generic_gradient_magnitude
import matlab.engine
import nibabel as nib

from MEGAN.image_utils import read_image, save_image, normalize_image

"""Script for preprocessing the LPBA images. Extracting edges, cropping images, saving file lists. """

parser = argparse.ArgumentParser(description='Preprocess LPBA data')
parser.add_argument('--in_img_dir', type=str, default='../Data/LPBA40', help='Image dir of the LPBA images')
parser.add_argument('--out_img_dir', type=str, default='../Data/LPBA3D', help='Output dir of preprocessed images and sketches')
parser.add_argument('--out_list_dir', type=str, default='../Data/SKETCH2BRATST23D',help='Output dir sketches list (test)')
opt = parser.parse_args()


### python MEGAN/preprocess_LPBA3D.py --in_img_dir /share/data_rosita1/sschultz/LPBA40withArtificialLesions2D/originalData2DAnd3D/LPBA40/ --out_img_dir /share/data_tiffy2/uzunova/GAN/Data/LPBA3DTmp/ --out_list_dir /share/data_tiffy2/uzunova/GAN/Data/SKETCH2BRATST23D/A/testtmp


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


if __name__ == "__main__":
    LPBA_img_dir = opt.in_img_dir
    out_dir = opt.out_img_dir
    out_list_dir = opt.out_list_dir

    if not os.path.exists(out_list_dir):
        os.makedirs(out_list_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n_patients = 40
    out_list_dir=os.path.join(out_list_dir, 'A/test')
    out_sketch_list = open(out_list_dir + '/data_list.txt', 'w')

    for pat in range(n_patients):
        image_path = os.path.join(LPBA_img_dir,
                                  '{:02d}'.format(pat) + '/delin.brain.nii')  # what is your dir structure?
        if not os.path.exists(image_path):
            continue
        print("Processing patient nr.: " + str(pat))
        nii_file = nib.load(image_path)
        image_np = np.array(nii_file.get_fdata())
        # crop using a bounding box
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

        # save an info json file per image
        info_path = out_dir + 'shape_info_' + str(pat) + '.json'
        info_json = {}
        info_json['pat'] = pat
        info_json['imgSizeWhole'] = image_np.shape
        info_json['imgSizeCropped'] = image_np[x_min:x_max, y_min:y_max, z_min:z_max].shape
        info_json['affine'] = nii_file.get_affine().tolist()
        info_json['coords'] = []
        info_json['coords'].append({'x_min': int(x_min), 'y_min': int(y_min), 'z_min': int(z_min)})
        with open(info_path, 'w') as outfile:
            json.dump(info_json, outfile)

        image_np = image_np[x_min:x_max, y_min:y_max, z_min:z_max]

        sketch_path = os.path.join(out_dir, 'sketch_pat_' + str(pat) + '.nii')
        if os.path.exists(sketch_path):
            sketch_img = read_image(sketch_path)
        else:
            sketch_img = generate_sketch(image_np)
        save_image(sketch_img.astype('uint8'), sketch_path)
        img_path = os.path.join(out_dir, 'pat_' + str(pat) + '.nii')
        save_image(normalize_image(image_np).astype('uint8'), img_path)
        out_sketch_list.write(sketch_path + '\n')
