import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image


def read_image(imgpath):
    """Read image and return numpy matrix. Possible image types: .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg.
    :param imgpath: path to read image from
    :return: numpy image
    """
    if imgpath.endswith('nii') or imgpath.endswith('nii.gz'):
        image_file = nib.load(imgpath)
        image_file_np = np.array(image_file.get_fdata())
    elif imgpath.endswith('mha') or imgpath.endswith('mhd') or imgpath.endswith('raw'):
        image_file = sitk.ReadImage(imgpath)
        image_file_np = sitk.GetArrayFromImage(image_file)
    else:
        try:
            image_file = Image.open(imgpath)
            image_file_np = np.array(image_file)
        except:
            print(
                'Image Type not recognized! Currently supporting .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg')
            return None
    return image_file_np


def save_image(img_np, imgpath, affine=np.eye(4)):
    """Save numpy image matrix to file. Possible image types: .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg.
    :param img_np: numpy image
    :param imgpath: path to save image to
    :param affine: if nii file type is chose, affine matrix (4x4) can be defined"""
    if imgpath.endswith('nii') or imgpath.endswith('nii.gz'):
        nib.save(nib.Nifti1Image(img_np, affine), imgpath)
    elif imgpath.endswith('mha') or imgpath.endswith('mhd') or imgpath.endswith('raw'):
        sitk.WriteImage(sitk.GetImageFromArray(img_np, isVector=False), imgpath)
    else:
        try:
            image_file = Image.fromarray(img_np)
            image_file.save(imgpath)
        except:
            print(
                'Image Type not recognized! Currently supporting .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg')
            return None


def normalize_image(img, min_value=0, max_value=255, value_type='uint8'):
    """Normalize image values in a certain range.
    :param img: input numpy image
    :param min_value: minimum gray value
    :param max_value: maximum gray value
    :param value_type: type of normalized img
    :return: nomralized image
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img * (max_value - min_value) + min_value
    return img.astype(value_type)
