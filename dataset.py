import os, sys
import numpy as np
import math
import scipy.ndimage as ndimage
import numpy.random as random

import torch
from torch.utils.data import Dataset

from MEGAN.image_utils import read_image


class MedicalImageDataset3D(Dataset):
    """This class creates a datset of 3D medical images.
       The dataset is build as follows: (sketch, grayValueImg)
       We use this to learn the whole low resolution images out of their edges.
     """

    def __init__(self, root, mode='train', listname='data_list.txt', new_size_imgs=None,
                 new_size_edges=None, augmentation=0):

        """Initialize the MedicalImageDataset3D class. We assume following input structure: root/A(B)/mode/data_list.txt
        Here we translate from domain A (sketches) to domain B (gray value images)
        Parameters:
            root -- root folder
            mode -- test oder train mode?
            listname --name of the list containing file names
            new_size_imgs -- if it is wished to resample the input images, their new size can be defined
            new_size_edges -- if it is wished to resample the sketch images, tehir new size can be defined
            augmentation -- [0,1] a percentage of data to be augmented
            """
        self.files_A = open(os.path.join(root, 'A/%s' % mode + '/' + listname)).readlines()
        self.files_B = open(os.path.join(root, 'B/%s' % mode + '/' + listname)).readlines()

        self.new_size_imgs = new_size_imgs
        self.new_size_edges = new_size_edges
        self.augmentation = augmentation

    def sample_rand_values(self, upper_limit=None, avg=0):
        """Samples random values for an affine transformation. The parameters are sampled in the range [avg-upper_limit, avg+upper_limit]
        :param upper_limit: upper and lowe limit of the sampling range
        :param avg: mean value
        :return: a random value within a range
        """
        random.seed(None)
        rand_nr = random.random()
        if not upper_limit is None:
            lower_limit = avg - upper_limit
            upper_limit = avg + upper_limit
            value = (upper_limit - lower_limit) * rand_nr + lower_limit
        else:
            value = rand_nr
        return value

    def create_random_affine_matrix(self, rot, transl, scale):
        """Creates and affine transformation matrix with random transformations for augmentation.
        :param rot: rotation degrees
        :param transl: translation percentage
        :param scale: scaling factor
        :return: an affine matrix
        """
        (rot_x, rot_y, rot_z) = rot
        (transl_x, transl_y, transl_z) = transl
        (scale_x, scale_y, scale_z) = scale
        rand_rot_x = self.sample_rand_values(rot_x) / (180. * np.pi)
        rand_rot_y = self.sample_rand_values(rot_y) / (180. * np.pi)
        rand_rot_z = self.sample_rand_values(rot_z) / (180. * np.pi)
        rand_transl_x = self.sample_rand_values(transl_x)
        rand_transl_y = self.sample_rand_values(transl_y)
        rand_transl_z = self.sample_rand_values(transl_z)
        rand_scale_x = self.sample_rand_values(scale_x, 1)
        rand_scale_y = self.sample_rand_values(scale_y, 1)
        rand_scale_z = self.sample_rand_values(scale_z, 1)

        rot_x = np.array([[1, 0, 0, 0],
                          [0, math.cos(rand_rot_x), -1 * math.sin(rand_rot_x), 0],
                          [0, math.sin(rand_rot_x), math.cos(rand_rot_x), 0],
                          [0, 0, 0, 1]])

        rot_y = np.array([[math.cos(rand_rot_y), 0, math.sin(rand_rot_y), 0],
                          [0, 1, 0, 0],
                          [-1 * math.sin(rand_rot_y), 0, math.cos(rand_rot_y), 0],
                          [0, 0, 0, 1]])

        rot_z = np.array([[math.cos(rand_rot_z), -1 * math.sin(rand_rot_z), 0, 0],
                          [math.sin(rand_rot_z), math.cos(rand_rot_z), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        transl_mat = np.array(
            [[1, 0, 0, rand_transl_x], [0, 1, 0, rand_transl_y], [0, 0, 1, rand_transl_z], [0, 0, 0, 1]])
        scale_mat = np.array([[rand_scale_x, 0, 0, 0], [0, rand_scale_y, 0, 0], [0, 0, rand_scale_z, 0], [0, 0, 0, 1]])
        return np.matmul(np.matmul(np.matmul(np.matmul(rot_x, rot_y), rot_z), transl_mat), scale_mat)

    def __getitem__(self, index):
        """Overwrite __getitem__. Returns a dataset item at an index position.
        :return: a tuple of (item_A(sketch), item_B(image))"""
        item_A = np.array(read_image(self.files_A[index % len(self.files_A)].rstrip()))
        item_B = np.array(read_image(self.files_B[index % len(self.files_B)].rstrip()))
        random.seed(None)

        prob = np.random.random_sample()
        if self.augmentation > 0 and prob < self.augmentation:
            transf_matrix = self.create_random_affine_matrix((10, 10, 10), (3, 3, 3), (0.1, 0.1, 0.1))
            item_A = ndimage.affine_transform(item_A, transf_matrix)
            item_B = ndimage.affine_transform(item_B, transf_matrix)
        item_A = torch.from_numpy(item_A).unsqueeze_(0)
        item_B = torch.from_numpy(item_B).unsqueeze_(0)
        if not self.new_size_edges is None:
            item_A = torch.nn.functional.interpolate(item_A.unsqueeze_(0), size=self.new_size_edges, mode='trilinear',
                                                     align_corners=True).float()
        if not self.new_size_imgs is None:
            item_B = torch.nn.functional.interpolate(item_B.unsqueeze_(0), size=self.new_size_imgs, mode='trilinear',
                                                     align_corners=True).float()
        item_A = item_A / 127.5 - 1
        item_B = item_B / 127.5 - 1
        return {'A': item_A.squeeze(0), 'B': item_B.squeeze(0)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
