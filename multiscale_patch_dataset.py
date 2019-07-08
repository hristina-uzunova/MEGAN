import sys

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

import math
import numpy as np
import scipy.ndimage.filters as filters
import multiprocessing as mp
import threading

import torch
from torch.utils.data import Dataset

from MEGAN.coords3D_utils import sample_coords, nib_load, crop_img

np.random.seed(2017)


class MedicalImagePatches3D(Dataset):
    """A class to create a dataset out of image patches. We always assume a patch pairs of a low-resolution and
    high-resolution patches, which we extract from a LR or HR images respectively. The images are observed in full
    resolution, thus the LR images is smaller. The LR patches are then rescaled to the size of the HR patches.
    We use this for learning generation of high resolution patches out of low resolution patches and their sketsches
    (edge images)"""

    def __init__(self, list_file_A, list_file_B, patch_shape, LR_size, HR_size, sample_size=20):
        """ Initialize the class
        :param list_file_A: path to a list of filenames of domain A
        :param list_file_B: path to a list fo filenames of domain B
        :param patch_shape: a tuple of the shape of patches to be extracted
        :param LR_size: a tuple of the size of the low resolution image
        :param HR_size: a tuple of the size of the high resolution image
        :param sample_size: number of patches per image
        """
        with open(list_file_A) as f:
            names_A = f.read().splitlines()
        with open(list_file_B) as f:
            names_B = f.read().splitlines()

        self.names_A = names_A
        self.names_B = names_B
        self.sample_size = sample_size
        self.patch_shape = patch_shape
        self.LR_size = LR_size
        self.HR_size = HR_size
        self.C = 1  # channels



    def __call__(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        """:return: HRP -- high resolution patch
                    LRP -- low resolution patch
                    HREP -- high resolution edge patch
                    LR_img_small -- full resolution low-resolution image (small)
                    LRP_aug -- augmented low-resolution patch
                    coords_small -- coordinates of the patches in the HR image (large receptive field)
                    """
        sketches_path = self.names_A[index]
        images_path = self.names_B[index]

        img = np.array(nib_load(images_path))
        img = img / 127.5 - 1
        img = np.expand_dims(img, 0)
        # rescale image to low resolution
        LR_img = torch.nn.functional.interpolate(torch.from_numpy(img).unsqueeze_(0), self.LR_size,
                                                 mode='trilinear', align_corners=True).float()
        LR_img_small = LR_img.squeeze_(0).numpy()

        sketch_file = sketches_path
        sketch = np.array([nib_load(sketch_file)])
        sketch = sketch / 127.5 - 1
        # make sure sketches and images are the size of high resolution (can be smaller than native resolution)
        sketch = torch.nn.functional.interpolate(torch.from_numpy(sketch).unsqueeze_(0), self.HR_size,
                                                 mode='trilinear', align_corners=True).squeeze_(0).float().numpy()
        img = torch.nn.functional.interpolate(torch.from_numpy(img).unsqueeze_(0), self.HR_size, mode='trilinear',
                                              align_corners=True).squeeze_(0).float().numpy()
        mask = (img > -1).astype('int32')[0, :, :, :]

        # coordinates of the patch in the high resolution images (smaller receptive field)
        coords_small = sample_coords(self.sample_size, self.patch_shape, mask)

        # coordinates of the patch in the low resolution images (larges receptive field, but same patch size)
        coords_big = np.zeros_like(coords_small)
        scale_factor = self.HR_size[0] / self.LR_size[0]
        new_patch_size = self.patch_shape[0] / scale_factor
        coords_big[:, :, 0] = np.floor(coords_small[:, :, 0] / scale_factor)
        coords_big[:, :, 1] = np.floor(coords_small[:, :, 0] / scale_factor) + self.patch_shape - 1

        HRP = crop_img(coords_small, img, self.patch_shape)
        LR_img_padded = np.pad(LR_img_small[0, :, :, :], int(round((self.patch_shape[0] - new_patch_size) / 2.)),
                               'constant', constant_values=-1)
        LRP = crop_img(coords_big, np.expand_dims(LR_img_padded, 0), self.patch_shape)
        HREP = crop_img(coords_small, sketch, self.patch_shape)
        # augment low resolution images by blurring them with a certain probability
        prob = np.random.random_sample()
        if prob < 0.0:
            LRP_aug = filters.gaussian_filter(LRP, 3)
        else:
            LRP_aug = LRP
        return HRP, LRP, HREP, LR_img_small, LRP_aug, coords_small

    def __len__(self):
        return len(self.names_A)


def default_collate_fn(batch):
    return [torch.cat([torch.from_numpy(t) for t in v]) for v in zip(*batch)]


class PEDataLoader(object):
    """
    A multiprocess-dataloader that parallels over elements as suppose to
    over batches (the torch built-in one)
    Input dataset must be callable with index argument: dataset(index)
    https://github.com/thuyen/nnet/blob/master/pedataloader.py
    """

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=None, pin_memory=False, num_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.collate_fn = default_collate_fn
        self.pin_memory_fn = \
            torch.utils.data.dataloader.pin_memory_batch if pin_memory else \
                lambda x: x

        self.num_samples = len(dataset)
        self.num_batches = num_batches or \
                           int(math.ceil(self.num_samples / float(self.batch_size)))

        self.pool = mp.Pool(num_workers)
        self.buffer = queue.Queue(maxsize=1)
        self.start()

    def generate_batches(self):
        if self.shuffle:
            indices = torch.LongTensor(self.batch_size)
            for b in range(self.num_batches):
                indices.random_(0, self.num_samples - 1)
                batch = self.pool.map(self.dataset, indices)
                batch = self.collate_fn(batch)
                batch = self.pin_memory_fn(batch)
                yield batch
        else:
            self.indices = torch.LongTensor(range(self.num_samples))
            for b in range(self.num_batches):
                start_index = b * self.batch_size
                end_index = (b + 1) * self.batch_size if b < self.num_batches - 1 \
                    else self.num_samples
                indices = self.indices[start_index:end_index]
                batch = self.pool.map(self.dataset, indices)
                batch = self.collate_fn(batch)
                batch = self.pin_memory_fn(batch)
                yield batch

    def start(self):
        def _thread():
            for b in self.generate_batches():
                self.buffer.put(b, block=True)
            self.buffer.put(None)

        thread = threading.Thread(target=_thread)
        thread.daemon = True
        thread.start()

    def __next__(self):
        batch = self.buffer.get()
        if batch is None:
            self.start()
            raise StopIteration
        return batch

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches
