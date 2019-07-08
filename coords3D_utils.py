import numpy as np
import nibabel as nib


"""Functions to deal with 3D patch coordinates."""
xrange=range

def sample_coords(sample_size, patch_shape, weight_map):
    ndim = len(patch_shape)
    dist2center = np.zeros((ndim, 2) , dtype='int32') # from patch boundaries
    for dim, shape in enumerate(patch_shape) :
        dist2center[dim] = [shape/2 - 1, shape/2] if shape % 2 == 0 \
                else [shape//2, shape//2]

    sx, sy, sz = dist2center[:, 0]                    # left-most boundary
    ex, ey, ez = weight_map.shape - dist2center[:, 1] # right-most boundary

    maps = np.zeros(weight_map.shape, dtype="float32")
    maps[sx:ex, sy:ey, sz:ez] = 1
    maps *= weight_map
    maps /= 1.0 * np.sum(maps)
    maps = maps.flatten()

    sampled_indices = np.random.choice(
            maps.size,
            size=sample_size,
            replace=True,
            p=maps)

    sampled_coords = np.asarray(np.unravel_index(sampled_indices, weight_map.shape))


    sampled_coords = sampled_coords.T
    slice_sampled_coords = np.zeros(sampled_coords.shape + (2, ), dtype="int32")
    slice_sampled_coords[:,:,0] = sampled_coords - dist2center[:, 0]
    slice_sampled_coords[:,:,1] = sampled_coords + dist2center[:, 1]

    return slice_sampled_coords

def get_all_coords(stride, patch_shape, image_shape, batch_size, mask=None):

    slice_coords = []

    zlo_next=0; z_done = False;
    while not z_done :
        zhi = min(zlo_next + patch_shape[2], image_shape[2]) # Excluding
        zlo = zhi - patch_shape[2]
        zlo_next = zlo_next + stride[2]
        z_done = False if zhi < image_shape[2] else True

        clo_next=0; c_done = False;
        while not c_done :
            chi = min(clo_next + patch_shape[1], image_shape[1]) # Excluding
            clo = chi - patch_shape[1]
            clo_next = clo_next + stride[1]
            c_done = False if chi < image_shape[1] else True

            rlo_next=0; r_done = False;
            while not r_done :
                rhi = min(rlo_next + patch_shape[0], image_shape[0]) # Excluding
                rlo = rhi - patch_shape[0]
                rlo_next = rlo_next + stride[0]
                r_done = False if rhi < image_shape[0] else True

                if isinstance(mask, np.ndarray):
                    # All of it is out of the brain so skip it.
                    if not np.any(mask[rlo:rhi, clo:chi, zlo:zhi]):
                        continue

                slice_coords.append([[rlo, rhi-1], [clo, chi-1], [zlo, zhi-1]])

    # Total num needs to be divisible by 'batch_size'.
    num = len(slice_coords)
    for _ in xrange(batch_size - num%batch_size) :
        slice_coords.append(slice_coords[-1])

    slice_coords = np.array(slice_coords)

    return slice_coords

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

def get_sub_patch_shape(patch_shape, receptive_field, factor) :
    patch_shape = np.array(patch_shape)
    receptive_field = np.array(receptive_field)
    factor = np.array(factor)
    patch_center = patch_shape - receptive_field + 1
    sub_patch_center = np.ceil(patch_center*1.0/factor).astype('int')
    sub_patch_size = receptive_field + sub_patch_center - 1
    return sub_patch_size

def get_receptive_field(kernel_sizes) :
    if not kernel_sizes : #list is []
        return 0
    ndim = len(kernel_sizes[0])
    receptive_field = [1]*ndim
    for dim in xrange(ndim) :
        for l in xrange(len(kernel_sizes)) :
            receptive_field[dim] += kernel_sizes[l][dim] - 1
    return np.array(receptive_field)


def get_offset(factor, receptive_field):
    x1 = ((factor - 1)/2)*receptive_field
    x2 = ((factor - 2)/2)*receptive_field + receptive_field/2
    m = factor % 2
    x = x1*m + x2*(1-m)

    d1 = factor/2
    d2 = factor/2 - 1
    d = d1*m + d2*(1-m)
    return d-x


def coord_to_slice(coord):
    return coord[:, 0], coord[:, 1] + 1


def crop_img(coords, image, patch_shape, C=1):
    """Crop image given patch coordinates.
    :param coords: coordinates to the patches containg the starting and ending slice coordinates of the patches.
    _:param image: image to be cropped
    :param patch_shape: shape of patches
    :param C: number of channels
    :return: a batch of image patches
    """
    N = coords.shape[0]
    patch_img = np.zeros((N, C) + tuple(patch_shape), dtype='float32')
    for n, coord in enumerate(coords):
        ss, ee = coord_to_slice(coord)
        patch_img[n] = image[:, ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2]]
    return patch_img
