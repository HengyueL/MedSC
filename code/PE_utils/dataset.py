import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import cv2
import h5py
import numpy as np
import os
import pickle
import random
import torch

from scipy.ndimage.interpolation import rotate

import numpy as np
import random

from PIL import Image
from torch.utils.data import Dataset
import PE_utils as util

"""Constants related to the CT PE dataset."""

# Hounsfield Units for Air
AIR_HU_VAL = -1000.


# Statistics for Hounsfield Units




CONTRAST_HU_MIN = -100.     # Min value for loading contrast
CONTRAST_HU_MAX = 900.      # Max value for loading contrast
CONTRAST_HU_MEAN = 0.15897  # Mean voxel value after normalization and clipping
CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping
"""

CONTRAST_HU_MIN = -250#-250.     # Min value for loading contrast
CONTRAST_HU_MAX = 450      # Max value for loading contrast
CONTRAST_HU_MEAN = 0.15897 #0.3  # Mean voxel value after normalization and clipping
CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping

"""


W_CENTER_DEFAULT = 400.     # Default window center for viewing
W_WIDTH_DEFAULT = 1000.     # Default window width


class BaseCTDataset(Dataset):
    """Base dataset for CT studies."""

    def __init__(self, data_dir, img_format, is_training_set=True):
        """
        Args:
            data_dir: Root data directory.
            img_format: Format of input image files. Options are 'raw' (Hounsfield Units) or 'png'.
            is_training_set: If training, shuffle pairs and define len as max of len(src_paths) and len(tgt_paths).
            If not training, take pairs in order and define len as len(src_paths).
            appear with the same set of tgt images.
        """
        if img_format != 'png' and img_format != 'raw':
            raise ValueError('Unsupported image format: {}'.format(img_format))

        self.data_dir = data_dir
        self.img_format = img_format
        self.is_training_set = is_training_set
        self.pixel_dict = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def _get_img(self, img_path):
        """Load an image from `img_path`. Use format `self.img_format`."""
        return np.load(img_path) if self.img_format == 'raw' else Image.open(img_path).convert('L')

    @staticmethod
    def _hide_and_seek_transform(np_block, to_show_set, mean, grid_dim_x, grid_dim_y, hide_prob, hide_level):
        """ Replaces grid squares in the image with the pixel mean with probability `hide_prob`
            , excepting those that are passed in through the 'to_show_set'

        Args:
            np_block: 2 or 3D npy block of pixel values (based on self.hide_level)
            to_show_set: set object containing the grid squares that should remain unhidden (contain the labeled feature)

        Returns:
            2 or 3D npy block of pixel values
        """
        # Blank out each square with P(hide_probability)
        grid_size = grid_dim_x * grid_dim_y
        to_hide_list = np.argwhere(np.array([random.random() for i in range(grid_size)]) < hide_prob)
        to_hide = np.array([])
        if len(to_hide_list) > 1:
            to_hide = np.concatenate(to_hide_list)

        # If there are no forced inclusions and we chose to hide everything in the image
        # Show at least one square
        if len(to_show_set) == 0 and len(to_hide_list) == grid_size:
            to_show_set = set([random.randint(0, grid_size - 1)])

        to_hide_set = set(to_hide.ravel())
        to_hide_set = to_hide_set - to_show_set
        
        if hide_level == 'image':
            image_dim = np_block.shape[0]
        else:
            image_dim = np_block.shape[1]
            
        for square in to_hide_set:
            row = square // grid_dim_x
            col = square % grid_dim_y
            width = image_dim // grid_dim_y
            remainder = width * grid_dim_y % image_dim
            
            pad_horizontal = remainder if row == grid_dim_x - 1 else 0
            pad_vertical = remainder if col == grid_dim_y - 1 else 0 
            if hide_level == 'image':
                np_block[row*width:(row+1)*width + pad_horizontal, col*width:(col+1)*width + pad_vertical] = mean 
            else:
                for idx in range(np_block.shape[0]):
                    np_block[idx, row*width:(row+1)*width + pad_horizontal, col*width:(col+1)*width + pad_vertical] = mean 
        
        return np_block

    def _normalize_raw(self, pixels):
        """Normalize an ndarray of raw Hounsfield Units to [-1, 1].

        Clips the values to [min, max] and scales into [0, 1],
        then subtracts the mean pixel (min, max, mean are defined in constants.py).

        Args:
            pixels: NumPy ndarray to convert. Any shape.

        Returns:
            NumPy ndarray with normalized pixels in [-1, 1]. Same shape as input.
        """
            
        #print(pixels.min(), pixels.max())
        # TODO normalize to range 
        #pixels = np.interp(pixels, (pixels.min(), pixels.max()), (-3024, 3071))

        pixels = pixels.astype(np.float32)


        pixels = (pixels - self.pixel_dict['min_val']) / (self.pixel_dict['max_val'] - self.pixel_dict['min_val'])

        pixels = np.clip(pixels, 0., 1.) - self.pixel_dict['avg_val']

        return pixels



class CTPEDataset3d(BaseCTDataset):
    def __init__(self, data_dir, phase, is_training_set=True):
        """
        Args:
            args: Command line arguments.
            phase: one of 'train','val','test'
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTPEDataset3d, self).__init__(data_dir, 'raw', is_training_set=is_training_set)
        self.phase = phase
        self.resize_shape = (224, 224)
        self.is_test_mode = not is_training_set
        self.pe_types = ["central", "segmental"]

        # Augmentation
        self.crop_shape = [208,208]

        self.threshold_size = 0
        self.pixel_dict = {
            'min_val': CONTRAST_HU_MIN,
            'max_val': CONTRAST_HU_MAX,
            'avg_val': CONTRAST_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT
        }

        # Load info for the CTPE series in this dataset
        pkl_path = os.path.join(data_dir, 'series_list.pkl')
        with open(pkl_path, 'rb') as pkl_file:
            all_ctpes = pickle.load(pkl_file)

        self.ctpe_list = [ctpe for ctpe in all_ctpes if self._include_ctpe(ctpe)]
        self.positive_idxs = [i for i in range(len(self.ctpe_list)) if self.ctpe_list[i].is_positive]
        self.min_pe_slices = 4
        self.num_slices = 32
        self.abnormal_prob = None
        self.use_hem =  None

        # Map from windows to series indices, and from series indices to windows
        self.window_to_series_idx = []  # Maps window indices to series indices
        self.series_to_window_idx = []  # Maps series indices to base window index for that series
        window_start = 0
        for i, s in enumerate(self.ctpe_list):
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0)
            self.window_to_series_idx += num_windows * [i]
            self.series_to_window_idx.append(window_start)
            window_start += num_windows

        if self.use_hem:
            # Initialize a HardExampleMiner with IDs formatted like (series_idx, start_idx)
            example_ids = []
            for window_idx in range(len(self)):
                series_idx = self.window_to_series_idx[window_idx]
                series = self.ctpe_list[series_idx]
                if not series.is_positive:
                    # Only include negative examples in the HardExampleMiner
                    start_idx = (window_idx - self.series_to_window_idx[series_idx]) * self.num_slices
                    example_ids.append((series_idx, start_idx))
            self.hard_example_miner = util.HardExampleMiner(example_ids)

    def _include_ctpe(self, pe):
        """Predicate for whether to include a series in this dataset."""
        if pe.phase != self.phase and self.phase != 'all':
            return False
        
        if pe.is_positive and pe.type not in self.pe_types:
            return False

        return True
    def __len__(self):
        return len(self.window_to_series_idx)

    def __getitem__(self, idx):
        # Choose ctpe and window within ctpe
        ctpe_idx = self.window_to_series_idx[idx]
        ctpe = self.ctpe_list[ctpe_idx]

        if self.abnormal_prob is not None and random.random() < self.abnormal_prob:
            # Force aneurysm window with probability `abnormal_prob`.
            if not ctpe.is_positive:
                ctpe_idx = random.choice(self.positive_idxs)
                ctpe = self.ctpe_list[ctpe_idx]
            start_idx = self._get_abnormal_start_idx(ctpe, do_center=self.do_center_abnormality)
        elif self.use_hem:
            # Draw from distribution that weights hard negatives more heavily than easy examples
            ctpe_idx, start_idx = self.hard_example_miner.sample()
            ctpe = self.ctpe_list[ctpe_idx]
        else:
            # Get sequential windows through the whole series
            # TODO
            start_idx = (idx - self.series_to_window_idx[ctpe_idx]) * self.num_slices

        # if self.do_jitter:
        #     # Randomly jitter start offset by num_slices / 2
        #     start_idx += random.randint(-self.num_slices // 2, self.num_slices // 2)
        #     start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)

        volume = self._load_volume(ctpe, start_idx)
        volume = self._transform(volume)

        is_abnormal = torch.tensor([self._is_abnormal(ctpe, start_idx)], dtype=torch.float32)

        # Pass series info to combine window-level predictions
        target = {'is_abnormal': is_abnormal,
                  'study_num': ctpe.study_num,
                  'dset_path': str(ctpe.study_num),
                  'slice_idx': start_idx,
                  'series_idx': ctpe_idx}

        return volume, target

    def get_series_label(self, series_idx):
        """Get a floating point label for a series at given index."""
        return float(self.ctpe_list[series_idx].is_positive)

    def get_series(self, study_num):
        """Get a series with specified study number."""
        for ctpe in self.ctpe_list:
            if ctpe.study_num == study_num:
                return ctpe
        return None

    def update_hard_example_miner(self, example_ids, losses):
        """Update HardExampleMiner with set of example_ids and corresponding losses.

        This should be called at the end of every epoch.

        Args:
            example_ids: List of example IDs which were used during training.
            losses: List of losses for each example ID (must be parallel to example_ids).
        """
        example_ids = [(series_idx, start_idx) for series_idx, start_idx in example_ids
                       if series_idx not in self.positive_idxs]
        if self.use_hem:
            self.hard_example_miner.update_distribution(example_ids, losses)

    def _get_abnormal_start_idx(self, ctpe, do_center=True):
        """Get an abnormal start index for num_slices from a series.

        Args:
            ctpe: CTPE series to sample from.
            do_center: If true, center the window on the abnormality.

        Returns:
            Randomly sampled start index into series.
        """
        abnormal_bounds = (min(ctpe.pe_idxs), max(ctpe.pe_idxs))

        # Get actual slice number
        if do_center:
            # Take a window from center of abnormal region
            center_idx = sum(abnormal_bounds) // 2
            start_idx = max(0, center_idx - self.num_slices // 2)
        else:
            # Randomly sample num_slices from the abnormality (taking at least min_pe_slices).
            start_idx = random.randint(abnormal_bounds[0] - self.num_slices + self.min_pe_slices,
                                       abnormal_bounds[1] - self.min_pe_slices + 1)

        return start_idx

    def _load_volume(self, ctpe, start_idx):
        """Load num_slices slices from a CTPE series, starting at start_idx.

        Args:
            ctpe: The CTPE series to load slices from.
            start_idx: Index of first slice to load.

        Returns:
            volume: 3D NumPy arrays for the series volume.
        """
        if self.img_format == 'png':
            raise NotImplementedError('No support for PNGs in our HDF5 files.')

        with h5py.File(os.path.join(self.data_dir, 'data.hdf5'), 'r') as hdf5_fh:
            volume = hdf5_fh[str(ctpe.study_num)][start_idx:start_idx + self.num_slices]

        return volume

    def _is_abnormal(self, ctpe, start_idx):
        """Check whether a window from `ctpe` starting at start_idx includes an abnormality.

        Args:
            ctpe: CTPE object to check for any abnormality.

        Returns:
            True iff (1) ctpe contains an aneurysm and (2) abnormality is big enough.
        """
        if ctpe.is_positive:
            abnormal_slices = [i for i in ctpe.pe_idxs if start_idx <= i < start_idx + self.num_slices]
            is_abnormal = len(abnormal_slices) >= self.min_pe_slices
        else:
            is_abnormal = False

        return is_abnormal

    def _crop(self, volume, x1, y1, x2, y2):
        """Crop a 3D volume (before channel dimension has been added)."""
        volume = volume[:, y1: y2, x1: x2]

        return volume

    def _rescale(self, volume, interpolation=cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)

    def _pad(self, volume):
        """Pad a volume to make sure it has the expected number of slices.
        Pad the volume with slices of air.

        Args:
            volume: 3D NumPy array, where slices are along depth dimension (un-normalized raw HU).

        Returns:
            volume: 3D NumPy array padded/cropped to have the expected number of slices.
        """

        def add_padding(volume_, pad_value=AIR_HU_VAL):
            """Pad 3D volume with air on both ends to desired number of slices.
            Args:
                volume_: 3D NumPy ndarray, where slices are along depth dimension (un-normalized raw HU).
                pad_value: Constant value to use for padding.
            Returns:
                Padded volume with depth args.num_slices. Extra padding voxels have pad_value.
            """
            num_pad = self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode='constant', constant_values=pad_value)

            return volume_

        volume_num_slices = volume.shape[0]

        if volume_num_slices < self.num_slices:
            volume = add_padding(volume, pad_value=AIR_HU_VAL)
        elif volume_num_slices > self.num_slices:
            # Choose center slices
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]

        return volume

    def _transform(self, inputs):
        """Transform slices: resize, random crop, normalize, and convert to Torch Tensor.

        Args:
            inputs: 2D/3D NumPy array (un-normalized raw HU), shape (height, width).

        Returns:
            volume: Transformed volume, shape (num_channels, num_slices, height, width).
        """
        if self.img_format != 'raw':
            raise NotImplementedError('Unsupported img_format: {}'.format(self.img_format))

        # Pad or crop to expected number of slices
        inputs = self._pad(inputs)

        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation=cv2.INTER_AREA)

        if self.crop_shape is not None:
            row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])
            col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
            # Random crop during training, center crop during test inference
            row = random.randint(0, row_margin) if self.is_training_set else row_margin // 2
            col = random.randint(0, col_margin) if self.is_training_set else col_margin // 2
            inputs = self._crop(inputs, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])

        # if self.do_vflip and random.random() < 0.5:
        #     inputs = np.flip(inputs, axis=-2)

        # if self.do_hflip and random.random() < 0.5:
        #     inputs = np.flip(inputs, axis=-1)

        # if self.do_rotate:
        #     angle = random.randint(-15, 15)
        #     inputs = rotate(inputs, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL)

        # Normalize raw Hounsfield Units
        inputs = self._normalize_raw(inputs)

        inputs = np.expand_dims(inputs, axis=0)  # Add channel dimension
        inputs = torch.from_numpy(inputs)

        return inputs


class SortedSampler(Sampler):
    """SortedSampler is a custom batch sampler for the dataloader which selects
    batch indices under the condition that within a batch, indices must be in sorted
    order."""
    def __init__(self, batch_size, drop_last, data_source, shuffle):
        super(SortedSampler, self).__init__()
        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                batch.sort()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch.sort()
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class PaddedInputs(object):
    """Wrapper class for sending padded inputs to a model."""
    def __init__(self, inputs, length):
        self.inputs = inputs
        self.length = length

    def to(self, device):
        self.inputs.to(device)
        self.length.to(device)



class CTDataLoader(DataLoader):
    """ Base class DataLoader for loading a 3d dataset. This data loader is designed to work with
    sequential models, and takes care of sorting batches and padding them for the pytorch
    recurrent networks. Note that the dataset MUST BE SORTED BY LENGTH for this to work."""
    def __init__(self, data_dir, batch_size, phase, is_training=True, num_workers=4):
        dataset = CTPEDataset3d(data_dir, phase, is_training)
        self.batch_size_ = batch_size
        self.phase = phase
        super(CTDataLoader, self).__init__(dataset,
                                            batch_size=batch_size,
                                            shuffle=is_training,
                                            num_workers=num_workers,
                                            pin_memory=True)

    def get_series_label(self, series_idx):
        """Get a floating point label for a series at given index."""
        return self.dataset.get_series_label(series_idx)

    @staticmethod
    def pad_sequences(batch):
        """Provides batching for the data loader by padding sequences and stacking them
        into a padded tensor.

        Args:
            batch: List of tensors of shape channels x seq_length x height x width.

        Returns: PaddedInputs object containing the padded sequences and their lengths,
            along with the labels.
        """
        data_batch = [slice_[0] for slice_ in batch]
        seq_lengths = [slice_.shape[1] for slice_ in data_batch]
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.int64)
        target = [item[1] for item in batch]
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
        padded_batch = pad_sequence(data_batch, batch_first=True)
        output = PaddedInputs(padded_batch, seq_lengths)

        return output, target

    def get_series(self, study_num):
        """Get a series with given dset_path. Note: Slow function, avoid this in training."""
        return self.dataset.get_series(study_num)
