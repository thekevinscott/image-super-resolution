from bz2 import compress
import json
import os
import random
from tqdm import tqdm

import imageio
import numpy as np

from ISR.utils.logger import get_logger
from ISR.utils.utils import sharpen_image, compress_image, write_image


class DataHandler:
    """
    DataHandler generate augmented batches used for training or validation.

    Args:
        lr_dir: directory containing the Low Res images.
        hr_dir: directory containing the High Res images.
        patch_size: integer, size of the patches extracted from LR images.
        scale: integer, upscaling factor.
        n_validation_samples: integer, size of the validation set. Only provided if the
            DataHandler is used to generate validation sets.
    """
    
    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None, should_check_size=True):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}  # image folders
        self.extensions = ('.png', '.jpeg', '.jpg')  # admissible extension
        self.img_list = {}  # list of file names
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        self._make_img_list(should_check_size=should_check_size)
        self._check_dataset()

    def _is_valid_img(self, file, res, should_check_size=True):
        if should_check_size:
            folder = self.folders[res]
            img = imageio.imread(f'{folder}/{file}') / 255.0
            min_side = min(img.shape[0:2])
            if min_side < self.patch_size[res]:
                return False
        return file.endswith(self.extensions)
        #         if not idx:
        #     # randomly select one image. idx is given at validation time.
        #     min_side = 0
        #     while min_side < self.patch_size['lr']:
        #         idx = np.random.choice(range(len(self.img_list['hr'])))
        #         lr_img_path = os.path.join(self.folders['lr'], self.img_list['lr'][idx])
        #         img['lr'] = imageio.imread(lr_img_path) / 255.0
        #         data = np.asarray(img['lr'])
        #         min_side = min(data.shape[0:2])
        #         print('min side', min_side)
        #         print('self patch size', self.patch_size['lr'])
        #     hr_img_path = os.path.join(self.folders['hr'], self.img_list['hr'][idx])
        #     img['hr'] = imageio.imread(hr_img_path) / 255.0
        # else:
        #     print('idx is specified', idx)
        #     for res in ['lr', 'hr']:
        #         img_path = os.path.join(self.folders[res], self.img_list[res][idx])
        #         img[res] = imageio.imread(img_path) / 255.0
        # print('lr shape', img['lr'].shape)
    
    def _make_img_list(self, should_check_size=True):
        """ Creates a dictionary of lists of the acceptable images contained in lr_dir and hr_dir. """
        
        assert len(list(os.listdir(self.folders['hr']))) == len(list(os.listdir(self.folders['lr']))), "Different number of files in lr and hr directories"
        lr_file_names = []
        hr_file_names = []
        for file in tqdm(list(os.listdir(self.folders['lr']))):
            hr_folder = self.folders['hr']
            if os.path.exists(f'{hr_folder}/{file}'):
                if self._is_valid_img(file, 'lr', should_check_size=should_check_size):
                    lr_file_names.append(file)
                    hr_file_names.append(file)
        
        self.img_list['lr'] = np.sort(lr_file_names)
        self.img_list['hr'] = np.sort(hr_file_names)

        # print(self.img_list['hr'])
        # print(self.img_list['lr'])
        if np.array_equal(self.img_list['hr'], self.img_list['lr']) is False:
            raise Exception('Images do not match')
        
        if self.n_validation_samples:
            samples = np.random.choice(
                range(len(self.img_list['hr'])), self.n_validation_samples, replace=False
            )
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]
    
    def _check_dataset(self):
        """ Sanity check for dataset. """
        
        # the order of these asserts is important for testing
        assert len(self.img_list['hr']) == self.img_list['hr'].shape[0], 'UnevenDatasets'
        try:
            assert self._matching_datasets(), 'Input/LabelsMismatch'
        except:
            LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.img_list['lr']]
            HR_name_root = [x.split('.')[0] for x in self.img_list['hr']]
            print(LR_name_root)
            print(HR_name_root)
            raise Exception('Mismatch of datasets')
    
    def _matching_datasets(self):
        """ Rough file name matching between lr and hr directories. """
        # LR_name.png = HR_name+x+scale.png
        # or
        # LR_name.png = HR_name.png
        LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0] for x in self.img_list['hr']]
        return np.all(HR_name_root == LR_name_root)
    
    def _not_flat(self, patch, flatness):
        """
        Determines whether the patch is complex, or not-flat enough.
        Threshold set by flatness.
        """
        
        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True
    
    def _crop_imgs(self, imgs, batch_size, flatness):
        """
        Get random top left corners coordinates in LR space, multiply by scale to
        get HR coordinates.
        Gets batch_size + n possible coordinates.
        Accepts the batch only if the standard deviation of pixel intensities is above a given threshold, OR
        no patches can be further discarded (n have been discarded already).
        Square crops of size patch_size are taken from the selected
        top left corners.
        """
        
        slices = {}
        crops = {}
        crops['lr'] = []
        crops['hr'] = []
        accepted_slices = {}
        accepted_slices['lr'] = []
        top_left = {'x': {}, 'y': {}}
        n = 50 * batch_size
        try:
            for i, axis in enumerate(['x', 'y']):
                top_left[axis]['lr'] = np.random.randint(
                    0, imgs['lr'].shape[i] - self.patch_size['lr'] + 1, batch_size + n
                )
                top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
            for res in ['lr', 'hr']:
                slices[res] = np.array(
                    [
                        {'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])}
                        for x, y in zip(top_left['x'][res], top_left['y'][res])
                    ]
                )
        except Exception as e:
            print('Exception with image')
            print('lr shape', imgs['lr'].shape)
            print('patch size', self.patch_size)
            print('batch_size', batch_size)
            print('n', n)
            raise e
        
        for slice_index, s in enumerate(slices['lr']):
            candidate_crop = imgs['lr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            if self._not_flat(candidate_crop, flatness) or n == 0:
                crops['lr'].append(candidate_crop)
                accepted_slices['lr'].append(slice_index)
            else:
                n -= 1
            if len(crops['lr']) == batch_size:
                break
        
        accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]
        
        for s in accepted_slices['hr']:
            candidate_crop = imgs['hr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            crops['hr'].append(candidate_crop)
        
        crops['lr'] = np.array(crops['lr'])
        crops['hr'] = np.array(crops['hr'])
        return crops
    
    def _apply_transform(self, img, transform_selection, kind, compression_quality=None, sharpen_amount=None, i=None, vary_compression_quality=False):
        """ Rotates and flips input image according to transform_selection. """

        write_image(f'/opt/ml/output/{kind}-{i}-orig.png', img)

        # the type: np.ndarray
        if kind == 'lr' and compression_quality is not None:
            # print('Apply compression of', compression_quality)
            if vary_compression_quality:
                compression_quality = random.randint(compression_quality, 100)
                print('Variable compression, using', compression_quality)
            img = compress_image(img, quality=compression_quality)
            write_image(f'/opt/ml/output/{kind}-{i}-compressed-{compression_quality}.png', img)
        elif kind == 'hr' and sharpen_amount is not None:
            # print('Apply sharpening of', sharpen_amount)
            img = sharpen_image(img, amount=sharpen_amount)
            write_image(f'/opt/ml/output/{kind}-{i}-sharpened-{sharpen_amount}.png', img)
        
        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # rotate left
        }
        
        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # flip along horizontal axis
            2: lambda x: np.flip(x, 1),  # flip along vertical axis
        }
        
        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]
        
        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)
        
        return img
    
    def _transform_batch(self, batch, transforms, kind, compression_quality=None, sharpen_amount=None, vary_compression_quality=False):
        """ Transforms each individual image of the batch independently. """
        
        t_batch = np.array(
            [self._apply_transform(img, transforms[i], kind, compression_quality=compression_quality, sharpen_amount=sharpen_amount, i=i, vary_compression_quality=vary_compression_quality) for i, img in enumerate(batch)]
        )
        return t_batch
    
    def get_batch(self, batch_size, idx=None, flatness=0.0, compression_quality=None, sharpen_amount=None, vary_compression_quality=False):
        """
        Returns a dictionary with keys ('lr', 'hr') containing training batches
        of Low Res and High Res image patches.

        Args:
            batch_size: integer.
            flatness: float in [0,1], is the patch "flatness" threshold.
                Determines what level of detail the patches need to meet. 0 means any patch is accepted.
        """
        
        img = {}
        if not idx:
            # randomly select one image. idx is given at validation time.
            idx = np.random.choice(range(len(self.img_list['hr'])))

        for res in ['lr', 'hr']:
            img_path = os.path.join(self.folders[res], self.img_list[res][idx])
            img[res] = imageio.imread(img_path) / 255.0

        batch = self._crop_imgs(img, batch_size, flatness)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        batch['lr'] = self._transform_batch(batch['lr'], transforms, 'lr', compression_quality=compression_quality, vary_compression_quality=vary_compression_quality)
        batch['hr'] = self._transform_batch(batch['hr'], transforms, 'hr', sharpen_amount=sharpen_amount)
        
        return batch
    
    def get_validation_batches(self, batch_size):
        """ Returns a batch for each image in the validation set. """
        
        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.get_batch(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )
    
    def get_validation_set(self, batch_size):
        """
        Returns a batch for each image in the validation set.
        Flattens and splits them to feed it to Keras's model.evaluate.
        """
        
        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            for res in ('lr', 'hr'):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )
