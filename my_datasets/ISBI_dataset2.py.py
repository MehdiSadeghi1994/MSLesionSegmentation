from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
import os 
import albumentations as A
import random
import torch
import itertools

import torchvision.transforms as transform


class ISBI_DataSet(Dataset):

    def __init__(self, dataset_path, input_shape, output_shape, lesion_pr=0.6, num_patch=10, n_tiles=[5,5,5], for_data='train', num_modality=1, multi_task=False, transform=None, two_label=False):
        super(ISBI_DataSet, self).__init__()
        self.transform = transform
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lesion_pr = lesion_pr
        self.file_names = self.traverse_dir(dataset_path, for_data)
        self.for_data = for_data
        self.num_patch = num_patch
        self.n_tiles = n_tiles
        self.two_label = two_label
        self.num_modality = num_modality
        self.multi_task = multi_task


    def __getitem__(self, idx):
        if self.for_data == 'train':
            idx = idx // self.num_patch

            flair_image_file = self.file_names["flair_image"][idx]
            t1_image_file = self.file_names["t1_image"][idx]

            flair_image = load_nifti(flair_image_file)
            t1_image = load_nifti(t1_image_file)

            if self.num_modality == 2:
                image = np.concatenate((np.expand_dims(flair_image, axis=4), np.expand_dims(t1_image, axis=4)), axis=4)
            else:
                image = flair_image
                
            mask_file = self.file_names["mask1"][idx]
            mask = load_nifti(mask_file)
            
            if self.two_label:
                mask_2_file = self.file_names["mask2"][idx]
                mask_2 = load_nifti(mask_2_file)
                
                mask = self.mix_labels(mask, mask_2)
                del mask_2

            if self.transform:
                seed = int(np.random.randint(0,100,1))
                random.seed(seed)
                # temp_image = image[:,:,:,0]
                augmented_0 = self.transform(image=image, mask=mask)
                temp_image = image[:,:,:,1]
                augmented_1 = self.transform(image=temp_image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            print(image.shape)
            image_patch, mask_patch = image, mask #self.get_random_patch(image, mask, self.input_shape, self.output_shape, self.lesion_pr, self.num_modality)

            image_patch = image_patch.transpose(3, 0, 1, 2)
            # if self.multi_task:
            #     flair_patch = np.multiply(image_patch[0,10:-10,10:-10,10:-10], np.logical_or(mask_patch[:,:,:,0], mask_patch[:,:,:,1]) )
            #     flair_lesion = flair_patch[np.nonzero(flair_patch)]
            #     mean_lesion = np.average(flair_lesion)
            #     std_lesion = np.std(flair_lesion)
                
            image_patch = torch.tensor(image_patch, dtype=torch.float32)
            mask_patch = torch.tensor(mask_patch, dtype=torch.long)
            return image_patch, mask_patch#, mean_lesion, std_lesion

        else:
            flair_image_file = self.file_names["flair_image"][idx]
            t1_image_file = self.file_names["t1_image"][idx]

            flair_image = load_nifti(flair_image_file)
            t1_image = load_nifti(t1_image_file)
            # image_file = self.file_names['flair_image'][idx]
            # image = load_nifti(image_file)
            # image = np.expand_dims(image, axis=-1)
            if self.num_modality == 2:
                image = np.concatenate((np.expand_dims(flair_image, axis=4), np.expand_dims(t1_image, axis=4)), axis=4)
            else:
                image = flair_image
            if self.for_data == 'validation':
                mask_file = self.file_names["mask1"][idx]
                mask = load_nifti(mask_file)
                data = {'patches':[], 'label':mask, 'cordinates':[], 'image_shape':image.shape}

                if self.two_label:
                  mask_2_file = self.file_names["mask2"][idx]
                  mask_2 = load_nifti(mask_2_file)
                  data = {'patches':[], 'label':self.mix_labels(mask, mask_2), 'cordinates':[], 'image_shape':image.shape}
                
            else:
                data = {'patches':[], 'cordinates':[], 'image_shape':image.shape, 'image_path':flair_image_file}

            centers = [[] for _ in range(3)]
            
            for img_len, len_out, center, n_tile in zip(image.shape, self.output_shape, centers, self.n_tiles):
        
        
                # if img_len >= len_out * n_tile:
                    # raise ValueError(f"{img_len} must be smaller than {len_out} x {n_tile}")
                stride = (img_len - len_out) // (n_tile - 1)
                center.append(len_out // 2)
                for i in range(n_tile - 2):
                    center.append(center[-1] + stride)
                center.append(img_len - len_out // 2)
    
            for x, y, z in itertools.product(*centers):
                patch = self.crop_patch(image, [x, y, z], self.input_shape, self.num_modality)
                # patch = np.expand_dims(patch, axis=0)

                data['cordinates'].append([x,y,z])
                data['patches'].append(patch)

            return data
        

    def __len__(self):
        if self.for_data == 'train':
            Len = len(self.file_names["flair_image"])*self.num_patch
        else:
            Len = len(self.file_names["flair_image"])

        return Len

    def __str__(self):
        return f'{self.__class__.__name__} DatSet\n Number of Samples is: {len(self)}\n'
    
    @staticmethod
    def mix_labels(label_1, label_2):
      # np.logical_or(label_1, label_2)
      # np.logical_and(label_1, label_2)
      # 
      return np.concatenate((np.expand_dims(label_1, axis=3), np.expand_dims(label_2, axis=3)), axis=3)

    @staticmethod
    def traverse_dir(dir, for_data):
        data = {'flair_image':[], 't1_image':[], 'mask1':[], 'mask2':[]}
        # data = {'image':[], 'mask_1':[], 'mask_2':[]}

        if for_data == 'test':
            data = {'flair_image':[], 't1_image':[]}

        for idx in os.listdir(dir):
            idx_folder = os.path.join(dir, idx)
            # image_address = os.path.join(idx_folder, idx+'_preprocessed.nii.gz')
            # data['image'].append(image_address)

            flair_image_address = os.path.join(idx_folder, idx+'_FLAIR_Pre.nii.gz')
            t1_image_address = os.path.join(idx_folder, idx+'_T1_Pre.nii.gz')
            data['flair_image'].append(flair_image_address)
            data['t1_image'].append(t1_image_address)


            if for_data == 'train' or for_data == 'validation':
                mask_1_address = os.path.join(idx_folder, idx+'_MASK_1.nii.gz')
                mask_2_address = os.path.join(idx_folder, idx+'_MASK_2.nii.gz')
                # mask2_address = ''
                data['mask1'].append(mask_1_address)
                data['mask2'].append(mask_2_address)

        return data

    @staticmethod
    def crop_patch(image, center, shape, num_modality):
        mini = [c - len_ // 2 for c, len_ in zip(center, shape)]
        maxi = [c + len_ // 2 for c, len_ in zip(center, shape)]
        if all(m >= 0 for m in mini) and all(m < img_len for m, img_len in zip(maxi, image.shape)):
            slices = [slice(mi, ma) for mi, ma in zip(mini, maxi)]
        else:
            slices = [
                np.clip(range(mi, ma), 0, img_len - 1)
                for mi, ma, img_len in zip(mini, maxi, image.shape)
            ]
            slices = np.meshgrid(*slices, indexing="ij")
        patch = image[slices]
        if num_modality == 2:
            patch = patch.transpose(3, 0, 1, 2, 4)
        else:
            patch = patch.transpose(3, 0, 1, 2)
        return patch
    @staticmethod
    def get_random_patch(image, label, input_shape, output_shape, pr_lesion, num_modality):
        pr = np.random.rand()
        if pr < pr_lesion:
            mask = np.int32(label > 0)
            slices = [slice(len_ // 2, -len_ // 2) for len_ in input_shape] 
            mask[tuple(slices)] *= 2
            indices = np.where(mask > 1.5)
            i = np.random.choice(len(indices[0]))
            input_slices = [
                slice(index[i] - len_ // 2, index[i] + len_ // 2)
                for index, len_ in zip(indices, input_shape)
            ]
            output_slices = [
                slice(index[i] - len_ // 2, index[i] + len_ // 2)
                for index, len_ in zip(indices, output_shape)
            ]
            image_patch = image[tuple(input_slices)]
            label_patch = label[tuple(output_slices)]
            if num_modality ==2:
                image_patch = image_patch.transpose(3, 0, 1, 2, 4)
            else: 
                image_patch = image_patch.transpose(3, 0, 1, 2)

        else:
            mask = np.int32(label <= 0)
            slices = [slice(len_ // 2, -len_ // 2) for len_ in input_shape] 
            mask[tuple(slices)] -= 2
            indices = np.where(mask == -1)
            i = np.random.choice(len(indices[0]))
            input_slices = [
                slice(index[i] - len_ // 2, index[i] + len_ // 2)
                for index, len_ in zip(indices, input_shape)
            ]
            output_slices = [
                slice(index[i] - len_ // 2, index[i] + len_ // 2)
                for index, len_ in zip(indices, output_shape)
            ]
            image_patch = image[tuple(input_slices)]
            label_patch = label[tuple(output_slices)]
            if num_modality ==2:
                image_patch = image_patch.transpose(3, 0, 1, 2, 4)
            else: 
                image_patch = image_patch.transpose(3, 0, 1, 2)

        return image_patch, label_patch

def load_nifti(filename, with_affine=False):
    
    img = nib.load(filename)
    data = img.get_data()
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data
   
