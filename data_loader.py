import os
import cv2
import tifffile
import numpy as np
from PIL import Image
import albumentations as A

import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """ Image Data Generator that loads and yield image data for each batch 
        link: https://github.com/keras-team/keras/issues/8130#issuecomment-406047311
              https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, img_files, task, gray=False, batch_size=32, shuffle=False, target_size=(256,256), augmentation=False):
        """
        Args:
            img_files: A list of path to image files.
            gray: Read grayscale image.
            shuffle: Shuffle data after each epoch.
        """
        self.img_files = img_files
        self.task = task
        self.gray = gray
        self.img_shape = target_size
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = self.image_augmentation()
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self.img_files) / self.batch_size))
        # return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X,Y = self.__data_generation(img_files_temp)

        return X,Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def image_augmentation(self):

        if self.task=="reconstruction":
            transform = A.Compose([])

        elif self.task=="denoise":
            transform = A.Compose([A.GaussNoise (var_limit=(10000,10000), mean=5, per_channel=True, always_apply=True)])

        elif self.task=="self-resolution":
            transform = A.Compose([A.Downscale (scale_min=0.1, scale_max=0.1, interpolation=0, always_apply=True)])

        elif self.task=="colorization":
            transform = A.Compose([A.FancyPCA (alpha=0.08, always_apply=True)])

        elif self.task=="rotation":
            # 角度はどうする？？これできるの?? # border_modeも変更した方が良いかも
            transform = A.Compose([A.RandomRotate90(always_apply=True)])
            # transform = A.Compose([A.SafeRotate(limit=10, interpolation=1, border_mode=1, value=None, mask_value=None, always_apply=True)])
            # transform = A.Compose([A.Rotate(limit=10, interpolation=1, border_mode=1, value=None, mask_value=None, always_apply=True)])

        elif self.task=="inpaint":
            transform = A.Compose([A.CoarseDropout(max_holes=32, max_height=32, max_width=32, always_apply=True)])

        else:
            raise ValueError("Invalid task name !!")
    
        return transform

    def image_transform(self,img):
        if self.task=="reconstruction":
            pass
        elif self.task=="denoise":
            pass            
        elif self.task=="self-resolution":
            pass
        elif self.task=="colorization":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif self.task=="rotation":
            pass
        elif self.task=="inpaint":
            pass
        else:
            ValueError("dfa")

        return img

    def __data_generation(self, img_files_temp):
        """ Generates data containing batch_size samples. """
        
        X,Y=[],[]
        # Generate data
        for i, img_file in enumerate(img_files_temp):

            # Read image (ignore grayscale)
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_shape)

            # 前処理
            x = self.transform(image=img)['image']

            img = self.image_transform(img)
            x = self.image_transform(x)
            
            # Data Augmentation
            # if self.augmentation:
            #     augmented = self.transforms(image=img)
            #     img = augmented['image']

            X.append(x)
            Y.append(img)
        
        X = np.array(X).astype(np.float32)/255.
        Y = np.array(Y).astype(np.float32)/255.

        return X,Y


def compress(x,size=8):    
    x_shape = x.shape[:-1]
    x_resized = cv2.resize(x,(size,size), cv2.INTER_NEAREST)
    x_resized = cv2.resize(x_resized, x_shape, cv2.INTER_NEAREST)
    return x_resized

def normalize(x):
    x_norm = x / np.max(x)
    return x_norm

def binarize(x,threshold=0.2):
    x_bin = np.where(x>threshold,1,0)    
    return x_bin


class ClassificationDataLoader(tf.keras.utils.Sequence):
    def __init__(self, img_list, attMap_list, label,
                input_method, output_method, pre_transform, batch_size=16):
        self.img_list = img_list
        self.attMap_list = attMap_list
        self.label = label 
        self.input_method = input_method
        self.output_method = output_method
        self.batch_size = batch_size
        self.pre_transform = pre_transform

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self.img_list)/self.batch_size))
    
    def __getitem__(self, idx):
        batch_img_list = self.img_list[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_attMap_list = self.attMap_list[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_label = self.label[idx*self.batch_size:(idx+1)*self.batch_size]

        # files -> imgs
        batch_imgs = np.array([np.array(Image.open(f_img).convert('RGB'), np.float32) for f_img in batch_img_list])
        batch_attMaps = np.array([np.array(Image.open(f_img).convert('L'), np.float32)[:,:,np.newaxis] for f_img in batch_attMap_list])

        # Transform
        batch_imgs_transform = self.pre_transform(batch_imgs)
        batch_attMaps_transform = self.pre_transform(batch_attMaps)
        
        # Make inputs and outputs for model.
        inputs = self._make_input(batch_imgs_transform , batch_attMaps_transform, self.input_method)
        outputs = self._make_output(batch_label, self.output_method)

        return inputs, outputs

    def _make_input(self, img, attMap, method):
        if method=='img_and_attMap':
            return [img, attMap]

        elif method=='img_only':
            return img

        elif method=='attMap_only':
            return attMap

        elif method=='4_channel':
            return np.concatenate([img,attMap],axis=-1)

        elif method=='attMap_to_img':
            return img + (img*attMap)

        elif method=='multiply':
            return img * attMap

    def _make_output(self,label, method):
        if method=='ADN_only':
            return [label]

        elif method=='AAN_and_ADN':
            return [label, label]