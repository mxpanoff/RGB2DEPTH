import numpy as np
import os
import warnings
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

class realsense_generator(Sequence):
    def __init__(self, batch_size, data_path, img_size, shuffle, augment):
        #self.FE_model = FE_model
        #self.fm_shape = FE_model.output_shape[1:]
        self.augment = augment

        # find each rbg image in each subdirctory
        self.setups = {}
        self.num_images = 0
        instances = []
        for filename in os.listdir(data_path):
            instance_num = filename.split('.')[0]
            if instance_num in instances:
                continue
            elif '_' in instance_num:
                continue
            instances.append(instance_num)
            self.num_images += 1

        self.data_path = data_path
        self.img_size = img_size

        self.max_trans = int(self.img_size[0]/10)
        self.min_scale = 0.75
        self.img_center = ((int(self.img_size[0]/2), int(self.img_size[1]/2)))

        self.shuffle = shuffle
        self.img_shape = (*img_size, 3)
        self.batch_size = batch_size

        self.rng = np.random.default_rng()
        self.instances = instances
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.instances[index] for index in indexes]

        # Generate data
        X, y, sample_weights = self.__data_generation(self.list_IDs_temp)

        return X, y, sample_weights

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.zeros((self.batch_size, *self.img_size))
        depths = np.zeros((self.batch_size, *self.img_size))
        sample_weights = np.ones((self.batch_size, *self.img_size[:-1], 1))
        sample_num = 0
        for instance in img_paths_tmp:
            img_path = self.data_path + os.path.sep + instance + '.png'
            img = tf.keras.utils.load_img(img_path, target_size=(self.img_size[0], self.img_size[1]*2, self.img_size[2]))
            img = np.array(img)
            rgb = img[:, :int(self.img_size[1])]
            depth = img[:, int(self.img_size[1]):]


            done_augment = False
            if self.augment:
                while done_augment == False:
                    scale = 1
                    if self.rng.uniform(0, 1) > 0.5:
                        scale = np.random.uniform(self.min_scale, 0.5+self.min_scale)

                    X_trans = 0
                    Y_trans = 0
                    if self.rng.uniform(0, 1) > 0.5:
                        X_trans = self.rng.uniform(-self.max_trans, self.max_trans)
                        Y_trans = self.rng.uniform(-self.max_trans, self.max_trans)

                    theta = 0
                    if self.rng.uniform(0, 1) > 0.5:
                        theta = self.rng.uniform(-180, 180)

                    M = cv2.getRotationMatrix2D((self.img_center[0]+X_trans, self.img_center[1]+Y_trans),
                                                scale=scale,
                                                angle=theta)

                    depth = cv2.warpAffine(depth, M, self.img_size, flags=cv2.INTER_NEAREST)
                    rgb = cv2.warpAffine(rgb, M, self.img_size)

                    if (np.any(depth) and np.any(img)):
                        done_augment = True
                    else:
                        warnings.warn('Improper warp in '+img_path+' all zeros', RuntimeWarning)

                        img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
                        img = np.array(img)
                        rgb = img[:, :int(self.img_size / 2)]
                        depth = img[:, int(self.img_size / 2):]


            imgs[sample_num] = rgb
            depths[sample_num] = depth
            sample_num += 1

        return depths, depths, sample_weights