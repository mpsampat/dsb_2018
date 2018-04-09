#!/usr/bin/env python
import sys
sys.path.append('Mask_RCNN/mrcnn')
import os
import sys
import random
import math
import re
import time
import sklearn
import imgaug
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import imageio
import skimage
from skimage import exposure
from config import Config
from skimage.morphology import label
from skimage.feature import canny
from skimage import exposure
from keras.callbacks import Callback
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.externals import joblib
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.morphology import watershed
from skimage.filters import sobel

import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class LossHistory(Callback):  
    def on_train_begin(self, logs={}):
        self.batch_id = 0
        self.epoch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1
        ctx.channel_send('loss', x=self.batch_id, y=float(logs.get('loss')))
        ctx.channel_send('rpn_class_loss', x=self.batch_id, y=float(logs.get('rpn_class_loss')))
        ctx.channel_send('rpn_bbox_loss', x=self.batch_id, y=float(logs.get('rpn_bbox_loss')))
        ctx.channel_send('mrcnn_class_loss', x=self.batch_id, y=float(logs.get('mrcnn_class_loss')))
        ctx.channel_send('mrcnn_bbox_loss', x=self.batch_id, y=float(logs.get('mrcnn_bbox_loss')))
        ctx.channel_send('mrcnn_mask_loss', x=self.batch_id, y=float(logs.get('mrcnn_mask_loss')))
    
    def on_epoch_end(self, batch, logs={}):
        self.epoch_id += 1
        ctx.channel_send('val_loss', x=self.epoch_id, y=float(logs.get('val_loss')))
        ctx.channel_send('val_rpn_class_loss', x=self.epoch_id, y=float(logs.get('val_rpn_class_loss')))
        ctx.channel_send('val_rpn_bbox_loss', x=self.epoch_id, y=float(logs.get('val_rpn_bbox_loss')))
        ctx.channel_send('val_mrcnn_class_loss', x=self.epoch_id, y=float(logs.get('val_mrcnn_class_loss')))
        ctx.channel_send('val_mrcnn_bbox_loss', x=self.epoch_id, y=float(logs.get('val_mrcnn_bbox_loss')))
        ctx.channel_send('val_mrcnn_mask_loss', x=self.epoch_id, y=float(logs.get('val_mrcnn_mask_loss')))

def train_valid_split(meta, validation_size, valid_category_ids=None):
    meta_train = meta[meta['is_train'] == 1]
    meta_train_split, meta_valid_split = split_on_column(meta_train,
                                                         column='vgg_features_clusters',
                                                         test_size=validation_size,
                                                         random_state=1234,
                                                         valid_category_ids=valid_category_ids
                                                         )
    return meta_train_split, meta_valid_split


def split_on_column(meta, column, test_size, random_state=1, valid_category_ids=None):
    if valid_category_ids is None:
        categories = meta[column].unique()
        np.random.seed(random_state)
        valid_category_ids = np.random.choice(categories,
                                              int(test_size * len(categories)))
    valid = meta[meta[column].isin(valid_category_ids)].sample(frac=1, random_state=random_state)
    train = meta[~(meta[column].isin(valid_category_ids))].sample(frac=1, random_state=random_state)
    return train, valid

meta = pd.read_csv('/public/dsb_2018_data/stage1_metadata.csv')

meta_ts = meta[meta['is_train']==0]
meta_train, meta_valid = train_valid_split( meta[meta['is_train']==1],0.2,[0])

#################################################################################
class DsbConfig(Config):

    # Give the configuration a recognizable name
    NAME = "dsb"
      
    LEARNING_RATE = 1e-3
    RPN_ANCHOR_RATIOS = [0.5, 1, 2] 
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution image
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Train on 1 GPU and 8 images per GPU. Batch size is GPUs * images/GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
    # typically be equal to the number of samples of your dataset divided by the batch size
    STEPS_PER_EPOCH = 312 # 175 * 4 = 700 ~~ train sample size of 670
    VALIDATION_STEPS = 35
    BACKBONE = 'resnet101'
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nucleis
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = "square"

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels, maybe add a 256?
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320 #300
    
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    TRAIN_ROIS_PER_IMAGE = 200
    RPN_NMS_THRESHOLD = 0.7
    MAX_GT_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 400 
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3 # 0.3
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([0.0,0.0,0.0])    
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
config = DsbConfig()
#################################################################################

class InferenceConfig(DsbConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # MEAN_PIXEL = np.array([56.02288505, 54.02376286, 54.26675248])
inference_config = InferenceConfig()
#####################################################################################
class DsbDataset(utils.Dataset):

    def load_dataset(self, ids, train_mode=True):
        self.add_class("dsb", 1, "nuclei")
        if train_mode:
            directory = dsb_dir
        else:
            directory = test_dir
        for i, id in enumerate(ids):
            image_dir = os.path.join(directory, id)
            self.add_image("dsb", image_id=i, path=image_dir)
            

    def load_image(self, image_id, non_zero=None):
        info = self.image_info[image_id]
        path = info['path']
        image_name = os.listdir(os.path.join(path, 'images'))
        image_path = os.path.join(path, 'images', image_name[0])
        image = imageio.imread(image_path)
        if image.shape[2] != 3:
            image = image[:,:,:3]
        image = self.preprocess(image)
        image = image.astype('float32')
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        mask_dir = os.path.join(path, 'masks')
        mask_names = os.listdir(mask_dir)
        count = len(mask_names)
        mask = []
        for i, el in enumerate(mask_names):
            msk_path = os.path.join(mask_dir, el)
            msk = imageio.imread(msk_path)
            if np.sum(msk) == 0:
                print('invalid mask')
                continue
            msk = msk.astype('float32')/255.
            mask.append(msk)
        mask = np.asarray(mask)
        mask[mask > 0.] = 1.
        mask = np.transpose(mask, (1,2,0))
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        count = mask.shape[2]
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = [self.class_names.index('nuclei') for s in range(count)]
        class_ids = np.asarray(class_ids)
        return mask, class_ids.astype(np.int32)
    
    def preprocess(self, img):
        gray = skimage.color.rgb2gray(img.astype('uint8'))
        gray = exposure.equalize_adapthist(gray, nbins=256)
        img = skimage.color.gray2rgb(gray)
        mean = np.mean(img)
        std  = np.std(img)
        #img  = (img-mean)
        img *= 255.
        return img
#############################################################################################
dsb_dir = '/public/dsb_2018_data/stage1_train'
train_ids = meta_train.ImageId.values
val_ids = meta_valid.ImageId.values
test_dir = '/public/dsb_2018_data/stage1_test'
test_ids = os.listdir(test_dir)
#############################################################################
# Training dataset
dataset_train = DsbDataset()
dataset_train.load_dataset(train_ids)
dataset_train.prepare()

# Validation dataset
dataset_val = DsbDataset()
dataset_val.load_dataset(val_ids)
dataset_val.prepare()

# Test dataset
dataset_test = DsbDataset()
dataset_test.load_dataset(test_ids, train_mode=False)
dataset_test.prepare()
##############################################################################
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
################################################################################
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
#########################################################################################
#model.train(dataset_train, dataset_val, 
#                learning_rate=config.LEARNING_RATE,
#                epochs=15, 
#                layers="all")

#augmentation = imgaug.augmenters.Affine(rotate=(-10, 10),scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
augmentation = imgaug.augmenters.Affine(rotate=(-10, -10))
#augmentation = imgaug.augmenters.EdgeDetect(alpha=(0.0,1.0))
#augmentation = imgaug.augmenters.Fliplr(0.5)
augmentation = imgaug.augmenters.OneOf([
               imgaug.augmenters.Fliplr(0.5),
               imgaug.augmenters.AdditiveGaussianNoise(scale=(0, 0.05*255))
               #imgaug.augmenters.Affine(rotate=(-10,10)),
               #imgaug.augmenters.Flipud(1.0)
])
#augmentation = None
model.train(dataset_train, dataset_val,
                     learning_rate=config.LEARNING_RATE,
                     epochs=25,
                     layers='all',
                     augmentation=augmentation)

#model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE/10,
#                epochs=10,
#                layers="all")
#model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE/100,
#                epochs=10,
#                layers="all")
os.system('sudo poweroff')
