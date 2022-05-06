# Dam-Reservoir-Extraction
This is the code base for IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING (TGRS 2022) paper [Dam reservoir extraction from remote sensing
imagery using tailored metric learning strategies](https://ieeexplore.ieee.org/document/9768672)

# Prepare data
Download dataset from [google drive](https://drive.google.com/file/d/14ley7T2J0Vy2rP1ezDq5WH2coJPBvzVc/view?usp=sharing) or [BaiduYun (code: kd6c)](https://pan.baidu.com/s/1akK5K5gyircO51mmqgdA7A) and unzip it into the the ./dataset directory as below.
```
├── dataset
│   ├── classification
│   │   ├── train
│   │   |   ├── 0
│   │   |   ├── 1
│   │   ├── valid
│   │   |   ├── 0
│   │   |   ├── 1
│   │   ├── test
│   │   |   ├── 0
│   │   |   ├── 1
│   ├── segmentation
│   │   ├── train
│   │   |   ├── images
│   │   |   ├── labels
│   │   ├── valid
│   │   |   ├── images
│   │   |   ├── labels
│   │   ├── test
│   │   |   ├── images
│   │   |   ├── labels

```

# Main Results
![](./results.png) 

The segmentation model takes DeepLabV3+ as the backbone. We use an input image resolution of 256 × 256, batch size of 4, initial learning rate of 0.0003, and apply the polynomial decay with a factor of 0.9. The network is optimized by the Adam optimizer with 150 epochs in total. Data augmentation is applied to training images, which includes random horizontal and vertical flip, random brightness change, random rotation and random channel shift.

The classification model takes ResNet50V2 [13] as the backbone, which is pretrained on the ILSVRC classification task. We use an input image resolution of 224 × 224, batch size of 64, and learning rate of 1e-4. The network is optimized by the Adam optimizer with 400 epochs in total. Each imagefeature is l2-normalized.

You can download our well-trained weights from [google drive](https://drive.google.com/file/d/1XLPNBRQQXEjXp5IXjxd9VEqOKJwPinSa/view?usp=sharing) or [BaiduYun (code: iy4i)](https://pan.baidu.com/s/1T7XuQSNo3Hw3HP4EkJatsg) and unzip it into ./checkpoint directory to reproduce the results.

# Installation
## Dependencies
+ python 3.7
+ Tensorflow 3.6.0

# Get Started

## Water body segmentation

The segmentation directory contains complete codes for model training. 
Start with a simple command as this:
```buildoutcfg
python train.py
```
The detailed command line parameters are as follows:
```buildoutcfg
usage: train.py [-h] --model MODEL [--base_model BASE_MODEL] --dataset DATASET
                [--loss {CE,Focal_Loss}] --num_classes NUM_CLASSES
                [--random_crop RANDOM_CROP] [--crop_height CROP_HEIGHT]
                [--crop_width CROP_WIDTH] [--batch_size BATCH_SIZE]
                [--valid_batch_size VALID_BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--initial_epoch INITIAL_EPOCH]
                [--h_flip H_FLIP] [--v_flip V_FLIP]
                [--brightness BRIGHTNESS [BRIGHTNESS ...]]
                [--rotation ROTATION]
                [--zoom_range ZOOM_RANGE [ZOOM_RANGE ...]]
                [--channel_shift CHANNEL_SHIFT]
                [--data_aug_rate DATA_AUG_RATE]
                [--checkpoint_freq CHECKPOINT_FREQ]
                [--validation_freq VALIDATION_FREQ]
                [--num_valid_images NUM_VALID_IMAGES]
                [--data_shuffle DATA_SHUFFLE] [--random_seed RANDOM_SEED]
                [--weights WEIGHTS]

```
```buildoutcfg
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Choose the semantic segmentation methods.
  --base_model BASE_MODEL
                        Choose the backbone model.
  --dataset DATASET     The path of the dataset.
  --loss {CE,Focal_Loss}
                        The loss function for traning.
  --num_classes NUM_CLASSES
                        The number of classes to be segmented.
  --random_crop RANDOM_CROP
                        Whether to randomly crop the image.
  --crop_height CROP_HEIGHT
                        The height to crop the image.
  --crop_width CROP_WIDTH
                        The width to crop the image.
  --batch_size BATCH_SIZE
                        The training batch size.
  --valid_batch_size VALID_BATCH_SIZE
                        The validation batch size.
  --num_epochs NUM_EPOCHS
                        The number of epochs to train for.
  --initial_epoch INITIAL_EPOCH
                        The initial epoch of training.
  --h_flip H_FLIP       Whether to randomly flip the image horizontally.
  --v_flip V_FLIP       Whether to randomly flip the image vertically.
  --brightness BRIGHTNESS [BRIGHTNESS ...]
                        Randomly change the brightness (list).
  --rotation ROTATION   The angle to randomly rotate the image.
  --zoom_range ZOOM_RANGE [ZOOM_RANGE ...]
                        The times for zooming the image.
  --channel_shift CHANNEL_SHIFT
                        The channel shift range.
  --data_aug_rate DATA_AUG_RATE
                        The rate of data augmentation.
  --checkpoint_freq CHECKPOINT_FREQ
                        How often to save a checkpoint.
  --validation_freq VALIDATION_FREQ
                        How often to perform validation.
  --num_valid_images NUM_VALID_IMAGES
                        The number of images used for validation.
  --data_shuffle DATA_SHUFFLE
                        Whether to shuffle the data.
  --random_seed RANDOM_SEED
                        The random shuffle seed.
  --weights WEIGHTS     The path of weights to be loaded.
```

## Dam reservoir recognition

Just start with our python script in the notebook under ./classification directory.

## Dam reservoir extraction

Just start with our python script in the notebook under ./pipeline directory.

# Citation

```
@ARTICLE{9768672,
  author={Van Soesbergen, Arnout and Chu, Zedong and Shi, Miaojing and Mulligan, Mark},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Dam reservoir extraction from remote sensing imagery using tailored metric learning strategies}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2022.3172883}}
```
