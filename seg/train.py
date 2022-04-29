"""
The file defines the training process.

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.losses import *
from utils.learning_rate import *
from utils.metrics import *
from utils import utils
from builders import builder
import tensorflow as tf
import argparse
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, default='DeepLabV3Plus')
parser.add_argument('--base_model', help='Choose the backbone model.', type=str, default='Xception-DeepLab')
parser.add_argument('--dataset', help='The path of the dataset.', type=str, default='../dataset/segmentation')
parser.add_argument('--loss', help='The loss function for traing.', type=str, default='focal_loss',
                    choices=['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss'])
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, default=2)
parser.add_argument('--random_crop', help='Whether to randomly crop the image.', type=str2bool, default=False)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--batch_size', help='The training batch size.', type=int, default=4)
parser.add_argument('--valid_batch_size', help='The validation batch size.', type=int, default=4)
parser.add_argument('--num_epochs', help='The number of epochs to train for.', type=int, default=150)
parser.add_argument('--initial_epoch', help='The initial epoch of training.', type=int, default=0)
parser.add_argument('--h_flip', help='Whether to randomly flip the image horizontally.', type=str2bool, default=True)
parser.add_argument('--v_flip', help='Whether to randomly flip the image vertically.', type=str2bool, default=True)
parser.add_argument('--brightness', help='Randomly change the brightness (list).', type=float, default=None, nargs='+')
parser.add_argument('--rotation', help='The angle to randomly rotate the image.', type=float, default=1.0)
parser.add_argument('--zoom_range', help='The times for zooming the image.', type=float, default=1.0, nargs='+') 
parser.add_argument('--channel_shift', help='The channel shift range.', type=float, default=1.0)
parser.add_argument('--data_aug_rate', help='The rate of data augmentation.', type=float, default=1.0)
parser.add_argument('--checkpoint_freq', help='How often to save a checkpoint.', type=int, default=1)
parser.add_argument('--validation_freq', help='How often to perform validation.', type=int, default=1)
parser.add_argument('--num_valid_images', help='The number of images used for validation.', type=int, default=59)
parser.add_argument('--data_shuffle', help='Whether to shuffle the data.', type=str2bool, default=True)
parser.add_argument('--random_seed', help='The random shuffle seed.', type=int, default=0)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=float, default=False)
parser.add_argument('--steps_per_epoch', help='The training steps of each epoch', type=int, default=None)
parser.add_argument('--lr_scheduler', help='The strategy to schedule learning rate.', type=str, default='poly_decay',
                    choices=['step_decay', 'poly_decay', 'cosine_decay']) 
parser.add_argument('--lr_warmup', help='Whether to use lr warm up.', type=bool, default=False) 
parser.add_argument('--learning_rate', help='The initial learning rate.', type=float, default=3e-4) 
parser.add_argument('--optimizer', help='The optimizer for training.', type=str, default='adam', 
                    choices=['sgd', 'adam', 'nadam', 'adamw', 'nadamw', 'sgdw'])
parser.add_argument('--stage', help='stage', type=str, default='c5', 
                    choices=['c1', 'c2', 'c3', 'c4', 'c5'])
parser.add_argument('--cuda', help='Choose the running gpu.', type=str, default='0')
parser.add_argument('--n', help='anchor.', type=int, default=50)
parser.add_argument('--margin', help='margin.', type=float, default=0.01)
parser.add_argument('--weight', help='weight.', type=float, default=0.01)

args = parser.parse_args()

stage = args.stage
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda
N = args.n
margin = args.margin
weight = args.weight

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
train_image_names, train_label_names, valid_image_names, valid_label_names, _, _ = get_dataset_info(args.dataset)

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

# load weights
if args.weights:
    print('Loading the weights...')
    net.load_weights('')

# chose loss
losses = {'ce': categorical_crossentropy_with_logits,
          'focal_loss': focal_loss(),
          'miou_loss': miou_loss(num_classes=args.num_classes),
          'self_balanced_focal_loss': self_balanced_focal_loss()}

loss = {
    'tf.concat': metric_loss(stage,N,margin),
    'seg': focal_loss() 
}


# chose optimizer
total_iterations = len(train_image_names) * args.num_epochs // args.batch_size
wd_dict = utils.get_weight_decays(net)
ordered_values = []
weight_decays = utils.fill_dict_in_order(wd_dict, ordered_values)

optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
              'nadam': tf.keras.optimizers.Nadam(learning_rate=args.learning_rate),
              'sgd': tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.99),
              'adamw': AdamW(learning_rate=args.learning_rate, batch_size=args.batch_size,
                             total_iterations=total_iterations),
              'nadamw': NadamW(learning_rate=args.learning_rate, batch_size=args.batch_size,
                               total_iterations=total_iterations),
              'sgdw': SGDW(learning_rate=args.learning_rate, momentum=0.99, batch_size=args.batch_size,
                           total_iterations=total_iterations)}

# lr schedule strategy
if args.lr_warmup and args.num_epochs - 5 <= 0:
    raise ValueError('num_epochs must be larger than 5 if lr warm up is used.')

lr_decays = {'step_decay': step_decay(args.learning_rate, args.num_epochs - 5 if args.lr_warmup else args.num_epochs,
                                      warmup=args.lr_warmup),
             'poly_decay': poly_decay(args.learning_rate, args.num_epochs - 5 if args.lr_warmup else args.num_epochs,
                                      warmup=args.lr_warmup),
             'cosine_decay': cosine_decay(args.num_epochs - 5 if args.lr_warmup else args.num_epochs,
                                          args.learning_rate, warmup=args.lr_warmup)}
lr_decay = lr_decays[args.lr_scheduler]

# training and validation steps
steps_per_epoch = len(train_image_names) // args.batch_size if not args.steps_per_epoch else args.steps_per_epoch
validation_steps = args.num_valid_images // args.valid_batch_size

# compile the model
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics={'seg':[BiIOU()]},
        run_eagerly=True,
        loss_weights={'tf.concat':weight,'seg': 1} 
)

# data generator
# data augmentation setting
train_gen = ImageDataGenerator(random_crop=args.random_crop,
                               rotation_range=args.rotation,
                               brightness_range=args.brightness,
                               zoom_range=args.zoom_range,
                               channel_shift_range=args.channel_shift,
                               horizontal_flip=args.v_flip,
                               vertical_flip=args.v_flip)

valid_gen = ImageDataGenerator()

train_generator = train_gen.flow(images_list=train_image_names,
                                 labels_list=train_label_names,
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
                                 target_size=(args.crop_height, args.crop_width),
                                 shuffle=args.data_shuffle,
                                 seed=args.random_seed,
                                 data_aug_rate=args.data_aug_rate)

valid_generator = valid_gen.flow(images_list=valid_image_names,
                                 labels_list=valid_label_names,
                                 num_classes=args.num_classes,
                                 batch_size=args.valid_batch_size,
                                 target_size=(args.crop_height, args.crop_width))

# callbacks setting
# checkpoint setting
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(
        '../checkpoint/'+
        '{model}_based_on_{base}.h5'.format(model=args.model, base=base_model) ),
    save_best_only=True, period=args.checkpoint_freq, monitor='val_seg_biou', mode='max', save_weights_only=True) 

# tensorboard setting
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/')
# learning rate scheduler setting
learning_rate_scheduler = LearningRateScheduler(lr_decay, args.learning_rate, args.lr_warmup, steps_per_epoch,
                                                verbose=1)

callbacks = [model_checkpoint, tensorboard, learning_rate_scheduler]# 

# begin training
print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Num Images -->", len(train_image_names))
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Initial Epoch -->", args.initial_epoch)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", args.num_classes)

print("Data Augmentation:")
print("\tData Augmentation Rate -->", args.data_aug_rate)
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("\tZoom -->", args.zoom_range)
print("\tChannel Shift -->", args.channel_shift)

print("")

# training...
net.fit(train_generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=args.num_epochs,
                  callbacks=callbacks,
                  validation_data=valid_generator,
                  validation_steps=validation_steps,
                  validation_freq=args.validation_freq,
                  max_queue_size=10,
                  workers=os.cpu_count(),
                  use_multiprocessing=False,
                  initial_epoch=args.initial_epoch,
                    verbose=2,
       )
