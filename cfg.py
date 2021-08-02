# TODO: better way to wrap config parameters
# TODO: split preprocessing params and flag
import os
from utils import train_utils

# EPOCH = 300
# BATCH_SIZE = 6
# SHUFFLE = True
# LEARNING_RATE = 1e-2
# CHECKPOINT_SAVING_STEPS = 50
PROJECT_PATH = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\"
DATA_PATH = os.path.join(PROJECT_PATH, "archive\\Dataset_BUSI_with_GT")
CHECKPOINT_PATH = train_utils.create_training_path(os.path.join(PROJECT_PATH, 'models'))
SHOW_PREPROCESSING = False

preprocess_config = {'crop_size': (512,512),
                     'PadToSquare': False,
                     'HorizontalFlip': True,
                     'RandCrop': True,
                     'RandScale': True,
                     'ScaleToSize': False,
                     'ScaleLimitSize': True,
                     'padding_height': 512,
                     'padding_width': 512,
                     'padding_value': 0.0,
                     'flip_prob': 0.5,
                     'min_scale_factor': 0.75,
                     'max_scale_factor': 1.0,
                     'step_size': 0.125,
                     'resize_method': 'Bilinear',
                     'flip_prob': 0.5,
                     'scale_size': (512, 512),
                     }
dataset_config = {'data_path': DATA_PATH,
                  'data_split': (0.7,0.3),
                  'preprocess_config': preprocess_config}
# train_config = {
#                 'epoch': EPOCH,
#                 'batch_size': BATCH_SIZE,
#                 'shuffle': SHUFFLE,
#                 'lr': LEARNING_RATE,
#                 'checkpoint_saving_steps': CHECKPOINT_SAVING_STEPS,
#                 'project_path': PROJECT_PATH,
#                 'data_path': DATA_PATH,
#                 'checkpoint_path': CHECKPOINT_PATH
#                 }