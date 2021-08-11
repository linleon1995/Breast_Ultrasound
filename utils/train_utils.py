import os
import time
import torch
import argparse
import yaml

# TODO: understand this code
class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

def load_config():
    parser = argparse.ArgumentParser(description='UNet2D')
    parser.add_argument('--config_path', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = load_config_yaml(args.config_path)
    # # Get a device to train on
    # device_str = config.get('device', None)
    # if device_str is not None:
    #     logger.info(f"Device specified in config: '{device_str}'")
    #     if device_str.startswith('cuda') and not torch.cuda.is_available():
    #         logger.warn('CUDA not available, using CPU')
    #         device_str = 'cpu'
    # else:
    #     device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
    #     logger.info(f"Using '{device_str}' device")

    # device = torch.device(device_str)
    # config['device'] = device
    return config


def load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def create_training_path(train_logdir):
    idx = 0
    path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    while os.path.exists(path):
        # if len(os.listdir(path)) == 0:
        #     os.remove(path)
        idx += 1
        path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    os.makedirs(path)
    return path

# TODO: different indent of dataset config, preprocess config, train config
# TODO: recursively
def logging(path, config, access_mode):
    with open(path, access_mode) as fw:
        for dict_key in config:
            dict_value = config[dict_key]
            if isinstance(dict_value , dict):
                for sub_dict_key in dict_value:
                    fw.write(f'{sub_dict_key}: {dict_value[sub_dict_key]}\n')
            else:
                fw.write(f'{dict_key}: {dict_value}\n')

# TODO: create train_loggting.txt automatically
# TODO: check complete after training
# TODO: train config
# TODO: python logging
def train_logging(path, config):
    with open(path, 'r+') as fw:
        if os.stat(path).st_size == 0:
            number = 0
        else:
            for last_line in fw:
                pass
            number = int(last_line.split(' ')[0][1:])
            fw.write('\n')
        local_time = time.ctime(time.time())
        experiment = config['experiment']
        cur_logging = f'#{number+1} {local_time} {experiment}'
        print(cur_logging)
        fw.write(cur_logging)
        
if __name__ == '__main__':
    PROJECT_PATH = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\"
    create_training_path(os.path.join(PROJECT_PATH, 'models'))
