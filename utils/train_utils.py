import os

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

def logging(path, config, filename, access_mode):
    with open(os.path.join(path, filename), access_mode) as fw:
        for dict_key in config:
            dict_value = config[dict_key]
            if isinstance(dict_value , dict):
                for sub_dict_key in dict_value:
                    fw.write(f'{sub_dict_key}: {dict_value[sub_dict_key]}\n')
            else:
                fw.write(f'{dict_key}: {dict_value}\n')

if __name__ == '__main__':
    PROJECT_PATH = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\"
    create_training_path(os.path.join(PROJECT_PATH, 'models'))
