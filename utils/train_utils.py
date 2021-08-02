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

if __name__ == '__main__':
    PROJECT_PATH = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\"
    create_training_path(os.path.join(PROJECT_PATH, 'models'))
