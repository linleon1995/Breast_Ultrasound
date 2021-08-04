import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cfg import dataset_config
from dataset.dataloader import ImageDataset
from model import UNet_2d
from utils import train_utils
from utils import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

EPOCH = 500
BATCH_SIZE = 6
SHUFFLE = True
PROJECT_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound'
DATAPATH = os.path.join(PROJECT_PATH, rf'archive\Dataset_BUSI_with_GT')
CHECKPOINT = train_utils.create_training_path(os.path.join(PROJECT_PATH, 'models'))
LEARNING_RATE = 1e-2
SAVING_STEPS = 50


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=EPOCH,
                    help='Model training epoch.')

parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                    help='Training batch size.')

parser.add_argument('--shuffle', type=bool, default=SHUFFLE,
                    help='The flag of shuffling input data.')

parser.add_argument('--datapath', type=str, default=DATAPATH,
                    help='')

parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT,
                    help='')

# TODO: Check Dice Loss implementation cur: smooth 1 --> 1e-5
# TODO: Iprove the visualization of training process
# TODO: tensorboard
# TODO: step time
class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1e-5
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = (2*intersection.sum(1)  + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
		return loss

def main():
    net = UNet_2d(input_channels=1, num_class=1)
    if torch.cuda.is_available():
        net.cuda()
    # TODO: Select optimizer by config file
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Dataloader
    train_dataset = ImageDataset(dataset_config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle)

    # dataset_config.pop('preprocess_config')
    test_dataset_config = dataset_config.copy()
    test_dataset_config.pop('preprocess_config')
    test_dataset = ImageDataset(test_dataset_config, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    # TODO: training parameter to cfg.py
    # TODO: modify vars name

    # Start training
    training_samples = len(train_dataloader.dataset)
    step_loss, total_train_loss= [], []
    total_test_acc, total_test_loss  = [], []
    min_loss = 1e5
    max_acc = -1
    saving_steps = SAVING_STEPS
    training_steps = int(training_samples/FLAGS.batch_size)
    if training_samples%FLAGS.batch_size != 0:
        training_steps += 1
    testing_steps = len(test_dataloader.dataset)
    experiment = [s for s in FLAGS.checkpoint_path.split('\\') if 'run' in s][0]
    times = 5
    level = training_steps//times
    length = 0
    temp = level
    while (temp):
        temp = temp // 10
        length += 1
    level = round(level / 10**(length-1)) * 10**(length-1)
    print("Start Training!!")
    print("Training epoch: {} Batch size: {} Shuffling Data: {} Training Samples: {}".
            format(FLAGS.epoch, FLAGS.batch_size, FLAGS.shuffle, training_samples))
    print(60*"-")
    config = dataset_config.copy()
    config['epoch'], config['batch_size'], config['lr'] = FLAGS.epoch, FLAGS.batch_size, LEARNING_RATE
    # TODO: different indent of dataset config, preprocess config, train config
    train_utils.logging(os.path.join(FLAGS.checkpoint_path, 'logging.txt'), config, access_mode='w+')
    # TODO: train_logging
    config['experiment'] = experiment
    train_utils.train_logging(os.path.join(PROJECT_PATH, 'models', 'train_logging.txt'), config)

    for epoch in range(1, FLAGS.epoch+1):
        total_loss = 0.0
        for i, data in enumerate(train_dataloader):
            net.train()
            # print('Epoch: {}/{}  Step: {}'.format(epoch, FLAGS.epoch, i))
            inputs, labels = data['input'], data['gt']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # TODO: select loss
            # loss = nn.BCELoss()(outputs, labels)
            loss = DiceLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_loss.append(loss)
            
            if i%level == 0:
                print('Step {}  Step loss {}'.format(i, loss))
        total_train_loss.append(total_loss/training_steps)
        print('**Epoch {}/{}  Epoch loss {}  Step loss {}'.
            format(epoch, FLAGS.epoch, total_train_loss[-1], step_loss[-1]))
        with torch.no_grad():
            net.eval()
            # loss_list = []
            test_loss, test_acc = 0.0, 0.0
            for _, data in enumerate(test_dataloader):
                inputs, labels = data['input'], data['gt']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                test_loss += DiceLoss()(outputs, labels)
                # loss_list.append(test_loss)

                prediction = torch.round(outputs)
                test_acc += metrics.dsc(prediction, labels)
            avg_test_acc = test_acc / testing_steps
            avg_test_loss = test_loss / testing_steps
            total_test_loss.append(avg_test_loss)
            total_test_acc.append(avg_test_acc)
            print("**Testing Loss:{:.3f}".format(avg_test_loss))
            
            
            # Saving model
            # TODO: add run000 in front of ckpt name and plot name
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch
                }
                
            if epoch%saving_steps == 0:
                print("Saving model with testing loss {:.3f} in epoch {} ".format(0.0, epoch))
                checkpoint_name = 'ckpt_best_{:04d}.pth'.format(epoch)
                torch.save(checkpoint, os.path.join(FLAGS.checkpoint_path, checkpoint_name))

            # if avg_test_loss < min_loss:
            #     min_loss = avg_test_loss
            #     print("Saving best model with testing loss {:.3f}".format(min_loss))
            #     checkpoint_name = 'ckpt_best.pth'
            #     torch.save(checkpoint, os.path.join(FLAGS.checkpoint_path, checkpoint_name))
                
            if avg_test_acc > max_acc:
                max_acc = avg_test_acc
                print("Saving best model with testing accuracy {:.3f}".format(max_acc))
                checkpoint_name = 'ckpt_best.pth'
                torch.save(checkpoint, os.path.join(FLAGS.checkpoint_path, checkpoint_name))

        # if epoch%10 == 0:
            # plt.plot(list(range(1,len(total_train_loss)+1)), total_train_loss)
            # plt.plot(list(range(1,len(total_train_loss)+1)), total_test_loss)
            # plt.legend(['train', 'test'])
            # plt.xlabel('epoch')
            # plt.ylabel('loss')
            # plt.title('Losses')
            # plt.savefig(os.path.join(FLAGS.checkpoint_path, 'training_loss.png'))
            # # plt.show()
        if epoch%10 == 0:
            _, ax = plt.subplots()
            ax.plot(list(range(1,len(total_train_loss)+1)), total_train_loss, 'C1', label='train')
            ax.plot(list(range(1,len(total_train_loss)+1)), total_test_loss, 'C2', label='test')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('Losses')
            ax.legend()
            plt.savefig(os.path.join(FLAGS.checkpoint_path, f'{experiment}_loss.png'))

            _, ax = plt.subplots()
            ax.plot(list(range(1,len(total_test_acc)+1)), total_test_acc, 'C1', label='testing accuracy')
            ax.set_xlabel('epoch')
            ax.set_ylabel('accuracy')
            ax.set_title('Testing Accuracy')
            ax.legend()
            plt.savefig(os.path.join(FLAGS.checkpoint_path, f'{experiment}_accuracy.png'))
        print(60*"=")    

if __name__ == "__main__":
    FLAGS, _ = parser.parse_known_args()
    main()