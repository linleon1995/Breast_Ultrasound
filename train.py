import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from model import UNet_2d
from torch.utils.data import Dataset, DataLoader
from dataset.dataloader import ImageDataset
from dataset.preprocessing import DataPreprocessing
import numpy as np
from utils import train_utils
from cfg import dataset_config
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
EPOCH = 400
BATCH_SIZE = 6
SHUFFLE = True
PROJECT_PATH = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\"
DATAPATH = os.path.join(PROJECT_PATH, "archive\\Dataset_BUSI_with_GT")
# CHECKPOINT = os.path.join(PROJECT_PATH, 'models')
CHECKPOINT = train_utils.create_training_path(os.path.join(PROJECT_PATH, 'models'))
LEARNING_RATE = 1e-2
# TODO: decide automatically
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

# TODO: Check Dice Loss implementation
# TODO: Iprove the visualization of training process
# TODO: tensorboard
# TODO: plot image iteratively
class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
		return loss


# class Trainer(dataloader, model, epoch, loss, optimizer, eval_metrics)
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
    test_dataset = ImageDataset(dataset_config, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    # TODO: loss plot
    # TODO: training parameter to cfg.py
    # TODO: test original image
    # TODO: modify vars name

    # Start training
    training_samples = len(train_dataloader.dataset)
    step_loss, epoch_loss, total_test_loss = [], [], []
    min_loss = 1e5
    saving_steps = SAVING_STEPS
    training_steps = int(training_samples/FLAGS.batch_size)
    if training_samples%FLAGS.batch_size != 0:
        training_steps += 1
    print("Start Training!!")
    print("Training epoch: {} Batch size: {} Shuffling Data: {} Training Samples: {}".
            format(FLAGS.epoch, FLAGS.batch_size, FLAGS.shuffle, training_samples))
    print(60*"-")
    fw = open("logging.txt",'w+')
    fw.write(str(dataset_config))
    fw.write(str({'epoch': FLAGS.epoch, 'batch_size': FLAGS.batch_size, 'lr': LEARNING_RATE}))
    fw.close()

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
            if i%10 == 0:
                print('Step {}  Step loss {}'.format(i, loss))
        epoch_loss.append(total_loss/training_steps)
        print('**Epoch {}/{}  Epoch loss {}  Step loss {}'.
            format(epoch, FLAGS.epoch, epoch_loss[-1], step_loss[-1]))
        # print(50*'=')
        with torch.no_grad():
            net.eval()
            loss_list = []
            for _, data in enumerate(test_dataloader):
                inputs, labels = data['input'], data['gt']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                test_loss = DiceLoss()(outputs, labels)
                loss_list.append(test_loss)
            avg_test_loss = sum(loss_list) / len(loss_list)
            total_test_loss.append(avg_test_loss)
            print("**Testing Loss:{:.3f}".format(avg_test_loss))
            print(60*"=")
            
            # Saving model
            # TODO: logging step
            # TODO: logging docs --> txt
            # TODO: save one best model and save model in steps
            # TODO: simplize below
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

            if avg_test_loss < min_loss:
                min_loss = avg_test_loss
                print("Saving best model with testing loss {:.3f}".format(min_loss))
                print(50*'-')
                checkpoint_name = 'ckpt_best.pth'
                torch.save(checkpoint, os.path.join(FLAGS.checkpoint_path, checkpoint_name))
                
        if epoch%10 == 0:
            plt.plot(list(range(1,len(epoch_loss)+1)), epoch_loss)
            plt.plot(list(range(1,len(epoch_loss)+1)), total_test_loss)
            plt.legend(['train', 'test'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Losses')
            plt.savefig(os.path.join(FLAGS.checkpoint_path, 'training_loss.png'))
            # plt.show()

if __name__ == "__main__":
    FLAGS, _ = parser.parse_known_args()
    main()