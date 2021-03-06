# Classification task on Breast Ultrasound
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from cfg import dataset_config
from dataset.dataloader import ClassificationImageDataset
# from model import ImageClassifier
from core.image_calssification import img_classifier
from utils import train_utils
from dataset import dataset_utils
from utils import configuration
from utils import metrics
ImageClassifier = img_classifier.ImageClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_2dunet_cls_train_config.yml'

logger = train_utils.get_logger('TrainingSetup')


def main():
    # Configuration
    config = configuration.load_config(CONFIG_PATH)
    logger.info(config)
    
    # Select device

    # Load and log experiment configuration
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = train_utils.create_training_path(os.path.join(config.train.project_path, 'models'))
    # TODO: selective pretrained
    # TODO: dynamic output structure
    net = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=config.model.pretrained, dim=1, output_structure=None)
    if torch.cuda.is_available():
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.train.learning_rate)
    # print(net)
    # Logger
    
    # Dataloader
    train_dataset = ClassificationImageDataset(config, mode='train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.dataset.train.batch_size, shuffle=config.dataset.train.shuffle, pin_memory=True)
    test_dataset = ClassificationImageDataset(config, mode='test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.dataset.val.batch_size, shuffle=config.dataset.val.shuffle, pin_memory=True)


    # Start training
    training_samples = len(train_dataloader.dataset)
    step_loss, total_train_loss= [], []
    total_test_acc, total_test_loss  = [], []
    min_loss = 1e5
    max_acc = -1
    saving_steps = config.train.checkpoint_saving_steps
    training_steps = int(training_samples/config.dataset.train.batch_size)
    if training_samples%config.dataset.train.batch_size != 0:
        training_steps += 1
    testing_steps = len(test_dataloader.dataset)
    experiment = [s for s in checkpoint_path.split('\\') if 'run' in s][0]
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
            format(config.train.epoch, config.dataset.train.batch_size, config.dataset.train.shuffle, training_samples))
    print(60*"-")
    
    train_utils._logging(os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')
    # TODO: train_logging
    config['experiment'] = experiment
    train_utils.train_logging(os.path.join(config.train.project_path, 'models', 'train_logging.txt'), config)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(1, config.train.epoch+1):
        total_loss = 0.0
        for i, data in enumerate(train_dataloader):
            net.train()
            # print('Epoch: {}/{}  Step: {}'.format(epoch, config.train.epoch, i))
            inputs, labels = data['input'], data['gt']
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = F.one_hot(labels, num_classes=config.model.out_channels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_loss.append(loss)
            
            if i%level == 0:
                print('Step {}  Step loss {}'.format(i, loss))
        total_train_loss.append(total_loss/training_steps)
        # TODO: check Epoch loss correctness
        print(f'**Epoch {epoch}/{config.train.epoch}  Training Loss {total_train_loss[-1]}')
        with torch.no_grad():
            net.eval()
            # loss_list = []
            test_loss, test_acc = 0.0, []
            eval_tool = metrics.SegmentationMetrics(config.model.out_channels, ['accuracy'])
            for _, data in enumerate(test_dataloader):
                inputs, labels = data['input'], data['gt']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                test_loss += loss_func(outputs, labels)
                prediction = torch.argmax(outputs, dim=1)

                labels = labels.cpu().detach().numpy()
                prediction = prediction.cpu().detach().numpy()
                evals = eval_tool(labels, prediction)
            avg_test_acc = metrics.accuracy(
                np.sum(eval_tool.total_tp), np.sum(eval_tool.total_fp), np.sum(eval_tool.total_fn))
            total_test_acc.append(avg_test_acc)
            avg_test_loss = test_loss / testing_steps
            total_test_loss.append(avg_test_loss)
            print("**Testing Loss:{:.3f}".format(avg_test_loss))


            #     total_tp += tp
            #     total_fp += fp

            #     accuracy = metrics.accuracy(np.sum(evaluator.total_tp), np.sum(evaluator.total_fp), np.sum(evaluator.total_fn))
            #     tp, fp, fn = eval_tool.tp, eval_tool.fp, eval_tool.fn
            #     if (tp + fp + fn) != 0:
            #         test_acc.append(evals['accuracy'])
            #     else:
            #         test_acc.append(0)
            # avg_test_acc = sum(test_acc) / len(test_acc)
            # avg_test_loss = test_loss / testing_steps
            # total_test_loss.append(avg_test_loss)
            # total_test_acc.append(avg_test_acc)
            # print("**Testing Loss:{:.3f}".format(avg_test_loss))
            
            
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch
                }
                
            if epoch%saving_steps == 0:
                print("Saving model with testing accuracy {:.3f} in epoch {} ".format(avg_test_loss, epoch))
                checkpoint_name = 'ckpt_best_{:04d}.pth'.format(epoch)
                torch.save(checkpoint, os.path.join(checkpoint_path, checkpoint_name))

            if avg_test_acc > max_acc:
                max_acc = avg_test_acc
                print("Saving best model with testing accuracy {:.3f}".format(max_acc))
                checkpoint_name = 'ckpt_best.pth'
                pp = os.path.join(checkpoint_path, checkpoint_name)
                print(pp)
                torch.save(checkpoint, pp)

        if epoch%10 == 0:
            _, ax = plt.subplots()
            ax.plot(list(range(1,len(total_train_loss)+1)), total_train_loss, 'C1', label='train')
            ax.plot(list(range(1,len(total_train_loss)+1)), total_test_loss, 'C2', label='test')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('Losses')
            ax.legend()
            ax.grid()
            plt.savefig(os.path.join(checkpoint_path, f'{experiment}_loss.png'))

            _, ax = plt.subplots()
            ax.plot(list(range(1,len(total_test_acc)+1)), total_test_acc, 'C1', label='testing accuracy')
            ax.set_xlabel('epoch')
            ax.set_ylabel('accuracy')
            ax.set_title('Testing Accuracy')
            ax.legend()
            ax.grid()
            plt.savefig(os.path.join(checkpoint_path, f'{experiment}_accuracy.png'))
        print(60*"=")    



    # # create trainer
    # default_trainer_builder_class = 'UNetTrainerBuilder'
    # trainer_builder_class = config['trainer'].get('builder', default_trainer_builder_class)
    # trainer_builder = dataset_utils.get_class(trainer_builder_class, modules=['utils.trainer'])
    # trainer = trainer_builder.build(config)
    # # Start training
    # trainer.fit()


if __name__ == '__main__':
    main()