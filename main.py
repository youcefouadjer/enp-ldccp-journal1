import argparse
import numpy as np
import random
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data

import dataset
import models
import loss


# Define argument parser function for pretraining.

def main_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', default='GestureNet', type=str)
    parser.add_argument('--model', default='pred_model', type=str)
    parser.add_argument('--dataset', default='HMOG_ID', type=str)
    parser.add_argument('--train_type', default='pretrain', type=str)
    parser.add_argument('--train_phase', default=True, type=bool)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--input_planes', default=9, type=int, help='number of input channels from data')
    parser.add_argument('--window', default=32, type=int, help='the signal length selected from multiple timesteps')
    parser.add_argument('--epoch', default=180, type=int, help='the number of total epochs')
    parser.add_argument('--num_classes', default=10, type=int, help='the number of output classes')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate value')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum value for SGD optimizer')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay value for SGD optimizer')
    parser.add_argument('--seed', default=0, type=int, help="seed for reproducibility")

    arguments = parser.parse_args()

    return arguments


args = main_parser()

# set seed for reproducibility:
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'MobileNetV1' in args.net:
    model = models.MobileNetV1(args.batch_size, args.input_planes, args.window, args.num_classes, layers=[2, 2, 6, 1])

elif 'MobileNetV2' in args.net:
    model = models.MobileNetV2(args.batch_size, args.input_planes, args.window, args.num_classes)

elif 'EfficientNetB0' in args.net:
    model = models.EfficientNet(args.batch_size, args.input_planes, args.window, args.num_classes)
else:
    model = models.GestureNet(args.batch_size, args.input_planes, args.window, args.num_classes)

contrastive_model = models.Pred_Model(model, 1024, 128).to(device)

# Optimizer, Loss, and Scheduler

optimizer = optim.SGD(contrastive_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
criterion = loss.Contrastive_Loss(batch_size=args.batch_size, temperature=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180)

# Load the data

# Train set and Test set for contrastive learning.

if 'HMOG_ID' in args.dataset:
    train_set = dataset.HMOG_ID(args.train_type, args.train_phase)
    val_set = dataset.HMOG_ID(args.train_type, train_phase=False)

else:
    train_set = dataset.TOUCH_ID(args.train_type, args.train_phase)

    val_set = dataset.TOUCH_ID(args.train_type, train_phase=False)

train_sampler = data.BatchSampler(data.SequentialSampler(train_set), args.window, True)
val_sampler = data.BatchSampler(data.SequentialSampler(val_set), args.window, True)

# Instantiating Data Loader with a specific sequential sampler

train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler)

# checkpoint saving function
default_model = args.net

if not os.path.isdir('pretrained_models'):
    os.mkdir('pretrained_models')
path = 'pretrained_models/'


def save_checkpoint(state, filename=default_model + '.pth'):
    path_folder = os.path.join(path, filename)
    print("=>> Saving check_point...")
    torch.save(state, path_folder)


# Implement the main training loop

model_history = {'training_loss': [], 'validation_loss': []}

for epoch in range(args.epoch):
    contrastive_model.train()
    train_loss = 0
    total = 0
    # contrastive prediction on training data
    if epoch == 179:
        state_epoch = {'state_dict': contrastive_model.state_dict()}
        save_checkpoint(state_epoch)

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        z_i = contrastive_model(x_i)
        z_j = contrastive_model(x_j)

        loss = criterion(z_i, z_j)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        total += x_i.size(0)

    model_history['training_loss'].append(train_loss / total)

    print("Epoch: ", epoch,
          "Loss: ", train_loss)

    # Contrastive prediction on test data

    contrastive_model.eval()
    test_loss = 0
    total = 0

    with torch.no_grad():
        for step, (x_i, x_j) in enumerate(val_loader):
            x_i = x_i.squeeze().to(device).float()
            x_j = x_j.squeeze().to(device).float()

            z_i = contrastive_model(x_i)
            z_j = contrastive_model(x_j)

            loss = criterion(z_i, z_j)

            test_loss += loss.item()
            total += x_i.size(0)

        model_history['validation_loss'].append(test_loss / total)

    scheduler.step()

    if epoch == 179:
        print("Pretraining is done ! please go to fine-tuning....")
