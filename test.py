import os
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data

import dataset
import models


# Define argument parser function for fine-tuning.

def test_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', default='GestureNet', type=str)
    parser.add_argument('--model', default='pred_model', type=str)
    parser.add_argument('--dataset', default='HMOG_VER', type=str)
    parser.add_argument('--train_type', default='finetune', type=str)
    parser.add_argument('--train_phase', default=True, type=bool)
    parser.add_argument('--supervision', default=False, type=bool)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--input_planes', default=9, type=int, help='number of input channels from data')
    parser.add_argument('--window', default=32, type=int, help='the signal length selected from multiple timesteps')
    parser.add_argument('--epoch', default=50, type=int, help='the number of total epochs')
    parser.add_argument('--num_classes', default=10, type=int, help='the number of output classes')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate value')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum value for SGD optimizer')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay value for SGD optimizer')

    arguments = parser.parse_args()
    return arguments


args = test_parser()

# set seed for reproducibility:

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If test is Fully supervised run the experimentation with ground-truth labels

if args.supervision:
    print("Perform user Identification or verification with fully supervised training and testing")
    # Model

    if 'MobileNetV1' in args.net:

        print("Supervised training and testing on user Identification... model -->", args.net)

        model = models.MobileNetV1(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes,
                                   layers=[2, 2, 6, 1])

    elif 'MobileNetV2' in args.net:

        print("Supervised training and testing on user Identification... model -->", args.net)

        model = models.MobileNetV2(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes)

    elif 'EfficientNetB0' in args.net:

        print("Supervised training and testing on user Identification... model -->", args.net)

        model = models.EfficientNet(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes)

    else:
        print("Supervised training and testing on user Identification... model -->", args.net)
        model = models.GestureNet(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes)
    # Dataset

    if 'HMOG_ID' in args.dataset:
        print("Self-Supervised User Identification on: ", args.dataset)

        train_set = dataset.HMOG_ID(args.train_phase)

        val_set = dataset.HMOG_ID(args.train_type, train_phase=False)

    elif 'TOUCH_ID' in args.dataset:

        print("Self-Supervised User Identification on: ", args.dataset)

        train_set = dataset.TOUCH_ID(args.train_type, args.train_phase)

        val_set = dataset.TOUCH_ID(args.train_type, train_phase=False)

    elif 'TOUCH_VER' in args.dataset:

        print("Self-Supervised User Verification on: ", args.dataset)

        train_set = dataset.TOUCH_VER(args.train_phase)

        val_set = dataset.TOUCH_VER(train_phase=False)

    else:
        print("Self-Supervised User Verification on: ", args.dataset)
        train_set = dataset.HMOG_VER(args.train_phase)

        val_set = dataset.HMOG_VER(train_phase=False)

    model.to(device)

    # Optimizer, Loss, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180)
    # Load the data

    # Instantiating Batch sampler for loading data in form of slices
    train_sampler = data.BatchSampler(data.SequentialSampler(train_set), args.window, True)
    val_sampler = data.BatchSampler(data.SequentialSampler(val_set), args.window, True)

    # Instantiating Data Loader with a specific sequential sampler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler)

    train_history = {'training_loss': [], 'training_accuracy': []}
    test_history = {'validation_loss': [], 'validation_accuracy': []}

    print("Train on user Identification for 180 Epochs")

    # Implementing the main loop

    for epoch in range(180):

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            labels = y_train[:, 0]
            p_train = torch.randperm(args.batch_size)
            train_samples = x_train[p_train]
            train_labels = labels[p_train]

            train_samples, train_labels = train_samples.to(device), train_labels.to(device)
            train_samples = train_samples.float()

            optimizer.zero_grad()

            outputs = model(train_samples)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predictions = outputs.max(1)
            total += train_labels.size(0)
            correct += predictions.eq(train_labels).sum().item()

        train_history['training_loss'].append(train_loss / total)
        train_history['training_accuracy'].append(100. * correct / total)

        print(
            "Epoch: ", epoch,
            "Loss: ", train_loss,
            "Accuracy: ", 100. * correct / total)
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x_test, y_test) in enumerate(val_loader):
                test_labels = y_test[:, 0]

                test_samples, test_labels = x_test.to(device), test_labels.to(device)
                test_samples = test_samples.float()
                outputs = model(test_samples)

                loss = criterion(outputs, test_labels)
                test_loss += loss.item()
                _, predictions = outputs.max(1)
                total += test_labels.size(0)
                correct += predictions.eq(test_labels).sum().item()

            test_history['validation_loss'].append(test_loss / total)
            test_history['validation_accuracy'].append(100. * correct / total)

        scheduler.step()

    avg_accuracy = torch.mean(torch.tensor(test_history['validation_accuracy']))

    print("The accuracy on", args.dataset, "for supervised user verification with: ", args.net, "is: ", avg_accuracy)

else:
    path = 'pretrained_models/'
    # Select pre-trained Model

    if 'MobileNetV1' in args.net:

        model = models.MobileNetV1(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes,
                                   layers=[2, 2, 6, 1])
        contrastive_model = models.pred_model.Pred_Model(model, 1024, 128)
        file_name = str(args.net) + '.pth'
        path_folder = os.path.join(path, file_name)
        contrastive_model.load_state_dict(torch.load(path_folder), strict=False)
        ft_model = models.downstream_model.DsModel(contrastive_model, num_classes=2)

    elif 'MobileNetV2' in args.net:
        model = models.MobileNetV2(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes)
        contrastive_model = models.pred_model.Pred_Model(model, 1024, 128)
        file_name = str(args.net) + '.pth'
        path_folder = os.path.join(path, file_name)
        contrastive_model.load_state_dict(torch.load(path_folder), strict=False)
        ft_model = models.downstream_model.DsModel(contrastive_model, num_classes=2)

    elif 'EfficientNetB0' in args.net:
        model = models.EfficientNet(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes)
        contrastive_model = models.pred_model.Pred_Model(model, 1024, 128)
        file_name = str(args.net) + '.pth'
        path_folder = os.path.join(path, file_name)
        contrastive_model.load_state_dict(torch.load(path_folder), strict=False)
        ft_model = models.downstream_model.DsModel(contrastive_model, num_classes=2)
    else:
        model = models.GestureNet(args.batch_size, args.input_planes, args.window, num_classes=args.num_classes)
        contrastive_model = models.pred_model.Pred_Model(model, 1024, 128)
        file_name = str(args.net) + '.pth'
        path_folder = os.path.join(path, file_name)
        contrastive_model.load_state_dict(torch.load(path_folder), strict=False)
        ft_model = models.downstream_model.DsModel(contrastive_model, num_classes=2)

    ft_model.to(device)
    # Load the data

    if 'HMOG_ID' in args.dataset:
        print("Self-Supervised User Identification on: ", args.dataset)

        train_set = dataset.HMOG_ID(args.train_phase)

        val_set = dataset.HMOG_ID(args.train_type, train_phase=False)

    elif 'TOUCH_ID' in args.dataset:

        print("Self-Supervised User Identification on: ", args.dataset)

        train_set = dataset.TOUCH_ID(args.train_type, args.train_phase)

        val_set = dataset.TOUCH_ID(args.train_type, train_phase=False)

    elif 'TOUCH_VER' in args.dataset:

        print("Self-Supervised User Verification on: ", args.dataset)

        train_set = dataset.TOUCH_VER(args.train_phase)

        val_set = dataset.TOUCH_VER(train_phase=False)

    else:
        print("Self-Supervised User Verification on: ", args.dataset)
        train_set = dataset.HMOG_VER(args.train_phase)

        val_set = dataset.HMOG_VER(train_phase=False)

    # Optimizer, Loss, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in ft_model.parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum,
                          weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    # Load the data

    # Instantiating Batch sampler for loading data in form of slices
    train_sampler = data.BatchSampler(data.SequentialSampler(train_set), args.window, True)
    val_sampler = data.BatchSampler(data.SequentialSampler(val_set), args.window, True)

    # Instantiating Data Loader with a specific sequential sampler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler)

    train_history = {'training_loss': [], 'training_accuracy': []}
    test_history = {'validation_loss': [], 'validation_accuracy': []}

    print("Train on user verification for", args.epochs)

    # Implementing the main loop for downstream evaluation #

    for epoch in range(args.epoch):

        ft_model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            labels = y_train[:, 0]
            p_train = torch.randperm(args.batch_size)
            train_samples = x_train[p_train]
            train_labels = labels[p_train]

            train_samples, train_labels = train_samples.to(device), train_labels.to(device)
            train_samples = train_samples.float()

            optimizer.zero_grad()

            outputs = ft_model(train_samples)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predictions = outputs.max(1)
            total += train_labels.size(0)
            correct += predictions.eq(train_labels).sum().item()

        train_history['training_loss'].append(train_loss / total)
        train_history['training_accuracy'].append(100. * correct / total)

        print(
            "Epoch: ", epoch,
            "Loss: ", train_loss,
            "Accuracy: ", 100. * correct / total)
        ft_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (x_test, y_test) in enumerate(val_loader):
                test_labels = y_test[:, 0]

                test_samples, test_labels = x_test.to(device), test_labels.to(device)
                test_samples = test_samples.float()
                outputs = ft_model(test_samples)

                loss = criterion(outputs, test_labels)
                test_loss += loss.item()
                _, predictions = outputs.max(1)
                total += test_labels.size(0)
                correct += predictions.eq(test_labels).sum().item()

            test_history['validation_loss'].append(test_loss / total)
            test_history['validation_accuracy'].append(100. * correct / total)

        scheduler.step()

avg_accuracy = torch.mean(torch.tensor(test_history['validation_accuracy']))

print("The accuracy on", args.dataset, "for user verification with: ", args.net, "is: ", avg_accuracy)
