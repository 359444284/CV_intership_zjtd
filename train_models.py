import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data_tools import gtsrb_dataset
from torchvision.models import inception_v3, Inception3, resnet50, vgg16
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

import train
from models import inception_ResNetv2

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

NUM_CLASSES = 43
BATCH_SIZE = 4

if __name__ == '__main__':

    # traindir = './images/ILSVRC2012_img_val'
    # valdir = './images/ILSVRC2012_img_val'
    # inception_v3 299, restnet50 224 vgg16 224
    input_size = (224, 224)

    train_data_transform = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(max(input_size)),
        transforms.ColorJitter(brightness=0.8, contrast=0.5, hue=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    test_data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(max(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    trainset = gtsrb_dataset.GTSRB(
        root_dir='./images', train=True, transform=train_data_transform)
    testset = gtsrb_dataset.GTSRB(
        root_dir='./images', train=False, transform=test_data_transform)


    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # data = next(iter(train_loader))
    # print(data)

    # ---------------------------------------------------------------------------------------------
    # inception_ResNetv2
    # model = inception_ResNetv2.Inception_ResNetv2()
    # model.load_state_dict(torch.load('./pre_trained_models/inceptionresnetv2.pth'))

    # inception_v3
    # model = inception_v3(pretrained=True)
    # model = Inception3()

    # model.aux_logits = False
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = nn.Linear(2048, 43)  # where args.num_classes = 43
    # model.load_state_dict(torch.load('./v3_0.9751.bin'))


    # resnet50
    # model = resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.fc = nn.Sequential(
    #     nn.Dropout(0.4),
    #     nn.Linear(2048, 43)
    # )
    # model.load_state_dict(torch.load('./pre_trained_models/rs50_0.9675.bin'))

    # vgg16
    # model = vgg16(pretrained=True)
    # model.num_classes = 43
    #
    # for param in model.features.parameters():
    #     param.requires_grad = False


    # model.load_state_dict(torch.load('./pre_trained_models/vgg16_fconly_0.9349.bin'))


    # EfficientNet
    # model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=43, weights_path='./pre_trained_models/eff_0.9744.bin')
    model = EfficientNet.from_name('efficientnet-b6', num_classes=43)
    model.load_state_dict(torch.load('./pre_trained_models/eff_0.9744.bin'))
    # ---------------------------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
    #
    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    #
    #
    # train.train_epoch(model, optimizer, scheduler, device, loss_fn, len(trainset), train_loader, len(testset), test_loader, 20)
    loss, acc = train.evaluate(model, len(testset), test_loader, device, loss_fn)
    print([loss,acc])
    # ---------------------------------------------------------------------------------------------







