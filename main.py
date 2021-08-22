import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import train
from models import inception_ResNetv2

# torch.cuda.current_device()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

if __name__ == '__main__':

    # traindir = './images/ILSVRC2012_img_val'
    valdir = './images/ILSVRC2012_img_val'

    batch_size = 32
    input_size = (299, 299, 3)
    scale = 0.875

    data_transform = transforms.Compose([
        transforms.Resize(int((math.floor(max(input_size) / scale)))),
        transforms.CenterCrop(max(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    val_dataset = ImageFolder(valdir, transform=data_transform)

    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

    # data = next(iter(val_data_loader))
    # print(data)

    model = inception_ResNetv2.Inception_ResNetv2()

    model.load_state_dict(torch.load('./pre_trained_models/inceptionresnetv2.pth'))
    #
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    #
    model.to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)
    #
    # train.train_epoch(model, optimizer, scheduler, device, loss_fn, train_data_loader, val_data_loader, 30)

    model.eval()
    tp_1, tp_5 = 0, 0
    for i, data in enumerate(val_data_loader):
        input, label = data
        input, label = input.to(device), label.to(device)
        pred = model(input)
        _, pred = torch.topk(pred, 5, dim=1)
        correct = pred.eq(label.view(-1, 1).expand_as(pred)).cpu().numpy()
        tp_1 += correct[:, 0].sum()
        tp_5 += correct.sum()
        print(i, "top1: ", tp_1, "top5:", tp_5)
    print("Top1 accuracy: ", tp_1 / 50000)
    print("Top5 accuracy: ", tp_5 / 50000)
