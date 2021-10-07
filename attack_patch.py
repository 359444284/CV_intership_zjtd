import argparse
import sys
from glob import glob
from shutil import copyfile
from datetime import datetime

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from data_tools import gtsrb_dataset
from torchvision.models import Inception3, resnet50, vgg16
from efficientnet_pytorch import EfficientNet

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from models.GAN.discriminator import Discriminator
from models.GAN.generator import Generator
from models.GAN.generator_unet import Generator_UNet

from gan_train import *
from gan_validation import *

torch.cuda.current_device()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--gantype', default='wgangp', help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
parser.add_argument('--model_name', type=str, default='SinGAN', help='model name')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--patch_size_max', default=40, type=int, help='Input image size')
parser.add_argument('--patch_size_min', default=4, type=int, help='Input image size')
parser.add_argument('--img_to_use', default=-999, type=int, help='Index of the input image to use < 6287')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')

torch.cuda.current_device()

if __name__ == '__main__':
    args = parser.parse_args()

    args.gpu = 0
    print("Use GPU: {} for training".format(args.gpu))

    args.input_size = (224, 224)

    train_data_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        # transforms.CenterCrop(max(input_size)),
        # transforms.ColorJitter(brightness=0.8, contrast=0.5, hue=0.5),
        # transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    trainset = gtsrb_dataset.GTSRB(
        root_dir='./images', train=True, transform=train_data_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # vgg16
    vgg16_pre = vgg16(pretrained=True)
    vgg16_pre.num_classes = 43
    vgg16_pre.load_state_dict(torch.load('./pre_trained_models/vgg16_fconly_0.9349.bin'))

    # inception_v3
    inception_v3_pre = Inception3(init_weights=True)
    inception_v3_pre.aux_logits = False
    inception_v3_pre.fc = nn.Linear(2048, 43)  # where args.num_classes = 43
    inception_v3_pre.load_state_dict(torch.load('./pre_trained_models/v3_0.9751.bin'))

    model = EfficientNet.from_name('efficientnet-b6', num_classes=43)
    model.load_state_dict(torch.load('./pre_trained_models/eff_0.9744.bin'))


    # resnet50
    resnet = resnet50(pretrained=True)
    resnet.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, 43)
    )
    resnet.load_state_dict(torch.load('./pre_trained_models/rs50_0.9675.bin'))

    # target_layers = model.features[-1]
    target_layers = resnet.layer4[-1]
    cam = GradCAM(model=resnet, target_layer=target_layers, use_cuda=True)

    train_data = next(iter(train_loader))
    input_tensor, input_label = train_data
    target_category = input_label
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    fil = np.ones([40 ,40])
    res = cv2.filter2D(grayscale_cam, -1, fil)
    # avoid out of boundary
    victim_point = np.where(res == np.max(res[20:args.input_size[0]-20,20:args.input_size[1]-20]))
    victim_point = (victim_point[0][0], victim_point[1][0])

    # show mask in img
    # input_tensor[:,:,victim_point[0]-20:victim_point[0]+20,victim_point[1]-20:victim_point[1]+20] = 0
    # array1 = input_tensor[0]  # 将tensor数据转为numpy数据
    #
    # array1[0] = array1[0] * 0.5 + 0.5
    # array1[1] = array1[1] * 0.5 + 0.5
    # array1[2] = array1[2].mul(0.5) + 0.5
    # img = array1.mul(255).byte()
    # img = img.numpy().transpose((1, 2, 0))
    #
    # print('mat_shape:', img.shape)
    # visualization = show_cam_on_image(input_tensor, grayscale_cam)
    # cv2.imshow('image', img)
    # cv2.imshow('image1', visualization)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    makedirs(os.path.join(args.log_dir, 'codes'))
    makedirs(os.path.join(args.log_dir, 'codes', 'models'))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        modelfiles = glob('./models/*.py')
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)
        for py in modelfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    formatted_print('Total Number of Workers:', args.workers)
    formatted_print('Batch Size:', args.batch_size)
    formatted_print('Max image Size:', args.patch_size_max)
    formatted_print('Min image Size:', args.patch_size_min)
    formatted_print('Log DIR:', args.log_dir)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('GAN TYPE:', args.gantype)

    ###############
    # main worker 
    ###############

    scale_factor = 4/3
    tmp_scale = args.patch_size_max / args.patch_size_min
    args.num_scale = int(np.round(np.log(tmp_scale) / np.log(scale_factor)))

    img_size = round(max(args.input_size)/tmp_scale)
    args.patch_size_list = [round(args.patch_size_min * scale_factor ** i) for i in range(args.num_scale + 1)]
    args.img_size_list = [round(img_size * scale_factor ** i) for i in range(args.num_scale + 1)]
    args.victim_point_list = [(round(victim_point[0]*size/args.input_size[0]), round(victim_point[0]*size/args.input_size[1])
                               )for size in args.img_size_list]
    print(args.patch_size_list)
    print(args.img_size_list)
    print(args.victim_point_list)

    discriminator = Discriminator()
    # generator = Generator(args.patch_size_min, args.num_scale, scale_factor)
    generator = Generator_UNet(args.patch_size_min, args.num_scale, scale_factor)

    discriminator.cuda(args.gpu)
    generator.cuda(args.gpu)
    model.cuda(args.gpu)
    networks = [discriminator, generator, model]

    resnet.cuda(args.gpu)
    vgg16_pre.cuda(args.gpu)
    inception_v3_pre.cuda(args.gpu)
    blackbox = [resnet, vgg16_pre, inception_v3_pre]

    # d_opt = torch.optim.Adam(discriminator.sub_discriminators[0].parameters(), 5e-4, (0.5, 0.999))
    # g_opt = torch.optim.Adam(generator.sub_generators[0].parameters(), 5e-4, (0.5, 0.999))
    d_opt = torch.optim.AdamW(discriminator.sub_discriminators[0].parameters(), 5e-4, (0.5, 0.999))
    g_opt = torch.optim.AdamW(generator.sub_generators[0].parameters(), 5e-4, (0.5, 0.999))

    ##############
    # Load model #
    ##############
    args.stage = 0

    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            for _ in range(int(checkpoint['stage'])):
                generator.progress()
                discriminator.progress()
            networks = [discriminator, generator]
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                networks = [x.cuda(args.gpu) for x in networks]
            else:
                networks = [torch.nn.DataParallel(x).cuda() for x in networks]

            discriminator, generator, = networks

            args.stage = checkpoint['stage']
            args.img_to_use = checkpoint['img_to_use']
            discriminator.load_state_dict(checkpoint['D_state_dict'])
            generator.load_state_dict(checkpoint['G_state_dict'])
            d_opt.load_state_dict(checkpoint['d_optimizer'])
            g_opt.load_state_dict(checkpoint['g_optimizer'])
            print("=> loaded checkpoint '{}' (stage {})"
                  .format(load_file, checkpoint['stage']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))

    cudnn.benchmark = True

    ######################
    # Validate and Train #
    ######################

    if args.validation:
        validateSinGAN(train_loader, networks, args.stage, args)
        sys.exit(0)

    elif args.test:
        validateSinGAN(train_loader, networks, args.stage, args)
        sys.exit(0)
    
    check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
    record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
    record_txt.write('GANTYPE\t:\t{}\n'.format(args.gantype))
    record_txt.write('IMGTOUSE\t:\t{}\n'.format(args.img_to_use))
    record_txt.close()

    for stage in range(args.stage, args.num_scale + 1):

        trainSinGAN(train_data, networks, {"d_opt": d_opt, "g_opt": g_opt}, stage, args)
        validateSinGAN(train_data, networks, blackbox, stage, args)

        discriminator.progress()
        generator.progress()

        networks = [discriminator, generator]

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            networks = [x.cuda(args.gpu) for x in networks]
        else:
            networks = [torch.nn.DataParallel(x).cuda() for x in networks]

        networks.append(model)
        discriminator, generator, model = networks

        # Update the networks at finest scale

        for net_idx in range(generator.current_scale):
            for param in generator.sub_generators[net_idx].parameters():
                param.requires_grad = False
            for param in discriminator.sub_discriminators[net_idx].parameters():
                param.requires_grad = False

        # d_opt = torch.optim.Adam(discriminator.sub_discriminators[discriminator.current_scale].parameters(),
        #                          5e-4, (0.5, 0.999))
        # g_opt = torch.optim.Adam(generator.sub_generators[generator.current_scale].parameters(),
        #                          5e-4, (0.5, 0.999))
        d_opt = torch.optim.AdamW(discriminator.sub_discriminators[discriminator.current_scale].parameters(),
                                 5e-4, (0.5, 0.999))
        g_opt = torch.optim.AdamW(generator.sub_generators[generator.current_scale].parameters(),
                                 5e-4, (0.5, 0.999))

        ##############
        # Save model #
        ##############
        if stage == 0:
            check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
        save_checkpoint({
            'stage': stage + 1,
            'D_state_dict': discriminator.state_dict(),
            'G_state_dict': generator.state_dict(),
            'd_optimizer': d_opt.state_dict(),
            'g_optimizer': g_opt.state_dict(),
            'img_to_use': args.img_to_use
        }, check_list, args.log_dir, stage + 1)
        if stage == args.num_scale:
            check_list.close()

