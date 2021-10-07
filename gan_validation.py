from tqdm import trange
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils

from utils import *

def pad_transform(patch, image_size, patch_size, x, y, device):
    offset_x = x
    offset_y = y

    pad = nn.ConstantPad2d((offset_x - patch_size // 2, image_size - patch_size - offset_x + patch_size // 2,
                            offset_y - patch_size // 2, image_size - patch_size - offset_y + patch_size // 2),
                           0)  # left, right, top ,bottom
    mask = torch.ones((3, patch_size, patch_size))
    mask = mask.cuda(device)
    return pad(patch), pad(mask)

def validateSinGAN(data, networks, blackbox, stage, args, additional=None):
    # set nets
    D = networks[0]
    G = networks[1]
    model = networks[2]

    res = blackbox[0]
    vgg = blackbox[1]
    incep = blackbox[2]

    # switch to train mode
    D.eval()
    G.eval()
    model.eval()

    res.eval()
    vgg.eval()
    incep.eval()
    # summary writer
    # writer = additional[0]

    x_in, label = data
    x_in = x_in.cuda(args.gpu)
    label = label.cuda(args.gpu, non_blocking=False)

    x_org = x_in

    x_in = F.interpolate(x_in, (args.img_size_list[stage], args.img_size_list[stage]), mode='bilinear', align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORG_{}.png'.format(stage)),
                      nrow=1, normalize=True)
    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (args.img_size_list[xidx], args.img_size_list[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    patch_size = args.patch_size_list[stage]
    img_size = args.img_size_list[stage]
    x, y = args.victim_point_list[stage]

    with torch.no_grad():

        correct_rate_list = [0.0,0.0,0.0,0.0]
        att_rate_list = [0.0,0.0,0.0,0.0]
        count = 100
        for k in range(count):

            z_list = [torch.randn(args.batch_size, 3, args.patch_size_list[z_idx],
                                  args.patch_size_list[z_idx]).cuda(args.gpu, non_blocking=True) for z_idx in
                      range(stage + 1)]
            x_fake_list = G(z_list)

            patch, mask = pad_transform(x_fake_list[-1], img_size, patch_size, x, y, args.gpu)
            pad_patch = patch
            victimized_x_in = torch.mul((1 - mask), x_in) + torch.mul(mask, patch)

            org_img = x_in.detach()
            max_v = torch.max(org_img)
            min_v = torch.min(org_img)
            org_img = (org_img - min_v)/(max_v - min_v)

            patch_img = x_fake_list[-1].detach()
            max_v = torch.max(patch_img)
            min_v = torch.min(patch_img)
            patch_img = (patch_img - min_v) / (max_v - min_v)

            patch, mask = pad_transform(patch_img, img_size, patch_size, x, y, args.gpu)
            victimized_real = torch.mul((1 - mask), org_img) + torch.mul(mask, patch)


            final_patch_size = args.patch_size_list[-1]
            up_patch = F.interpolate(x_fake_list[-1], (final_patch_size, final_patch_size), mode='bilinear',
                                  align_corners=True)
            patch, mask = pad_transform(up_patch, max(x_org.size()), final_patch_size, x, y, args.gpu)
            victimized_img = torch.mul((1 - mask), x_org) + torch.mul(mask, patch)

            att_output = model(victimized_img)
            clean_output = model(x_org)

            vgg_clean_output = vgg(x_org)
            vgg_att_output = vgg(victimized_img)

            res_clean_output = res(x_org)
            res_att_output = res(victimized_img)

            v3_clean_input = F.interpolate(x_org, (299, 299), mode='bilinear', align_corners=True)
            v3_att_input = F.interpolate(victimized_img, (299, 299), mode='bilinear', align_corners=True)
            v3_clean_output = incep(v3_clean_input)
            v3_att_output = incep(v3_att_input)

            _, att_preds = torch.max(att_output, dim=1)
            _, clean_preds = torch.max(clean_output, dim=1)
            correct_rate_list[0] += torch.sum(clean_preds == label).cpu().numpy()
            att_rate_list[0] += torch.sum(att_preds == label).cpu().numpy()

            _, res_att_preds = torch.max(res_att_output, dim=1)
            _, res_clean_preds = torch.max(res_clean_output, dim=1)
            correct_rate_list[1] += torch.sum(res_clean_preds == label).cpu().numpy()
            att_rate_list[1] += torch.sum(res_att_preds == label).cpu().numpy()

            _, vgg_att_preds = torch.max(vgg_att_output, dim=1)
            _, vgg_clean_preds = torch.max(vgg_clean_output, dim=1)
            correct_rate_list[2] += torch.sum(vgg_clean_preds == label).cpu().numpy()
            att_rate_list[2] += torch.sum(vgg_att_preds == label).cpu().numpy()

            _, v3_att_preds = torch.max(v3_att_output, dim=1)
            _, v3_clean_preds = torch.max(v3_clean_output, dim=1)
            correct_rate_list[3] += torch.sum(v3_clean_preds == label).cpu().numpy()
            att_rate_list[3] += torch.sum(v3_att_preds == label).cpu().numpy()


            if k % 30 == 0:
                vutils.save_image(x_fake_list[-1].detach().cpu(),
                                  os.path.join(args.res_dir, 'GEN_patch_{}_{}.png'.format(stage, k)),
                                  nrow=1, normalize=True)
                vutils.save_image(victimized_real.cpu(),
                                  os.path.join(args.res_dir, 'GEN_normal_{}_{}.png'.format(stage, k)),
                                  nrow=1, normalize=False)

        for i in range(len(correct_rate_list)):
            correct_rate_list[i] /= count
            att_rate_list[i] /= count
            att_rate_list[i] = (correct_rate_list[i] - att_rate_list[i])/correct_rate_list[i]*100

        print('correct_rate', correct_rate_list[0])
        print('att_rate', att_rate_list[0])
        print('res_correct_rate', correct_rate_list[1])
        print('res_att_rate', att_rate_list[1])
        print('vgg_correct_rate', correct_rate_list[2])
        print('vgg_att_rate', att_rate_list[2])
        print('v3_correct_rate', correct_rate_list[3])
        print('v3_att_rate', att_rate_list[3])

