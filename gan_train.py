from tqdm import trange
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import *
from ops import compute_grad_gp_wgan, compute_grad_gp
import torchvision.utils as vutils


def print_grad(grad):
    print(grad.shape)
    print(grad)


def adv_loss_fun(outputs, labels, targeted=False):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(0)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())

    if targeted:
        return torch.clamp((i-j), min=0)
    else:
        return torch.clamp((j-i), min=0)

    # or maximize cross_entropy loss
    # loss_adv = -F.mse_loss(logits_model, onehot_labels)
    # loss_adv = - F.cross_entropy(logits_model, labels)


def pad_transform(patch, image_size, patch_size, x, y, device):
    offset_x = x
    offset_y = y

    pad = nn.ConstantPad2d((offset_x - patch_size // 2, image_size - patch_size - offset_x + patch_size // 2,
                            offset_y - patch_size // 2, image_size - patch_size - offset_y + patch_size // 2),
                           0)  # left, right, top ,bottom
    mask = torch.ones((3, patch_size, patch_size))
    mask = mask.cuda(device)
    return pad(patch), pad(mask)

def trainSinGAN(data, networks, opts, stage, args, additional=None):
    # avg meter
    d_losses = AverageMeter()
    g_losses = AverageMeter()

    # set nets
    D = networks[0]
    G = networks[1]
    model = networks[2]
    # set opts
    d_opt = opts['d_opt']
    g_opt = opts['g_opt']
    # switch to train mode
    D.train()
    G.train()
    model.eval()
    # summary writer
    # writer = additional[0]

    total_iter = int(680 * pow(1.04, args.num_scale - stage + 1.0))
    # decay_lr = int(750 * pow(1.06, args.num_scale - stage + 1.0))
    # total_iter = 650
    # decay_lr =400

    d_iter = 3
    g_iter = 3

    t_train = trange(0, total_iter, initial=0, total=total_iter)

    x_in, label = data

    x_in = x_in.cuda(args.gpu, non_blocking=True)

    label = label.cuda(args.gpu, non_blocking=True)

    x_org = x_in
    x_in = F.interpolate(x_in, (args.img_size_list[stage], args.img_size_list[stage]), mode='bilinear',
                         align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORGTRAIN_{}.png'.format(stage)),
                      nrow=1, normalize=True)

    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (args.img_size_list[xidx], args.img_size_list[xidx]), mode='bilinear',
                              align_corners=True)
        x_in_list.append(x_tmp)

    patch_size = args.patch_size_list[stage]
    img_size = args.img_size_list[stage]
    x, y = args.victim_point_list[stage]

    for i in t_train:
        # if i == decay_lr:
        #     for param_group in d_opt.param_groups:
        #         param_group['lr'] *= 0.1
        #         print("DISCRIMINATOR LEARNING RATE UPDATE TO :", param_group['lr'])
        #     for param_group in g_opt.param_groups:
        #         param_group['lr'] *= 0.1
        #         print("GENERATOR LEARNING RATE UPDATE TO :", param_group['lr'])

        for _ in range(g_iter):
            g_opt.zero_grad()

            z_list = [torch.randn(args.batch_size, 3, args.patch_size_list[z_idx],
                                  args.patch_size_list[z_idx]).cuda(args.gpu, non_blocking=True) for z_idx in range(stage + 1)]


            x_fake_list = G(z_list)
            # x_fake_list[-1].register_hook(print_grad)

            patch, mask = pad_transform(x_fake_list[-1], img_size, patch_size, x, y, args.gpu)
            patch_org = torch.mul(mask, x_in)
            victimized_in = torch.mul((1 - mask), x_in) + torch.mul(mask, patch)

            rmse_list = [torch.sqrt(F.mse_loss(patch, patch_org))]
            g_rec = F.mse_loss(patch, patch_org)

            g_fake_logit = D(victimized_in)

            tv_loss = (torch.sum(torch.abs(x_fake_list[-1][:, :, :, :-1] - x_fake_list[-1][:, :, :, 1:])) +
                       torch.sum(torch.abs(x_fake_list[-1][:, :, :-1, :] - x_fake_list[-1][:, :, 1:, :])))

            final_patch_size = args.patch_size_list[-1]
            up_patch = F.interpolate(x_fake_list[-1], (final_patch_size, final_patch_size), mode='bilinear',
                                     align_corners=True)
            patch, mask = pad_transform(up_patch, max(x_org.size()), final_patch_size, x, y, args.gpu)
            victimized_img = torch.mul((1 - mask), x_org) + torch.mul(mask, patch)
            # victimized_img.register_hook(print_grad)

            output = model(victimized_img)
            # output.register_hook(print_grad)

            adv_loss = adv_loss_fun(output, label).sum()
            # adv_loss.register_hook(print_grad)

            # wgan gp
            g_fake = -torch.mean(g_fake_logit, (2, 3))
            # g_loss = 1.0*adv_loss
            g_loss = 1*g_fake + 10* g_rec + 0.00001*tv_loss + 0.002*adv_loss
            # g_loss = 10*adv_loss
            g_loss.backward()
            g_opt.step()

            g_losses.update(g_loss.item(), x_in.size(0))

        # Update discriminator
        for _ in range(d_iter):
            x_in.requires_grad = True

            d_opt.zero_grad()
            x_fake_list = G(z_list)

            patch, mask = pad_transform(x_fake_list[-1], img_size, patch_size, x, y, args.gpu)
            # patch, mask = patch.cuda(args.gpu, non_blocking=True), mask.cuda(args.gpu, non_blocking=True)
            victimized_in = torch.mul((1 - mask), x_in) + torch.mul(mask, patch)

            d_fake_logit = D(victimized_in.detach())
            d_real_logit = D(x_in)


            # wgan gp
            d_fake = torch.mean(d_fake_logit, (2, 3))
            d_real = -torch.mean(d_real_logit, (2, 3))
            d_gp = compute_grad_gp_wgan(D, x_in, victimized_in, args.gpu)
            d_loss = d_real + d_fake + 0.1 * d_gp


            d_loss.backward()
            d_opt.step()

            d_losses.update(d_loss.item(), x_in.size(0))

        t_train.set_description('Stage: [{}/{}] Avg Loss: D[{d_losses.avg:.3f}] G[{g_losses.avg:.3f}] RMSE[{rmse:.3f}]'
                                .format(stage, args.num_scale, d_losses=d_losses, g_losses=g_losses,
                                        rmse=rmse_list[-1]))
