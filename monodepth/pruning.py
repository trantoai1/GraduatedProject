from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from layers import *
import sparselearning
from options import MonodepthOptions
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, plot_class_feature_histograms
from trainer import Trainer

options = MonodepthOptions()
opts = options.parse()
monodepth = Trainer(opts)
is_best = False
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None



def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    #logger.info(msg)

def compute_losses( inputs, outputs):
  """Compute the reprojection and smoothness losses for a minibatch
  """
  losses = {}
  total_loss = 0

  for scale in opts.scales:
      loss = 0
      
      disp = outputs[("disp", scale)]
      color = inputs[("color", 0, scale)]
      target = inputs[("color", 0, 0)]

      
      mean_disp = disp.mean(2, True).mean(3, True)
      norm_disp = disp / (mean_disp + 1e-7)
      smooth_loss = get_smooth_loss(norm_disp, color)

      loss += opts.disparity_smoothness * smooth_loss / (2 ** scale)
      total_loss += loss
      losses["loss/{}".format(scale)] = loss

  total_loss /= len(opts.scales)
  losses["loss"] = total_loss
  return losses

def generate_images_pred( inputs, outputs):
  """Generate the warped (reprojected) color images for a minibatch.
  Generated images are saved into the `outputs` dictionary.
  """
  for scale in opts.scales:
    disp = outputs[("disp", scale)]
    if opts.v1_multiscale:
        source_scale = scale
    else:
        disp = F.interpolate(
            disp, [opts.height, opts.width], mode="bilinear", align_corners=False)
        source_scale = 0

    _, depth = disp_to_depth(disp, opts.min_depth, opts.max_depth)

    outputs[("depth", 0, scale)] = depth
  



    

def train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask=None):
    model.train()
    count = 0
    for batch_idx, inputs in enumerate(train_loader):
        if lr_scheduler is not None: lr_scheduler.step()
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(device)
        optimizer.zero_grad()
        features = model(inputs["color_aug", 0, 0].to(device))
        
        outputs = monodepth.models['depth'](features)
        
        
        generate_images_pred(inputs, outputs)
        
        losses = compute_losses(inputs, outputs)
        loss = losses["loss"]
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()
        if count % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
        count +=1
        if count % 2000==0:
          save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer' : optimizer.state_dict()},
                            is_best=False, filename=args.save_model)

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)

def save_checkpoint(state, is_best=is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'encoder.pth.tar')

def main():
    args = opts


    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))


        train_loader = monodepth.train_loader

        test_loader = monodepth.val_loader

        model = monodepth.models['encoder']

        print_and_log(model)
        print_and_log('='*60)

        print_and_log('='*60)

        print_and_log('='*60)
        print_and_log('Prune mode: {0}'.format(args.prune))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('='*60)

        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_frequency, gamma=0.1)
        if args.resume:
            if os.path.isfile(args.save_model):
                print_and_log("=> loading checkpoint '{}'".format(args.save_model))
                checkpoint = torch.load(args.save_model)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.save_model, checkpoint['epoch']))
                print_and_log('Testing...')
                #evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                #plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.save_model))


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if not args.dense:
            if args.decay_schedule == 'cosine':
                decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
            elif args.decay_schedule == 'linear':
                decay = LinearDecay(args.prune_rate, len(train_loader)*(args.epochs))
            mask = Masking(optimizer, decay, prune_rate=args.prune_rate, prune_mode=args.prune, growth_mode=args.growth, redistribution_mode=args.redistribution,
                           verbose=args.verbose, fp16=args.fp16)
            mask.add_module(model, density=args.density)

        for epoch in range(1, args.num_epochs + 1):
            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, lr_scheduler, mask)

            monodepth.epoch = epoch
            
            

            if not args.dense and epoch < args.epochs:
                mask.at_end_of_epoch()
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer' : optimizer.state_dict()},
                            is_best=False, filename=args.save_model)
            monodepth.save_model()
            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

        monodepth.val()
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

if __name__ == '__main__':
   main()
