# -*- coding:utf-8 -*-


"""

@author: Jan
@file: config.py
@time: 2019/9/1 13:22
"""
import argparse

## optimizer
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam updater')
parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam updater')

## networks
parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in the first conv layer')
parser.add_argument('--norm_type', type=str, default='instance',
                    help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--init_type', type=str, default='normal',
                    help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--n_layers', type=int, default=3, help='only used if netD==n_layers')

## data
parser.add_argument('--dataroot', type=str, default='D:\\大学\\大四\\__data__\\horse2zebra')
parser.add_argument('--trainA', type=str, default='trainA', help='Path to training data A')
parser.add_argument('--trainB', type=str, default='trainB', help='Path to training data B')
parser.add_argument('--testA', type=str, default='testA', help='Path to trsting data A')
parser.add_argument('--testB', type=str, default='testB', help='Path to testing data B')
parser.add_argument('--input_nc', type=int, default=3, help='Number of channels of input')
parser.add_argument('--output_nc', type=int, default=3, help='Number of channels of output')
parser.add_argument('--load_size', type=int, default=32, help='Size of loaded in images')
parser.add_argument('--crop_size', type=int, default=32, help='Size of fed in images')
parser.add_argument('--batch_size', type=int, default=1, help='Size of a batch')
parser.add_argument('--n_threads', type=int, default=0, help='Number of workers')

## train
parser.add_argument('--mode', type=str, default='train', help='Training or testing')
parser.add_argument('--epoch', type=int, default=200, help='Epoches for training separately')
parser.add_argument('--gpu_id', type=int, default=+0, help='Gpu id for training/testing')
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpints directory')
parser.add_argument('--name', type=str, default='CrossNet_horse2zebra', help='Experiment name')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--epoch_count', type=int, default=1,
                    help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument('--gan_mode', type=str, default='lsgan',
                    help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

# losses
parser.add_argument('--lambda_gan', type=float, default=1.0, help='Weight of GAN loss')
parser.add_argument('--lambda_idt', type=float, default=3.0, help='Weight of Identity loss')
parser.add_argument('--lambda_ctc', type=float, default=3.0, help='Weight of latent cross-translation consistency')
parser.add_argument('--lambda_zid', type=float, default=6.0, help='Weight of latent cross-identity loss')
parser.add_argument('--lambda_zcyc', type=float, default=6.0, help='Weight of latent cycle consistency')

# log and validate
parser.add_argument('--log_freq', type=int, default=50, help='Frequency in iterations to log')
parser.add_argument('--vis_freq', type=int, default=200, help='Frequency in iterations to visualize')
parser.add_argument('--ckp_freq', type=int, default=5, help='Frequency in iterations to save models')
