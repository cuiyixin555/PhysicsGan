#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:05:33 2018

@author: yangliu
"""

import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
import time, math
import matplotlib.pyplot as plt
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.name = 'dehazing'
opt.model = 'cycle_gan_dehaze'
opt.phase = 'test'
opt.dataset_mode = 'aligned_dehaze_test'
opt.no_dropout = 'True'
opt.resize_or_crop = ''
opt.dataroot = './datasets/dehaze'
opt.which_model_netG = 'resnet_9blocks'

# for RGB image with range [0,255]
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border, :]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
psnr_all_epochs = np.array([])
for epoch in range(200,201,5):
    opt.which_epoch = str(epoch)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    psnr_epoch = np.array([])
    for i, data in enumerate(dataset):
#        if i >= opt.how_many:
#            break
        model.set_input_(data)
        model.test_()
        visuals = model.get_current_visuals()

        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path)

        # pred = visuals['fake_B'].astype(float)
        # gt = visuals['real_B'].astype(float)
        # psnr_epoch = np.append(psnr_epoch, PSNR(pred, gt))
    
    # webpage.save()

    # psnr_all_epochs = np.append(psnr_all_epochs, np.mean(psnr_epoch))
    
# np.save(opt.results_dir + opt.name + '/psnr_all_epochs.npy', psnr_all_epochs)
# psnr_all_epochs = np.load(web_dir + '/psnr_all_epochs.npy')
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Avg. PSNRs', fontsize=14)
# plt.xlim(118,200)
# plt.title('')
# plt.grid(True, linestyle = '-.')
# plt.plot(range(118, 201, 1), psnr_all_epochs, 'r', label = opt.name)
# plt.legend(loc='lower right', fontsize=12)
# plt.savefig(opt.results_dir + opt.name + '/psnr_all_epochs.pdf')

