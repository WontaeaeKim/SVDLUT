import argparse
import os
import math
import itertools
import sys


from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from model_losses import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import tqdm
import os

from torch.optim.lr_scheduler import StepLR 
import lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
    parser.add_argument("--n_epochs", type=int, default=400, help="total number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="fivek", help="name of the dataset: fivek or ppr10k")
    parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--output_dir", type=str, default="SVDLUTs", help="path to save model")
    parser.add_argument("--backbone_coef", type=int, default=8, help="backbone coefficient")
    parser.add_argument("--use_mask", type=bool, default=False,
                        help="whether to use the human region mask for weighted loss")
    
    parser.add_argument("--lut_n_vertices", type=int, default=33, help="number of LUT vertices")
    parser.add_argument("--lut_n_ranks", type=int, default=8, help="number of LUT generator ranks")
    parser.add_argument("--lut_weight_ranks", type=int, default=8, help="number of LUT weight generator ranks")
    parser.add_argument("--lut_n_singular", type=int, default=8, help="number of LUT weight generator ranks")
    
    parser.add_argument("--grid_n_vertices", type=int, default=17, help="number of GRID vertices")
    parser.add_argument("--grid_n_ranks", type=int, default=8, help="number of GRID generator ranks")
    parser.add_argument("--grid_weight_ranks", type=int, default=8, help="number of grid weight generator ranks")
    parser.add_argument("--grid_n_singular", type=int, default=8, help="number of LUT weight generator ranks")
    parser.add_argument("--ch_per_grid", type=int, default=2, help="number of GRID generator output channel")
    
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    opt.output_dir = opt.output_dir + '_' + opt.input_color_space
    print(opt)
    
    os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)
    
    file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
    file.write('\n\n')
    file.write(str(opt))
    file.write('\n\n')
    file.close()    
    
    torch.multiprocessing.set_start_method('spawn', True)

    cuda = True if torch.cuda.is_available() else False
    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss functions
    criterion_pixelwise = torch.nn.MSELoss()
  
    if opt.dataset_name == 'ppr10k':
        backbone_type = 'resnet'
    else:
        backbone_type = 'cnn'
 
  
    
    svdlut_inst = SVDLUT(backbone_type=backbone_type, backbone_coef=opt.backbone_coef,
                 lut_n_vertices=opt.lut_n_vertices, lut_n_ranks=opt.lut_n_ranks, 
                 grid_n_vertices=opt.grid_n_vertices, grid_n_ranks=opt.grid_n_ranks, ch_per_grid=opt.ch_per_grid,
                 lut_weight_ranks=opt.lut_weight_ranks, grid_weight_ranks=opt.grid_weight_ranks,
                 lut_n_singular=opt.lut_n_singular, grid_n_singular=opt.grid_n_singular)
  
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_deltaE = DeltaE_loss()
  


    if cuda:
        svdlut_inst = svdlut_inst.cuda()
        criterion_pixelwise.cuda()
        loss_fn_alex.cuda()
        loss_deltaE.cuda()


    optimizer_G = torch.optim.Adam(itertools.chain(svdlut_inst.parameters()), lr=opt.lr)
    scheduler = StepLR(optimizer_G, step_size=400, gamma=0.1)

    if opt.epoch != 0:
        # Load pretrained models
        svdlut_inst.load_state_dict(torch.load("saved_models/%s/svdlut_%d.pth" % (opt.output_dir, opt.epoch-1)))
        scheduler.load_state_dict(torch.load("saved_models/%s/scheduler_%d.pth" % (opt.output_dir, opt.epoch-1)))
    else:
        pass
        # Initialize weights
        svdlut_inst.init_weights()
        
        
    if opt.input_color_space == 'XYZ':
        dataloader = DataLoader(
            ImageDataset_XYZ("../../dataset/%s" % opt.dataset_name, mode = "train"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )

        psnr_dataloader = DataLoader(
            ImageDataset_XYZ("../../dataset/%s" % opt.dataset_name,  mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        if opt.dataset_name == 'ppr10k':            
            dataloader = DataLoader(
                ImageDataset_PPR10k("../../dataset/%s" % opt.dataset_name, mode = "train",use_mask=opt.use_mask),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,
            )

            psnr_dataloader = DataLoader(
                ImageDataset_PPR10k("../../dataset/%s" % opt.dataset_name,  mode="test",use_mask=opt.use_mask),
                batch_size=1,
                shuffle=False,
                #num_workers=1,
                num_workers=opt.n_cpu,
            )
        else:    
            dataloader = DataLoader(
                ImageDataset_sRGB("../../dataset/%s" % opt.dataset_name, mode = "train"),
                batch_size=1,
                shuffle=True,
                num_workers=opt.n_cpu,
            )

            psnr_dataloader = DataLoader(
                ImageDataset_sRGB("../../dataset/%s" % opt.dataset_name,  mode="test"),
                batch_size=1,
                shuffle=False,
                #num_workers=1,
                num_workers=opt.n_cpu,
            )
    
    # ----------
    #  Training
    # ----------
    max_psnr = 0
    max_epoch = 0
    
    if cuda:
        print('cuda is available!!')
        
    for epoch in range(opt.epoch, opt.n_epochs):
        mse_avg = 0
        psnr_avg = 0
        log_cnt = 0
        svdlut_inst.train()
        
        iterator = tqdm.tqdm(dataloader)
        
        #for i, batch in enumerate(dataloader):
        for batch in iterator:

            # Model inputs
            real_A = Variable(batch["A_input"].type(Tensor))
            real_B = Variable(batch["A_exptC"].type(Tensor))
            
            if opt.use_mask:
                mask = Variable(batch["mask"].type(Tensor))
                mask = torch.sum(mask, 1).unsqueeze(1)
                weights = torch.ones_like(mask)
                weights[mask > 0] = 5

            if cuda:
                real_A.cuda()
                real_B.cuda()
            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            fake_B, lut_weights, grid_weights, g3d_lut, gbilateral = svdlut_inst(real_A)

            
            # Pixel-wise loss
            if opt.use_mask:
                mse = criterion_pixelwise(fake_B * weights, real_B * weights)
            else:
                mse = criterion_pixelwise(fake_B, real_B)
            
            perceptual_loss = torch.mean(loss_fn_alex(fake_B, real_B))
            deltaE_loss = loss_deltaE(fake_B, real_B)
            
            
            
            loss = mse + 0.05*perceptual_loss + 0.005*deltaE_loss
            
            psnr_avg += 10 * math.log10(1 / mse.item())

            mse_avg += mse.item()

            loss.backward()

            optimizer_G.step()

            log_cnt = log_cnt +1
            # --------------
            #  Log Progress
            # --------------
           
            iterator.set_description(f"[Epoch {epoch+1}/{opt.n_epochs}][psnr: {psnr_avg / (log_cnt)}]")
        
        svdlut_inst.eval()
        avg_psnr = 0
        for i, batch in enumerate(psnr_dataloader):
            real_A = Variable(batch["A_input"].type(Tensor))
            real_B = Variable(batch["A_exptC"].type(Tensor))
            fake_B, _, _, _, _ = svdlut_inst(real_A)

            fake_B = torch.round(fake_B*255)
            real_B = torch.round(real_B*255)
            mse = criterion_pixelwise(fake_B, real_B)
            psnr = 10 * math.log10(255.0 * 255.0 / mse.item())

            avg_psnr += psnr

        avg_psnr = avg_psnr/ len(psnr_dataloader)
        
        if avg_psnr > max_psnr:
            max_psnr = avg_psnr
            max_epoch = epoch
        sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))
        
       


        if epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(svdlut_inst.state_dict(), "saved_models/%s/svdlut_%d.pth" % (opt.output_dir, epoch))
            torch.save(scheduler.state_dict(), "saved_models/%s/scheduler_%d.pth" % (opt.output_dir, epoch))
            file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
            file.write("[epoch:%d] [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (epoch, avg_psnr,  max_psnr, max_epoch))
            file.close()

        scheduler.step()


