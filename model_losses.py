import torch
import torch.nn as nn

import kornia



class DeltaE_loss(nn.Module):
    def  __init__(self):
        super(DeltaE_loss,self).__init__()
    def forward(self, img, gt):
        img_lab =  kornia.color.rgb_to_lab(img)
        gt_lab = kornia.color.rgb_to_lab(gt)
        
        #img_lab = lab_normalize(img_lab)
        #gt_lab = lab_normalize(gt_lab)        
        
        img_c = torch.sqrt(torch.square(img_lab[:,1]) + torch.square(img_lab[:,2]) + 1e-12)
        gt_c = torch.sqrt(torch.square(gt_lab[:,1]) + torch.square(gt_lab[:,2]) + 1e-12)
        
        sc = 1 + 0.045*img_c
        sh_2 = (1 + 0.015*img_c)**2
  
        dc = img_c - gt_c
        dh_2 = torch.square(img_lab[:,1] - gt_lab[:,1]) + torch.square(img_lab[:,2] - gt_lab[:,2]) - torch.square(dc)
        
        loss = torch.square(img_lab[:,0] - gt_lab[:,0]) + torch.square(dc/sc) + dh_2/sh_2
        loss = torch.sqrt(torch.clamp(loss, 0, torch.inf) + 1e-12).mean()
            
        
        return loss