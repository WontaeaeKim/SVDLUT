import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.transforms.functional import to_pil_image

from cpp_ext_interface import bilinear_2Dslicing_lut_transform


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

class Backbone(nn.Module):
    def __init__(self, backbone_coef=8):
        super(Backbone, self).__init__()
        self.backbone_coef = backbone_coef
        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, backbone_coef, 3, stride=2, padding=1), #8 x 128 x 128
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(backbone_coef, affine=True),
            *discriminator_block(backbone_coef, 2*backbone_coef, normalization=True), #16 x 64 x 64
            *discriminator_block(2*backbone_coef, 4*backbone_coef, normalization=True), #32 x 32 x 32
            *discriminator_block(4*backbone_coef, 8*backbone_coef, normalization=True), #64 x 16 x 16
            *discriminator_block(8*backbone_coef, 8*backbone_coef),   #64 x 8 x 8
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(5, stride=2) #64 x 2 x 2
        )

    def forward(self, img_input):

        return self.model(img_input).view([-1,self.backbone_coef*32])

class resnet18_224(nn.Module):

    def __init__(self, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        net.fc = nn.Identity()
        self.model = net
       
    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f
    
class Gen_2D_SVD_LUT(nn.Module):
    def __init__(self, n_colors=3, ch_per_lut = 3, n_lut_dim=2, n_vertices=17, n_feats=256, n_ranks=24, n_singlar=8):
        super(Gen_2D_SVD_LUT, self).__init__()
        
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        
        self.n_svd = n_vertices * n_singlar + n_singlar + n_singlar * n_vertices
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * ch_per_lut * self.n_svd)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        self.ch_per_lut = ch_per_lut
        self.n_singlar = n_singlar
    
    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        nn.init.zeros_(self.basis_luts_bank.bias)
        cols, rows = torch.stack(torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(2)]),dim=0).div(self.n_vertices - 1).flip(0)
        zero2d = torch.zeros(self.n_vertices, self.n_vertices)
        d = torch.stack([cols,cols,zero2d, 
                         rows,zero2d,cols,
                         zero2d,rows,rows], dim=0)
        
        u,s,v =torch.svd(d)
        
        u = u[:,:,:self.n_singlar].contiguous().view([3*self.ch_per_lut,-1])
        s = s[:,:self.n_singlar]
        v = v[:,:,:self.n_singlar].mT.contiguous().view([3*self.ch_per_lut,-1])
        
        d= torch.cat([u,s,v], dim=1)
        
        identity_lut = torch.stack([d,*[torch.zeros(3 * self.ch_per_lut, self.n_svd) for _ in range(self.n_ranks - 1)]], dim=0).view(self.n_ranks, -1)
        
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())
       
    def forward(self, img_feature):
        weights = self.weights_generator(img_feature)
        lut_svd = self.basis_luts_bank(weights)

        lut_svd = lut_svd.view([-1, self.n_svd])
        
        lut_u = lut_svd[:,:self.n_vertices * self.n_singlar]
        lut_s = lut_svd[:,self.n_vertices * self.n_singlar:self.n_vertices * self.n_singlar + self.n_singlar]
        lut_v = lut_svd[:,self.n_vertices * self.n_singlar + self.n_singlar:]
        
        lut_u = lut_u.view([-1, self.n_vertices, self.n_singlar])
        lut_s = torch.diag_embed(lut_s)
        lut_v = lut_v.view([-1, self.n_singlar, self.n_vertices])
        
        luts = torch.bmm(torch.bmm(lut_u,lut_s), lut_v)
        
        luts = luts.view([-1,self.n_colors, self.ch_per_lut, self.n_vertices,self.n_vertices])
        return luts, weights
    
class Gen_2D_LUT_weight_bias(nn.Module):
    def __init__(self, n_colors=3, ch_per_lut = 3, n_vertices=17, n_feats=256, n_ranks=24):
        super(Gen_2D_LUT_weight_bias, self).__init__()
        
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (ch_per_lut + 1))

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        self.ch_per_lut = ch_per_lut
  
    
    def init_weights(self):
        nn.init.ones_(self.weights_generator.bias)
        nn.init.zeros_(self.basis_luts_bank.bias)
        
        d = torch.tensor([[0.5,0.5,0,0],
                         [0.5,0,0.5,0],
                         [0,0.5,0.5,0]])
        
        
        identity_lut = torch.stack([d,
            *[torch.zeros(self.n_colors, self.ch_per_lut + 1) for _ in range(self.n_ranks - 1)]], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())
       
    def forward(self, img_feature):
        weights = self.weights_generator(img_feature)
        weights_bias = self.basis_luts_bank(weights)

        weights_bias = weights_bias.view([-1,self.n_colors, self.ch_per_lut + 1])
        lut_param_weights = weights_bias[:, :, :self.n_colors]
        lut_param_bias = weights_bias[:, :, self.n_colors:]
        return lut_param_weights, lut_param_bias      


class Gen_2D_bilateral_grids(nn.Module):
    def __init__(self, n_grid_dim=2, n_vertices=17, n_feats=256, n_ranks=24, ch_per_grid=2):
        super(Gen_2D_bilateral_grids, self).__init__()
        
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_grids_bank = nn.Linear(
            n_ranks, ch_per_grid * 3 * 3 * (n_vertices ** n_grid_dim))

        self.n_grid_dim = n_grid_dim
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        self.ch_per_grid = ch_per_grid
        self.n_grids = ch_per_grid * 3
  
    
    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        nn.init.zeros_(self.basis_grids_bank.bias)
        cols, rows = torch.stack(torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(2)]),dim=0).div(self.n_vertices - 1).flip(0)
        zero2d = torch.zeros(self.n_vertices, self.n_vertices)
        d = torch.stack([*[zero2d,rows,rows, 
                         zero2d,rows,rows,
                         zero2d,rows,rows] * self.ch_per_grid], dim=0)
        identity_grid = torch.stack([d,*[torch.zeros(self.n_grids * 3,self.n_vertices, self.n_vertices) for _ in range(self.n_ranks - 1)]], dim=0).view(self.n_ranks, -1)
        self.basis_grids_bank.weight.data.copy_(identity_grid.t())
        
    def forward(self, img_feature):
        weights = self.weights_generator(img_feature)
        grids = self.basis_grids_bank(weights)
        grids = grids.view([-1,self.n_grids,3,self.n_vertices,self.n_vertices])
        return grids, weights
    
    

class Gen_2D_bilateral_grids_weight_bias(nn.Module):
    def __init__(self, n_colors=3,  ch_per_grid=2, n_vertices=17, n_feats=256, n_ranks=24):
        super(Gen_2D_bilateral_grids_weight_bias, self).__init__()
        
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, ch_per_grid * (3 * n_colors  + n_colors))

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        self.ch_per_grid = ch_per_grid

  
    
    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        nn.init.zeros_(self.basis_luts_bank.bias)
        d = torch.tensor([*[[0,1,1,0],
                         [0,1,1,0],
                         [0,1,1,0]] * self.ch_per_grid]).div(self.ch_per_grid * 2)
       
        
        identity_lut = torch.stack([d,
            *[torch.zeros(3*self.ch_per_grid, self.n_colors + 1) for _ in range(self.n_ranks - 1)]], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())
       
    def forward(self, img_feature):
        weights = self.weights_generator(img_feature)
        weights_bias = self.basis_luts_bank(weights)
        
        weights_bias = weights_bias.view([-1,self.ch_per_grid, 3 *self.n_colors + self.n_colors])
        
        grid_param_weights = weights_bias[:, :, : 3 * self.n_colors]
        grid_param_bias = weights_bias[:, :, 3 * self.n_colors:]
        return grid_param_weights, grid_param_bias      
         

class SVDLUT(nn.Module):
    def __init__(self, backbone_type='cnn', backbone_coef=8,
                 lut_n_vertices=17, lut_n_ranks=24,  
                 grid_n_vertices=17, grid_n_ranks=24, ch_per_grid=2,
                 lut_weight_ranks=8, grid_weight_ranks=8,
                 lut_n_singular=8, grid_n_singular=8):
        super(SVDLUT, self).__init__()
        
        self.backbone_type = backbone_type.lower()
        
        if backbone_type.lower() == 'resnet':
            self.backbone = resnet18_224()
            print('Resnet backbone apply')
            n_feats = 512
        else:
            self.backbone = Backbone(backbone_coef=backbone_coef)
            print('CNN backbone apply')
            n_feats = 32*backbone_coef

        self.gen_2d_lut = Gen_2D_SVD_LUT(n_vertices=lut_n_vertices, n_feats=n_feats, n_ranks=lut_n_ranks, n_singlar=lut_n_singular) 
        self.gen_2d_lut_weight_bias = Gen_2D_LUT_weight_bias(n_vertices=lut_n_vertices, n_feats=n_feats, n_ranks=lut_weight_ranks)
        self.gen_2d_bilateral = Gen_2D_bilateral_grids(n_vertices=grid_n_vertices, n_feats=n_feats, n_ranks=grid_n_ranks, ch_per_grid=ch_per_grid)

        self.gen_2d_grid_weight_bias =Gen_2D_bilateral_grids_weight_bias(n_vertices=grid_n_vertices, n_feats=n_feats, n_ranks=grid_weight_ranks, ch_per_grid=ch_per_grid)
        

        self.slicing_transform = bilinear_2Dslicing_lut_transform

        self.relu = nn.ReLU()
    
    def init_weights(self):
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        if self.backbone_type != 'resnet':
            self.backbone.apply(special_initilization)
            
        self.gen_2d_lut.init_weights()
        self.gen_2d_lut_weight_bias.init_weights()
        self.gen_2d_bilateral.init_weights()
        self.gen_2d_grid_weight_bias.init_weights()
       
    def forward(self, img):
        img_feature = self.backbone(img)
        
        g3d_lut, lut_weights = self.gen_2d_lut(img_feature)
        lut_param_weights, lut_param_bias = self.gen_2d_lut_weight_bias(img_feature)
        
        gbilateral, grid_weights = self.gen_2d_bilateral(img_feature)
        grid_param_weights, grid_param_bias = self.gen_2d_grid_weight_bias(img_feature)
        

        output = self.slicing_transform(gbilateral, img, grid_param_weights, grid_param_bias, g3d_lut, lut_param_weights, lut_param_bias)
        output = self.relu(output)
        return output, lut_weights, grid_weights, g3d_lut, gbilateral

 