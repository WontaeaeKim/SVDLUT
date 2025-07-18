import argparse

import torch

import torchvision.transforms.functional as TF
import torchvision.transforms as T

from models import *
from PIL import Image
import cv2



parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="./demo_img/input/a4632.jpg", help="path of pretrained model")
parser.add_argument("--output_path", type=str, default="./demo_img/result/a4632.png", help="path of pretrained model")
parser.add_argument("--dataset_name", type=str, default="fivek", help="name of the dataset: fivek or ppr10k")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")

parser.add_argument("--backbone_coef", type=int, default=8, help="backbone coefficient")

parser.add_argument("--pretrained_path", type=str, default="./pretrained/fiveK_sRGB.pth", help="path of pretrained model")

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

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()

if opt.dataset_name == "ppr10k":
    backbone_type = 'resnet'
    lut_n_ranks = 10
else:
    backbone_type = 'cnn'
    lut_n_ranks = 8    


svdlut_inst = SVDLUT(backbone_type=backbone_type, backbone_coef=opt.backbone_coef,
                 lut_n_vertices=opt.lut_n_vertices, lut_n_ranks=opt.lut_n_ranks, 
                 grid_n_vertices=opt.grid_n_vertices, grid_n_ranks=opt.grid_n_ranks, ch_per_grid=opt.ch_per_grid,
                 lut_weight_ranks=opt.lut_weight_ranks, grid_weight_ranks=opt.grid_weight_ranks,
                 lut_n_singular=opt.lut_n_singular, grid_n_singular=opt.grid_n_singular)



if cuda:
    svdlut_inst = svdlut_inst.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
svdlut_inst.load_state_dict(torch.load(opt.pretrained_path))
svdlut_inst.eval()



img = Image.open(opt.input_path)
real_A = TF.to_tensor(img).type(Tensor)
real_A = real_A.unsqueeze(0)
result, _, _, _, _ = svdlut_inst(real_A)

result = result.squeeze().mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite(opt.output_path, result)


