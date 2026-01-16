import argparse
import torch

from models import *
from datasets import *
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="fivek", help="name of the dataset: fivek or ppr10k")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--lut_inter", type=str, default="tri", help="LUT interpolation method")
parser.add_argument("--pretrained_path", type=str, default="./pretrained/fiveK_sRGB.pth", help="path of pretrained model")

parser.add_argument("--backbone_coef", type=int, default=8, help="backbone coefficient")


parser.add_argument("--lut_n_vertices", type=int, default=33, help="number of LUT vertices")
parser.add_argument("--lut_n_ranks", type=int, default=8, help="number of LUT generator ranks")
parser.add_argument("--lut_weight_ranks", type=int, default=8, help="number of LUT weight generator ranks")
parser.add_argument("--lut_n_singular", type=int, default=8, help="number of LUT weight generator ranks")

parser.add_argument("--grid_n_vertices", type=int, default=17, help="number of GRID vertices")
parser.add_argument("--grid_n_ranks", type=int, default=8, help="number of GRID generator ranks")
parser.add_argument("--grid_weight_ranks", type=int, default=8, help="number of grid weight generator ranks")
parser.add_argument("--grid_n_singular", type=int, default=8, help="number of LUT weight generator ranks")
parser.add_argument("--ch_per_grid", type=int, default=2, help="number of GRID generator output channel")

parser.add_argument("--input_video", type=str, default="cutting_orange_tuil_15000kbps_1080p_59.94fps_h264.mp4", help="Input video path for video test")
parser.add_argument("--output_video", type=str, default="output_compare.mp4", help="Input video path for video test")

opt = parser.parse_args()
print(opt)

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False

#cuda = False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()

if opt.dataset_name == "ppr10k":
    backbone_type = 'resnet'
    lut_n_ranks = 10
    device = 'cuda'
else:
    backbone_type = 'cnn'
    lut_n_ranks = 8
    device = 'cpu'    


svdlut_inst = SVDLUT(backbone_type=backbone_type, backbone_coef=opt.backbone_coef,
                 lut_n_vertices=opt.lut_n_vertices, lut_n_ranks=opt.lut_n_ranks, 
                 grid_n_vertices=opt.grid_n_vertices, grid_n_ranks=opt.grid_n_ranks, ch_per_grid=opt.ch_per_grid,
                 lut_weight_ranks=opt.lut_weight_ranks, grid_weight_ranks=opt.grid_weight_ranks,
                 lut_n_singular=opt.lut_n_singular, grid_n_singular=opt.grid_n_singular)


if cuda:
    svdlut_inst = svdlut_inst.cuda()
    criterion_pixelwise.cuda()


svdlut_inst.load_state_dict(torch.load(opt.pretrained_path))
svdlut_inst.eval()    
    
input_video = opt.input_video
output_video = opt.output_video

cap = cv2.VideoCapture(input_video)

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width * 2, height))



with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).type(Tensor).unsqueeze(0)
    
        out, _, _, _, _ = svdlut_inst(tensor)
              
        out_img = (out.squeeze(0).clamp(0,1).permute(1,2,0).detach().cpu().numpy() * 255).astype("uint8")
    
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        cv2.putText(frame, 'Original Video', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out_img, 'Enhanced Video', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        
        compare = np.hstack([frame, out_img])


        writer.write(compare)

        cv2.imshow("Original | Model Output", compare)
        if cv2.waitKey(1) == 27:  # ESC
            break



cap.release()
writer.release()
cv2.destroyAllWindows()

print("Save Video:", output_video)