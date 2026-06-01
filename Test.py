import torch
import torch.nn as nn
import math
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import cv2

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
from utils import *
from model import FDMATNet


parser = ArgumentParser(description='FDMATNet')

parser.add_argument('--epoch_num', type=int, default=800, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DBNet')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=5, help='from {5, 10, 20, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--test_name', type=str, default='BrainImages', choices=['BrainImages', 'IXI', 'CC359'],
                    help='name of test set, choose BrainImages, IXI, CC359')
parser.add_argument('--mask_name', type=str, default='radial', choices=['radial', 'random', 'cartesian'],
                    help='name of mask, radial, random, cartesian')
parser.add_argument('--net', type=str, default='FDMATNet', help='Name of Net')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

args = parser.parse_args()

epoch_num = args.epoch_num
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
mask_name = args.mask_name
net_name = args.net
###########################################################################################

try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################################################################################
# Load CS Sampling Matrix: phi
# Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data_Name = './%s/mask_%s_%d.mat' % (args.matrix_dir, mask_name, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask']

mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = mask_matrix.to(device)
###########################################################################################

model = FDMATNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

###########################################################################################
model_dir = "./%s/MRI_CS_%s_%s_%s_layer_%d" % (args.model_dir, args.net, mask_name, test_name, layer_num)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


print('\n')
print("MRI CS Reconstruction Start")

model.eval()
with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Iorg = cv2.imread(imgName, 0)

        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        Img_output = Icol

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        x_in_k_space = torch.fft.fft2(batch_x)  # 图像域转化k-space data
        masked_x_in_k_space = x_in_k_space * mask  # 下采样k-space data

        PhiTb = torch.fft.ifft2(masked_x_in_k_space)
        PhiTb = torch.view_as_real(PhiTb).squeeze(1).permute(0, 3, 1, 2)    #[2,256,256]
        masked_x_in_k_space = torch.view_as_real(masked_x_in_k_space).squeeze(1).permute(0, 3, 1, 2)  # 下采样kspace数据

        start = time()
        x_output = model(PhiTb, masked_x_in_k_space, mask)

        end = time()
        runtime = (end - start) * 1000  # 转换为毫秒
        print(runtime, 'ms')

        PhiTb = complex_abs(PhiTb)
        x_output = complex_abs(x_output)

        initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)

        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)

        X_init = np.clip(initial_result, 0, 1).astype(np.float64)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)

        init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
        init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)

        rec_PSNR = psnr(X_rec*255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255., Iorg.astype(np.float64), data_range=255)


        print("[%02d/%02d] Run time for %s is %.4f, Initial  PSNR is %.2f, Initial  SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), init_PSNR, init_SSIM))
        print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        # im_rec_rgb = np.clip(X_rec*255, 0, 255).astype(np.uint8)
        #
        # resultName = imgName.replace(args.data_dir, args.result_dir)
        # cv2.imwrite("%s_%s_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, net_name, cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)

        # im_init__rgb = np.clip(X_init*255, 0, 255).astype(np.uint8)
        #
        # resultName = imgName.replace(args.data_dir, args.result_dir)
        # cv2.imwrite("%s_ZF_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, cs_ratio, init_PSNR, init_SSIM), im_init__rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

        Init_PSNR_All[0, img_no] = init_PSNR
        Init_SSIM_All[0, img_no] = init_SSIM

print('\n')
init_data =   "%s CS ratio is %d, Avg Initial  PSNR/SSIM for %s is %.2f/%.4f" % (mask_name, cs_ratio, args.test_name, np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
output_data = "%s CS ratio is %d, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (mask_name, cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(init_data)
print(output_data)

output_file_name = "./%s/PSNR_SSIM_Results_MRI_CS_%s_layer_%d_group_%d.txt" % (args.log_dir, net_name, layer_num, group_num)

output_file = open(output_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("MRI CS Reconstruction End")