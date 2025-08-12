import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils_val import validation_stylevec
import numpy as np
import random
from model.EncDec import Network_top    #default
from model.style_filter64 import StyleFilter_Top


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument("-restore-from-stylefilter", help='the weights of feature extraction network', type=str, default='./checkpoints/test_model/backbone')
parser.add_argument('-restore-from-backbone', help='the weights of the image restoration backbone', default='./checkpoints/test_model/style_filter', type=str)     
parser.add_argument('-val_data_dir', default='./data/test/', type=str)
parser.add_argument('-val_filename', default='snowtest100k_L.txt', type=str)
args = parser.parse_args()

val_batch_size = args.val_batch_size

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = args.val_data_dir
val_filename = args.val_filename ## This text file should contain all the names of the images and must be placed in val data directory

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- the network backbone --- #
net = Network_top().cuda()
net = nn.DataParallel(net, device_ids=device_ids)
# weights = torch.load(args.restore_from_backbone, map_location=lambda storage, loc: storage).module.state_dict()
# weights_dict = {}
# for k, v in weights.items():
#     new_k = 'module.' + k.replace('EncDec_film64_hyperW_conv', 'EncDec').replace('Transweather', 'Network_top')
#     weights_dict[new_k] = v
weights_dict = torch.load(args.restore_from_backbone)
net.load_state_dict(weights_dict)
net.eval()

# --- the style filter --- #
StyleFilter = StyleFilter_Top() 
StyleFilter.to(device)
StyleFilter = nn.DataParallel(StyleFilter, device_ids=device_ids)
# restore = torch.load(args.restore_from_stylefilter, map_location=lambda storage, loc: storage).module.state_dict()
# weights_dict = {}
# for k, v in restore.items():
#     new_k = 'module.' + k
#     weights_dict[new_k] = v
weights_dict = torch.load(args.restore_from_stylefilter)
StyleFilter.load_state_dict(weights_dict)
for param in StyleFilter.parameters():
    param.require_grad = False
StyleFilter.eval()

# --- Use the evaluation model in testing --- #
print('--- Testing starts! ---')
start_time = time.time()
with torch.no_grad():
    val_psnr, val_ssim = validation_stylevec(StyleFilter, net, val_data_loader, device)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))