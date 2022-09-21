import os
import argparse
from tqdm import tqdm
import time

# from torchsummary import summary

import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from HPCNet import HPCNet
from skimage import img_as_ubyte
import thop
import torch.nn.functional as F
from math import floor

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/model_latest54.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = HPCNet()

device_ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration, device_ids=[0]).cuda()
model_restoration.eval()
model_restoration.to(device)

datasets = ['R100L']  # ['Test100', 'Test1200', 'coco350', 'bdd350', 'Rain100L', 'Rain100H', 'Real127']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             drop_last=False, pin_memory=True)  # num_workers=4,

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    all_time = 0
    count = 0
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]
            b, c, h, w = input_.shape
            st_time = time.time()
            restored = model_restoration(input_)
            ed_time = time.time()
            cost_time = ed_time - st_time
            all_time += cost_time
            count += 1

            restored = torch.clamp(restored[0], 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)
    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time / (count)))  # (ii+1)
# print(input_.shape)
# [1, 3, 384, 512]
# summary(model_restoration, (3, 384, 512,))


_input = torch.randn(1, 3, 512, 512).cuda()
flops, params = thop.profile(model_restoration, (_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
