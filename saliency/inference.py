from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from saliency.models.model import SODModel
from saliency.models.dataloader import InfDataloader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your models.')
    parser.add_argument('--imgs_folder', default='./data', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to models', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=1, help='Batch Size for testing', type=int)
    return parser.parse_args()

def run_inference(args):
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['models'])
    model.to(device)
    model.eval()
    inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)
    print("Press 'q' to quit.")
    with torch.no_grad():
        for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)
            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))
            print('Image :', batch_idx)
            cv2.imshow('Input Image', img_np)
            cv2.imshow('Generated Saliency Mask', pred_masks_raw)
            cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

if __name__ == '__main__':
    rt_args = parse_arguments()
    run_inference(rt_args)
