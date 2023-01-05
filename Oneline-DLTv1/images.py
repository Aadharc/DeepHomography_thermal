from dataset import TrainDataset
import cv2
import os
import torch
import natsort
exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
vis_path = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/Train/VIS/"
ir_path = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/Train/IR/"

all_vis = os.listdir(vis_path)
all_ir = os.listdir(ir_path)

sort_vis = natsort.natsorted(all_vis)
sort_ir = natsort.natsorted(all_ir)

for index in range(10):
    vis_img_loc = os.path.join(vis_path, sort_vis[index])
    ir_img_loc = os.path.join(ir_path, sort_ir[index])

    img1 = cv2.imread(vis_img_loc)
    img2 = cv2.imread(ir_img_loc)
    cv2.imshow('vis', img1)
    cv2.imshow('ir', img2)
    cv2.waitKey(0)