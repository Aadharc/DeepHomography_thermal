import numpy as np
import torch
import torch.nn as nn


# # def make_mesh(patch_w,patch_h):
# patch_w, patch_h = 512, 512
# HEIGHT, WIDTH = 640, 640
# rho = 16
# x_flat = np.arange(0,patch_w)
# # print(x_flat)
# # print(x_flat.shape)
# x_flat = x_flat[np.newaxis,:]
# # print(x_flat)
# # print(x_flat.shape)

# y_one = np.ones(patch_h)
# y_one = y_one[:,np.newaxis]
# x_mesh = np.matmul(y_one , x_flat)
# # print(x_mesh)
# print(x_mesh.shape)
# x_t_flat = np.reshape(x_mesh, (-1))

# y_flat = np.arange(0,patch_h)
# y_flat = y_flat[:,np.newaxis]
# x_one = np.ones(patch_w)
# x_one = x_one[np.newaxis,:]
# y_mesh = np.matmul(y_flat,x_one)
# y_t_flat = np.reshape(y_mesh, (-1))
# print(y_t_flat.shape)
# # print(512*512)


# x = np.random.randint(rho, WIDTH - rho - patch_w)
# print( WIDTH - rho - patch_w)
# print(x)
# # print(x)
# y = np.random.randint(rho, HEIGHT - rho -patch_h)
# print(y)
# patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)
# print(patch_indices)
# print(patch_indices.shape)
# # return x_mesh,y_mesh

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False, size_average=False)
mask_ap = torch.rand(1, 3, 4)
sum_value = torch.sum(mask_ap)
patch_1 = torch.rand(1, 3, 4)
patch_2 = torch.rand(1, 3, 4)
pred_I2_CnnFeature = torch.rand(1, 3, 4)
feature_loss_mat = triplet_loss(patch_2, pred_I2_CnnFeature, patch_1)
print(feature_loss_mat)

feature_loss = torch.sum(torch.mul(feature_loss_mat, mask_ap)) / sum_value
print(feature_loss)
feature_loss = torch.unsqueeze(feature_loss, 0)
print(feature_loss)
