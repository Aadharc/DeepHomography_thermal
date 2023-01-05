from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import natsort


def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh

def center_crop(img, dim):
    """
    Return the cropped image
    
    Args:
    img : image to be center cropped
    dim : dimension(w,h) of cropped image
    """
    width, height = img.shape[1], img.shape[0]

    width_crop = dim[0] if dim[0] < width else width
    height_crop = dim[1] if dim[1] < height else height

    mid_x, mid_y = int(width/2), int(height/2)
    mid_cw, mid_ch = int(width_crop/2), int(height_crop/2)

    crop_img = img[mid_y - mid_ch: mid_y + mid_ch, mid_x - mid_cw : mid_x + mid_cw]
    return crop_img

class TrainDataset(Dataset):
    def __init__(self, data_path_vis, data_path_ir, exp_path, patch_w=256, patch_h=256, rho=16):

        # self.imgs = open(data_path, 'r').readlines()
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.data_path_vis = data_path_vis
        self.data_path_ir = data_path_ir
        all_imgs_vis = os.listdir(data_path_vis)
        all_imgs_ir = os.listdir(data_path_ir)
        self.total_img_vis = natsort.natsorted(all_imgs_vis)
        self.total_img_ir = natsort.natsorted(all_imgs_ir)

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = 640
        self.HEIGHT = 512
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w, self.patch_h)
        self.train_path = os.path.join(exp_path, 'Data/Train/')

    def __getitem__(self, index):

        # value = self.imgs[index]
        # img_names = value.split(' ')
        vis_img_loc = os.path.join(self.data_path_vis, self.total_img_vis[index])
        ir_img_loc = os.path.join(self.data_path_ir, self.total_img_ir[index])

        img_1 = cv2.imread(vis_img_loc)
        # print(img_1.shape)
        img_1 = center_crop(img_1, (2700, 2160))

        height, width = img_1.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))
        
        img_1 = (img_1 - self.mean_I) / self.std_I
        img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        img_2 = cv2.imread(ir_img_loc)
        # print(img_2.shape)
        height, width = img_2.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))

        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_2, img_1], axis=0)
        # print('org img',org_img.shape)

        x = np.random.randint(self.rho, self.WIDTH - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.HEIGHT - self.rho - self.patch_h)

        # x, y =0, 0

        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]
        # print("input tensor", input_tesnor.shape)
        

        y_t_flat = np.reshape(self.y_mesh, (-1))
        x_t_flat = np.reshape(self.x_mesh, (-1))
        patch_indices = (y_t_flat + y) * self.WIDTH + (x_t_flat + x)
        # print("patch indices", patch_indices.shape)

        top_left_point = (x + self.rho, y + self.rho)
        bottom_left_point = (x + self.rho, y + self.patch_h - self.rho)
        bottom_right_point = (self.patch_w + x - self.rho, self.patch_h + y - self.rho)
        top_right_point = (x + self.patch_w - self.rho, y + self.rho)
        h4p = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        h4p = np.reshape(h4p, (-1))

        org_img = torch.tensor(org_img)
        input_tesnor = torch.tensor(input_tesnor)
        patch_indices = torch.tensor(patch_indices)
        h4p = torch.tensor(h4p)
        # print("h4p", h4p.shape)

        return (org_img, input_tesnor, patch_indices, h4p)

    def __len__(self):

        return len(self.total_img_ir)


class TestDataset(Dataset):
    def __init__(self, data_path, patch_w=560, patch_h=315, rho=16, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w,self.patch_h)

        self.work_dir = os.path.join(data_path, 'Data')
        self.pair_list = list(open(os.path.join(self.work_dir, 'Test_List.txt')))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Test/')
        self.npy_path = os.path.join(self.work_dir, 'Coordinate/')

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')
        npy_id = pari_id[0].split('/')[1] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
        npy_id = self.npy_path + npy_id
        video_name = img_pair.split('/')[0]

        # load img1
        if pari_id[0][-1] == 'M':
            img_1 = cv2.imread(self.img_path + pari_id[0][:-2])
        else:
            img_1 = cv2.imread(self.img_path + pari_id[0])

        # load img2
        if pari_id[1][-2] == 'M':
            img_2 = cv2.imread(self.img_path + pari_id[1][:-3])
        else:
            img_2 = cv2.imread(self.img_path + pari_id[1][:-1])
        
        height, width = img_1.shape[:2]
 
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))

        print_img_1 = img_1.copy()
        print_img_1 = np.transpose(print_img_1, [2, 0, 1])

        img_1 = (img_1 - self.mean_I) / self.std_I
        img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        height, width = img_2.shape[:2]

        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))

        print_img_2 = img_2.copy()
        print_img_2 = np.transpose(print_img_2, [2, 0, 1])
        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_1, img_2], axis=0)
        WIDTH = org_img.shape[2]
        HEIGHT = org_img.shape[1]

        x = np.random.randint(self.rho, WIDTH - self.rho - self.patch_w)
        x = 40  # patch should in the middle of full img when testing
        y = np.random.randint(self.rho, HEIGHT - self.rho - self.patch_h)
        y = 23  # patch should in the middle of full img when testing
        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, [-1])
        x_t_flat = np.reshape(self.x_mesh, [-1])
        patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        four_points = np.reshape(four_points, (-1))

        return (org_img, input_tesnor, patch_indices, four_points,print_img_1, print_img_2, video_name, npy_id)

    def __len__(self):

        return len(self.pair_list)
