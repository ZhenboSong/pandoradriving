import numpy as np
import cv2
import matplotlib.pyplot as plt

input_height = h = 66
input_width = w = 200
horizontal_view_angle = 52.

fx, fy, cx, cy = 1452.267863, 1438.239667, 607.192313, 348.549486
K = [fx, fy, cx, cy]


class Img_map:
    def __init__(self):
        h, w = 720, 1280
        horiz_image = np.arange(0, w).repeat(h).reshape(w, h).T.astype(np.float)
        vert_image = np.arange(0, h).repeat(w).reshape(h, w).astype(np.float)
        horiz_image = np.arctan2(horiz_image - cx, fx)*180./np.pi
        vert_image = np.arctan2(vert_image - cy, fy)*180./np.pi
        
        max_hor = np.max(horiz_image)
        min_hor = np.min(horiz_image)
        horiz_image = (horiz_image-min_hor)/(max_hor-min_hor)
        
        max_ver = np.max(vert_image)
        min_ver = np.min(vert_image)
        vert_image = (vert_image-min_ver)/(max_ver-min_ver)
        
        self.horiz_image = horiz_image
        self.vert_image = vert_image
        
    def get_dep(self, path):
        
        img = cv2.imread(path)
        h, w, _ = img.shape
        img = img/255.0

        horiz_image= self.horiz_image
        vert_image = self.vert_image
        
        horiz_image = horiz_image.reshape(horiz_image.shape[0], horiz_image.shape[1], 1)
        vert_image = vert_image.reshape(vert_image.shape[0], vert_image.shape[1], 1)
        img5 = np.concatenate((img, horiz_image, vert_image), axis=-1)
        img5 = img5[-433:-10,:,:]
        img5 = cv2.resize(img5, (200,66))
        return img5

if __name__ == '__main__':
    # img = get_dep('/data/wp/data/linggusixi_2018_11_28/00/00031033.jpg')
    map = Img_map()
    img = map.get_dep('/data/wp/data/linggusixi_2018_11_28/00/00031033.jpg')
    
    print(img.shape)
    print(img[0,0,:])
    print(np.max(img[:,:,3]), np.min(img[:,:,3]))
    print(np.max(img[:,:4]), np.min(img[:,:,4]))