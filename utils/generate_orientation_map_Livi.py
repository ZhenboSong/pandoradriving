import numpy as np
import cv2
import matplotlib.pyplot as plt

input_height = h = 66
input_width = w = 200
horizontal_view_angle = 60.
cx = input_width / 2
cy = input_height / 2
fx = fy = cx / np.tan(np.pi * horizontal_view_angle/2./180.)
K = [fx, fy, cx, cy]
def get_dep():
    horiz_image = np.arange(0, w).repeat(h).reshape(w, h).T.astype(np.float)
    vert_image = np.arange(0, h).repeat(w).reshape(h, w).astype(np.float)
    horiz_image = np.arctan2(horiz_image - cx, fx)*180./np.pi
    vert_image = np.arctan2(vert_image - cy, fy)*180./np.pi
    
    horiz_image = (horiz_image - np.min(horiz_image))/(np.max(horiz_image) - np.min(horiz_image))
    vert_image = (vert_image - np.min(vert_image)) / (np.max(vert_image) - np.min(vert_image))
    
    horiz_image = horiz_image.reshape(horiz_image.shape[0], horiz_image.shape[1], 1)
    vert_image = vert_image.reshape(vert_image.shape[0], vert_image.shape[1], 1)
    
    return horiz_image, vert_image

if __name__ == '__main__':
    horiz_image, vert_image = get_dep()
    
    # print(img.shape)
    print(horiz_image[0,:])
    print(vert_image[0,0])
    print(np.max(horiz_image), np.min(horiz_image))
    print(np.max(vert_image), np.min(vert_image))