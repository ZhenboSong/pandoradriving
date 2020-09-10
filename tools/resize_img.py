""" 
Simple python scripts for croping and resizing images
Author: Jingkang Wang
Date: November 2017
Dependency: python-opencv
"""

import argparse
import glob
import os

import cv2
import re
import sys

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def crop(input_dir="DVR_1920x1080",
         output_dir="DVR_1080x600"):
    """
    Crop images in folders
        :param input_dir: path of input directory
        :param output_dir: path of output directory
    """
    assert os.path.exists(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # subfolders = glob.glob(os.path.join(input_dir, "*"))
    # files = glob.glob(os.path.join(input_dir, "*"))
    files = os.listdir(input_dir)
    print('file', files[:2])
    # sys.exit('exit!!')
    for filename in files:
        out_filename= os.path.join(output_dir, filename)
        print(out_filename)

        print(os.path.join(input_dir, filename))
        # crop_img(filename, out_filename)
        img = cv2.imread(os.path.join(input_dir, filename))
        print(img.shape)
        # crop_img = img[200:800, 500:1580]
        crop_img = img[197:-250, :, :]
        print(crop_img.shape)
        # sys.exit('exit')
        newimage = cv2.resize(crop_img, (200,66))
        cv2.imwrite(out_filename, newimage)
        # sys.exit('exit')
        # print(out_filename)
    # sys.exit('exit')
        # cv2.imwrite(out_filename, newimage)
        # sys.exit('exit!!')

    # for folder in subfolders:
        # new_subfolder = os.path.join(output_dir, folder[folder.rfind("/")+1:])
        # # print new_subfolder
        # if not os.path.exists(new_subfolder):
            # os.mkdir(new_subfolder)
        # files = glob.glob(os.path.join(folder, "*.jpg"))
        # # print files
        # for filename in files:
            # out_filename = os.path.join(output_dir, filename[filename.find("/")+1:])
            # print filename, out_filename
            # crop_img(filename, out_filename)
    
    
def crop_img(input_img, output_img, 
             left=500, right=1580, down=200, up=800):
    """
    Crop single image
        :param input_img: path of input image
        :param output_img: path of cropped image
        :param left, right, down, up: cropped positions
    """
    img = cv2.imread(input_img)
    crop_img = img[down:up, left:right]
    cv2.imwrite(output_img, crop_img)


def resize(input_dir="DVR_1080x600",
           output_dir="DVR_200x66"):
    """
    Resize images in folders
        :param input_dir: path of input directory
        :param output_dir: path of output directory
    """
    width = int(output_dir.split("_")[-1].split("x")[0])
    height = int(output_dir.split("_")[-1].split("x")[-1])
    print width, height
    assert os.path.exists(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subfolders = glob.glob(os.path.join(input_dir, "*"))
    for folder in subfolders:
        new_subfolder = os.path.join(output_dir, folder[folder.rfind("/")+1:])
        # print new_subfolder
        if not os.path.exists(new_subfolder):
            os.mkdir(new_subfolder)
        files = glob.glob(os.path.join(folder, "*.jpg"))
        # print files
        for filename in files:
            out_filename = os.path.join(output_dir, filename[filename.find("/")+1:])
            print filename, out_filename
            resize_img(filename, out_filename, width, height)


def resize_img(input_img, output_img, newx, newy):
    """
    Resize single image
        :param input_img: path of input image
        :param output_img: path of cropped image
        :param newx, newy: scale of resized image
    """
    img = cv2.imread(input_img)
    newimage = cv2.resize(img, (newx, newy))
    cv2.imwrite(output_img, newimage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/data/wp/Data/DBnet_data/val",
                        help='Path of input directory [default: DVR_1920x1080]')
    parser.add_argument('--output_dir', type=str, default="DVR_1080x600",
                        help='Path of input directory [default: DVR_1080x600]')
    parser.add_argument('--oper', type=str, default="crop",
                        help='Operation to conduct (crop/resize) [default: crop]')
    FLAGS = parser.parse_args()

    INPUT_DIR = FLAGS.input_dir
    OUTPUT_DIR = FLAGS.output_dir
    OPER = FLAGS.oper

    assert (os.path.exists(INPUT_DIR))
    subfolders = os.listdir(INPUT_DIR)
    subfolders.sort(key=natural_keys)
    # print(subfolders)
    # sys.exit("exit!!")
    for path in subfolders:
        if int(path)>56:
            continue
        input_dir = os.path.join(INPUT_DIR, path, 'dvr_1920x1080')
        output_dir = os.path.join(INPUT_DIR, path, 'dvr_66x200_img')
        # output_dir = os.path.join(INPUT_DIR, path, 'dvr_600x1080')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            # continue
        # if os.listdir(output_dir):
            # print(output_dir, 'have img!!')
            # continue
        crop(input_dir, output_dir)
