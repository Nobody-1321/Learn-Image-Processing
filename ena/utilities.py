from .image_pro import *
import cv2 as cv
import numpy as np

def create_channel_image(img, title):

    red, green, blue = channels_bgr(img)
    red_green, red_blue, green_blue = get_combine_channels_rg_rb_gb(img)

    if title == "blue channel":
        img = blue
    
    if title == "green channel":
        img = green

    if title == "red channel":
        img = red

    if title == "red-green channels":
        img = red_green
    
    if title == "red-blue channels":
        img = red_blue
    
    if title == "green-blue channels":
        img = green_blue

    return img

def get_flip_flop_flipflop(img, title):
    imgc = img.copy()
    flip, flop, flipflop = flip_flop_flipflop(img)

    if title == "Flip":
        imgc = flip
    
    if title == "Flop":
        imgc = flop
    
    if title == "Flip-Flop":
        imgc = flipflop

    return imgc

def get_rotate_image(img, title):
    imgc = img.copy()
    img1 = rotate_image(img, -90)
    img2 = rotate_image(imgc, 90)
    img3 = rotate_image(img, 180)

    if title == "-90":
        imgc = img1

    if title == "90":
        imgc = img2

    if title == "180":
        imgc = img3

    return imgc
