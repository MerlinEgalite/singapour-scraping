import pytesseract
import sys
import argparse
import cv2
import PIL
import numpy as np

import matplotlib.pyplot as plt

"""try:
    import Image
except ImportError:
    from PIL import Image"""

from PIL import Image, ImageDraw, ImageFont

#1 Preprocessing
#2

def threshold(file_path, new_file_path, threshold):
    img = cv2.imread(file_path, 0)
    ret, new_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(new_file_path, new_image)

#30 pour les captchas

img = cv2.imread('/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/captcha_images/cap11.png', 0)
ret, new_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite('/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/captcha_images_reprocessed/PZZVU.png', new_image)


def crop_image_generated_captcha(file_path, dataset_path):
    image = Image.open(file_path)
    print(image.size)
    text = file_path[-9:-4]
    w = 24
    left, upper, right, lower = 12, 25, 35, 65
    #image.crop((left, upper, right, lower)).save(dataset_path + text[0] + '.png')
    for i in range(len(text)):
        image.crop((left, upper, right, lower)).save(dataset_path + text[i] + '.png')
        left += w
        right += w


def crop_image_captcha(file_path, dataset_path):
    image = Image.open(file_path)
    print(image.size)
    text = file_path[-9:-4]
    w = 22
    left, upper, right, lower = 5, 25, 45, 65
    #image.crop((left, upper, right, lower)).save(dataset_path + text[0] + '.png')
    for i in range(len(text)):
        image.crop((left, upper, right, lower)).save(dataset_path + text[i] + '.png')
        left += w
        right += w


#threshold('/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/data/KUFYG.png', '/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/data/NEW.png', 160)

crop_image_captcha('/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/stanford_dataset/renamed/QPRLG.png', '/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/dataset/test/')

#crop_image_captcha('/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/Convert/YDMQS.png', '/Users/merlinegalite/Desktop/octobot/Scraping/CaptchaResolve/dataset/test')
