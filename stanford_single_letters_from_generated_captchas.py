import os
import os.path
import cv2
import glob
import imutils
import pytesseract
import sys
import argparse
import cv2
import PIL
import numpy as np

from PIL import Image, ImageDraw, ImageFont


CAPTCHA_IMAGE_FOLDER = "stanford_dataset/renamed"
OUTPUT_FOLDER = ["stanford_dataset/extracted_letter_images_real/extracted_1_letter", "stanford_dataset/extracted_letter_images_real/extracted_2_letter", "stanford_dataset/extracted_letter_images_real/extracted_3_letter", "stanford_dataset/extracted_letter_images_real/extracted_4_letter", "stanford_dataset/extracted_letter_images_real/extracted_5_letter"]

#CAPTCHA_IMAGE_FOLDER = "dataset/test"
#OUTPUT_FOLDER = "dataset/test"

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    #print(filename)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the full image
    complete_path = CAPTCHA_IMAGE_FOLDER + '/' + filename
    image = Image.open(complete_path)

    # characteristic before cropping
    w = 22 # wide of a letter
    left, upper, right, lower = 5, 25, 45, 65

    # Save out each letter as a single image
    for (k, letter_text) in enumerate(captcha_correct_text):

        try:
            int(letter_text)

        except:

            if len(captcha_correct_text) == 5:

                letter_image = image.crop((left, upper, right, lower))
                left += w
                right += w

                # Get the folder to save the image in
                save_path = os.path.join(OUTPUT_FOLDER[k], letter_text)

                # if the output directory does not exist, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # write the letter image to a file
                count = counts.get(letter_text, 1)
                p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
                letter_image.save(p)

                # increment the count for the current key
                counts[letter_text] = count + 1
