from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import os
import imutils
import cv2
import pickle

from time import time

from PIL import Image, ImageDraw, ImageFont

MODEL_FILENAME = ["stanford_models/model_real_dataset_8epoch/letter_1.hdf5", "stanford_models/model_real_dataset_8epoch/letter_2.hdf5", "stanford_models/model_real_dataset_8epoch/letter_3.hdf5", "stanford_models/model_real_dataset_8epoch/letter_4.hdf5", "stanford_models/model_real_dataset_8epoch/letter_5.hdf5"]
MODEL_LABELS_FILENAME = ["stanford_models/model_real_dataset_8epoch/letter_1.dat", "stanford_models/model_real_dataset_8epoch/letter_2.dat", "stanford_models/model_real_dataset_8epoch/letter_3.dat", "stanford_models/model_real_dataset_8epoch/letter_4.dat", "stanford_models/model_real_dataset_8epoch/letter_5.dat"]

CAPTCHA_IMAGE_FOLDER = "captcha_images_reprocessed"

labels = []
models = []

number = 0

for i in range(len(MODEL_LABELS_FILENAME)):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(MODEL_LABELS_FILENAME[i], "rb") as f:
        labels.append(pickle.load(f))

    # Load the trained neural network
    models.append(load_model(MODEL_FILENAME[i]))


# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
#captcha_image_files = np.random.choice(captcha_image_files, size=(5,), replace=False)

# loop over the image paths
for image_file in captcha_image_files:

    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    #image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #cv2.imshow("threshold", image)

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # characteristic before cropping
    w = 22
    #left, upper, right, lower = 14, 27, 38, 60
    left, upper, right, lower = 5, 25, 45, 65

    # loop over the letters
    for i in range(5):

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[upper:lower, left:right]
        #cv2.imshow("letter_image", letter_image)
        #cv2.waitKey()

        # Re-size the letter image to 40x40 pixels to match training data
        letter_image = resize_to_fit(letter_image, 40, 40)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = models[i].predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = labels[i].inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (left , upper ), (right, lower ), (0, 255, 0), 1)
        cv2.putText(output, letter, (left - 5, upper - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # add the wide of a single letter
        left += w
        right += w

    filename = os.path.basename(image_file)
    #print(filename)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {} while it is : {}".format(captcha_text, captcha_correct_text))

    for i in range(5):
        if captcha_text[i] == captcha_correct_text[i]:
            number += 1

    # Show the annotated image
    #cv2.imshow("Output", output)
    #cv2.waitKey()

print('Number of correct letters : {} '.format(number))
