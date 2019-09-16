import cv2
import pickle
import os.path
import numpy as np
import time
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

t0 = time.clock()

LETTER_IMAGES_FOLDER = ["stanford_dataset/extracted_letter_images_real/extracted_1_letter", "stanford_dataset/extracted_letter_images_real/extracted_2_letter", "stanford_dataset/extracted_letter_images_real/extracted_3_letter", "stanford_dataset/extracted_letter_images_real/extracted_4_letter", "stanford_dataset/extracted_letter_images_real/extracted_5_letter"]
MODEL_FILENAME = ["stanford_models/model_real_dataset_8epoch/letter_1.hdf5", "stanford_models/model_real_dataset_8epoch/letter_2.hdf5", "stanford_models/model_real_dataset_8epoch/letter_3.hdf5", "stanford_models/model_real_dataset_8epoch/letter_4.hdf5", "stanford_models/model_real_dataset_8epoch/letter_5.hdf5"]
MODEL_LABELS_FILENAME = ["stanford_models/model_real_dataset_8epoch/letter_1.dat", "stanford_models/model_real_dataset_8epoch/letter_2.dat", "stanford_models/model_real_dataset_8epoch/letter_3.dat", "stanford_models/model_real_dataset_8epoch/letter_4.dat", "stanford_models/model_real_dataset_8epoch/letter_5.dat"]


for (i, letter_image_folder) in enumerate(LETTER_IMAGES_FOLDER):

    # initialize the data and labels
    data = []
    labels = []

    # loop over the input images
    for image_file in paths.list_images(letter_image_folder):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the letter so it fits in a 40x40 pixel box
        image = resize_to_fit(image, 40, 40)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)


    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the training data into separate train and test sets
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME[i], "wb") as f:
        pickle.dump(lb, f)

    # Build the neural network!
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(20, (3, 3), padding="same", input_shape=(40, 40, 1), activation="relu"))
    #model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(40, (3, 3), padding="same", input_shape=(40, 40, 1), activation="relu"))
    #model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Third convolutional layer with max pooling
    model.add(Conv2D(80, (3, 3), padding="same", activation="relu"))
    #model.add(Dropout(0.3))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fourth convolutional layer with max pooling
    model.add(Conv2D(40, (3, 3), padding="same", activation="relu"))
    #model.add(Dropout(0.2))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 1000 nodes
    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.2))

    # Hidden layer with 100 nodes
    #model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    #model.add(Dropout(0.2))

    # Output layer with 26 nodes (one for each possible letter we predict)
    model.add(Dense(25, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the neural network
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=25, epochs=8, verbose=1)

    # Save the trained model to disk
    model.save(MODEL_FILENAME[i])

    t1 = time.clock() - t0
    print("Time elapsed: ", t1 - t0) # CPU seconds elapsed (floating point)
