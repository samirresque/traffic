import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Invalid Command line arguments.")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # print the model

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    images = []
    labels = []

    for category in range(NUM_CATEGORIES):  # 0-42
        # workspace/13.../gtsrb/0
        category_path = os.path.join(data_dir, str(category))
        for image in os.listdir(category_path):  # image = actual images
            image_path = os.path.join(category_path, image)
            img = cv2.imread(image_path)

            # test if resizing can go wrong.
            try:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            except:
                break

            images.append(img)
            labels.append(category)

    return (images, labels)


def get_model():

    model = tf.keras.models.Sequential([

        # first Convolutional layer: Learn 32 filters using a 3x3 kernel with maxpool (3x3)
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            # strides=(2, 2),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # second convol layer: Learn 32 filters using a 3x3 kernel no maxpool
        tf.keras.layers.Conv2D(
            32,
            (4, 4),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # third convol layer: Learn 32 filters using a 3x3 kernel with maxpool (3,3)
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # 1st dense layer with dropout= 0.5
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # 2nd dense layer with dropout = 0.3
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),


        # Add an output layer with output units for all 43 different categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
