
# ================================================================
# CSCI 441/541 - Research Assignment
# Project: Deepfake Image Detection Using CNN
#
# Domain: Cybersecurity and Digital Image Forensics
# Research Focus: Binary Image Classification
#
# Objective:
# Train a Convolutional Neural Network (CNN) to classify facial
# images into two categories: Real and Fake.
#
# Dataset:
# https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
# ================================================================


# ================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ================================================================

# Path helps us work with folders and image files.
from pathlib import Path

# Random is used to select images reproducibly.
import random

# CSV is used to save training history and predictions.
import csv

# NumPy provides numerical operations.
import numpy as np

# Matplotlib creates accuracy and loss graphs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow is used to build, train, and test our CNN.
import tensorflow as tf


# ================================================================
# 2. CONFIGURE PROJECT SETTINGS
# ================================================================

# Set a random seed to make image selection reproducible.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Location of the dataset in our Codespaces project.
DATASET_PATH = Path("Dataset")

# Create a folder to save the trained model and results.
RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)

# Resize all images to 128 x 128 pixels.
IMAGE_SIZE = (128, 128)

# Number of images processed in each training batch.
BATCH_SIZE = 32

# Number of times the CNN will go through the training data.
EPOCHS = 8

# Define the two image categories.
# Fake = 0 and Real = 1.
CLASS_NAMES = ["Fake", "Real"]

# We are using a balanced subset of the full dataset.
# The original dataset remains unchanged.

TRAIN_IMAGES_PER_CLASS = 2000
VAL_IMAGES_PER_CLASS = 500
TEST_IMAGES_PER_CLASS = 500


# ================================================================
# 3. SELECT IMAGES FROM THE DATASET
# ================================================================

def select_images(split_name, images_per_class):
    """
    Select an equal number of Real and Fake images from one split.

    Parameters:
        split_name: Train, Validation, or Test
        images_per_class: Number of images from each category

    Returns:
        List of image paths and corresponding labels
    """

    image_paths = []
    image_labels = []

    # Process the Fake and Real folders separately.
    for class_index, class_name in enumerate(CLASS_NAMES):

        # Example: Dataset/Train/Fake
        folder_path = DATASET_PATH / split_name / class_name

        # Confirm that the expected folder exists.
        if not folder_path.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder_path}"
            )

        # Collect image files from the selected folder.
        images = sorted([
            file for file in folder_path.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        # Randomly select images using a fixed seed.
        rng = random.Random(SEED)
        rng.shuffle(images)

        # Check that enough images are available.
        if len(images) < images_per_class:
            raise ValueError(
                f"Not enough images in {folder_path}"
            )

        selected_images = images[:images_per_class]

        # Store the image paths and numerical class labels.
        for image_path in selected_images:
            image_paths.append(str(image_path))
            image_labels.append(class_index)

    # Shuffle image paths and labels together.
    combined = list(zip(image_paths, image_labels))
    random.Random(SEED).shuffle(combined)

    image_paths, image_labels = zip(*combined)

    return list(image_paths), list(image_labels)


# ================================================================
# 4. PREPROCESS THE IMAGES
# ================================================================

def preprocess_image(image_path, label):
    """
    Read an image, resize it, and normalize pixel values.

    All images are converted to RGB and resized to 128 x 128.
    Pixel values are normalized from 0-255 to 0-1.
    """

    # Read the image file.
    image = tf.io.read_file(image_path)

    # Decode the image and convert it to three RGB channels.
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Specify the expected image tensor shape.
    image.set_shape([None, None, 3])

    # Resize the image.
    image = tf.image.resize(image, IMAGE_SIZE)

    # Normalize pixel values between 0 and 1.
    image = tf.cast(image, tf.float32) / 255.0

    # Convert the label into a floating-point number.
    label = tf.cast(label, tf.float32)

    return image, label


# ================================================================
# 5. CREATE TENSORFLOW DATASETS
# ================================================================

def create_dataset(image_paths, image_labels):
    """
    Create a TensorFlow dataset for efficient image processing.
    """

    # Convert file paths and labels into a TensorFlow dataset.
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, image_labels)
    )

    # Apply image preprocessing.
    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=2
    )

    # Group images into batches.
    dataset = dataset.batch(BATCH_SIZE)

    # Prepare the next batch while the model processes one.
    dataset = dataset.prefetch(1)

    return dataset


# ================================================================
# 6. LOAD TRAINING, VALIDATION, AND TESTING DATA
# ================================================================

print("\nLoading Deepfake Dataset...")

# Select 4,000 training images.
train_paths, train_labels = select_images(
    "Train",
    TRAIN_IMAGES_PER_CLASS
)

# Select 1,000 validation images.
val_paths, val_labels = select_images(
    "Validation",
    VAL_IMAGES_PER_CLASS
)

# Select 1,000 testing images.
# These are not used during model training.
test_paths, test_labels = select_images(
    "Test",
    TEST_IMAGES_PER_CLASS
)

# Convert the selected images into TensorFlow datasets.
train_dataset = create_dataset(train_paths, train_labels)
val_dataset = create_dataset(val_paths, val_labels)
test_dataset = create_dataset(test_paths, test_labels)

# Display the dataset sizes.
print("\nDataset Summary")
print("-------------------------------")
print("Training Images:", len(train_paths))
print("Validation Images:", len(val_paths))
print("Testing Images:", len(test_paths))
print("Total Images:",
      len(train_paths) + len(val_paths) + len(test_paths))


# ================================================================
# 7. BUILD THE CONVOLUTIONAL NEURAL NETWORK
# ================================================================

print("\nBuilding CNN Model...")

# A Sequential model processes images layer by layer.
model = tf.keras.Sequential([

    # Input layer: 128 x 128 RGB images.
    tf.keras.layers.Input(shape=(128, 128, 3)),

    # First convolutional layer:
    # Learns simple features such as edges and textures.
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Reduces the spatial dimensions of feature maps.
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolutional layer:
    # Learns more detailed image patterns.
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolutional layer:
    # Learns more complex facial image features.
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Summarizes each feature map into a single value.
    tf.keras.layers.GlobalAveragePooling2D(),

    # Fully connected layer for learning feature combinations.
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    # Randomly disables 30% of neurons during training.
    # This can help reduce overfitting.
    tf.keras.layers.Dropout(0.3),

    # Output layer:
    # Sigmoid produces a probability between 0 and 1.
    # Values >= 0.5 indicate Real; otherwise Fake.
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ================================================================
# 8. COMPILE THE CNN MODEL
# ================================================================

# Adam adjusts the network weights during training.
# Binary cross-entropy is appropriate for two classes.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display the CNN architecture and parameter counts.
model.summary()


# ================================================================
# 9. TRAIN THE CNN MODEL
# ================================================================

print("\nStarting CNN Training...")

# Train using the training images.
# Evaluate validation accuracy after each epoch.
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

print("\nCNN Training Completed!")


# ================================================================
# 10. TEST THE CNN ON UNSEEN IMAGES
# ================================================================

print("\nEvaluating CNN on Unseen Test Images...")

# Evaluate performance using the separate test dataset.
test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

# Display the final test results.
print("\nFinal Test Results")
print("-------------------------------")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# ================================================================
# 11. SAVE THE TRAINED CNN MODEL
# ================================================================

# Save the model for future predictions without retraining.
model.save(
    str(RESULTS_PATH / "deepfake_cnn.keras")
)

print("\nTrained model saved successfully.")


# ================================================================
# 12. SAVE TRAINING HISTORY
# ================================================================

# Save accuracy and loss for every epoch.
history_file = RESULTS_PATH / "training_history.csv"

with open(history_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Training Accuracy",
        "Validation Accuracy",
        "Training Loss",
        "Validation Loss"
    ])

    for epoch in range(EPOCHS):

        writer.writerow([
            epoch + 1,
            history.history["accuracy"][epoch],
            history.history["val_accuracy"][epoch],
            history.history["loss"][epoch],
            history.history["val_loss"][epoch]
        ])


# ================================================================
# 13. GENERATE TRAINING AND VALIDATION ACCURACY GRAPH
# ================================================================

# Compare training accuracy with validation accuracy.
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the graph instead of opening a window.
plt.savefig(
    RESULTS_PATH / "accuracy_graph.png"
)

plt.close()


# ================================================================
# 14. GENERATE TRAINING AND VALIDATION LOSS GRAPH
# ================================================================

# Compare training loss with validation loss.
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "loss_graph.png"
)

plt.close()


# ================================================================
# 15. TEST INDIVIDUAL UNSEEN IMAGES
# ================================================================

print("\nPredicting 10 Unseen Test Images...")

# Create a CSV file to document model predictions.
predictions_file = RESULTS_PATH / "predictions.csv"

with open(predictions_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image Path",
        "Actual Label",
        "Predicted Label",
        "Probability of Real",
        "Correct Prediction"
    ])

    # Test 10 images that were not used for training.
    for index in range(10):

        image_path = test_paths[index]
        actual_label = test_labels[index]

        # Apply the same preprocessing used for training.
        image, _ = preprocess_image(
            tf.constant(image_path),
            actual_label
        )

        # Add a batch dimension so the CNN can predict.
        image = tf.expand_dims(image, axis=0)

        # Obtain the CNN prediction probability.
        prediction = model.predict(
            image,
            verbose=0
        )[0][0]

        # Convert probability into a class label.
        predicted_label = 1 if prediction >= 0.5 else 0

        # Check whether the prediction matches the true label.
        is_correct = predicted_label == actual_label

        # Save the prediction to the CSV file.
        writer.writerow([
            image_path,
            CLASS_NAMES[actual_label],
            CLASS_NAMES[predicted_label],
            float(prediction),
            is_correct
        ])

        # Display the result in the terminal.
        print(
            f"Image {index + 1}: "
            f"Actual = {CLASS_NAMES[actual_label]}, "
            f"Predicted = {CLASS_NAMES[predicted_label]}, "
            f"Correct = {is_correct}"
        )


# ================================================================
# 16. SAVE FINAL TEST METRICS
# ================================================================

# Save the measured test results for the assignment report.
with open(RESULTS_PATH / "test_results.txt", "w") as file:

    file.write("Deepfake Image Detection - CNN Results\n")
    file.write("-----------------------------------\n")

    file.write(
        f"Training Images: {len(train_paths)}\n"
    )

    file.write(
        f"Validation Images: {len(val_paths)}\n"
    )

    file.write(
        f"Testing Images: {len(test_paths)}\n"
    )

    file.write(
        f"Final Training Accuracy: "
        f"{history.history['accuracy'][-1] * 100:.2f}%\n"
    )

    file.write(
        f"Final Validation Accuracy: "
        f"{history.history['val_accuracy'][-1] * 100:.2f}%\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )


# ================================================================
# 17. DISPLAY COMPLETION MESSAGE
# ================================================================

print("Results saved in:", RESULTS_PATH)
print("\nGenerated Files:")
print("1. deepfake_cnn.keras")
print("2. training_history.csv")
print("3. accuracy_graph.png")
print("4. loss_graph.png")
print("5. predictions.csv")
print("6. test_results.txt")
