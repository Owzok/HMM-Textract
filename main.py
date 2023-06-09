import numpy as np          # math
import pandas as pd         # datasets
import os                   # find files
import cv2                  # machine learning

# Step 1: Data Preparation

# Function to load and preprocess images from a directory
def load_images_from_directory(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            # 0 flag will read it as a grayscale image
            image = cv2.imread(image_path, 0)
            # Preprocess the image as needed and append it to the images list
            images.append(preprocess_image(image))
            labels.append(get_ground_truth_label(filename))
    return images, labels

def resize_image(image, target_width, target_height):
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / float(original_height)

    # Calculate the scaling factor for resizing
    if original_width > original_height:
        scaling_factor = target_width / float(original_width)
    else:
        scaling_factor = target_height / float(original_height)

    # Resize the image while maintaining the aspect ratio
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Add padding to match the target dimensions
    top_padding = (target_height - new_height) // 2
    bottom_padding = target_height - new_height - top_padding
    left_padding = (target_width - new_width) // 2
    right_padding = target_width - new_width - left_padding
    resized_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, left_padding, right_padding,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return resized_image

# Function to preprocess the image
def preprocess_image(image):
    # TODO: Implement preprocessing techniques (e.g., resizing, noise removal, contrast adjustment, binarization, etc.)
    cv2.imshow("Original Image", image)
    # Resizing
    new_width = 600
    new_height = 600

    resized_image = resize_image(image, new_width, new_height)

    # Median filter for salt-and-pepper noise removal
#   denoised_image = cv2.medianBlur(image, 3)

    # Contrast adjustment (intensity scaling)
    min_intensity = image.min()
    max_intensity = image.max()
    new_min = 0
    new_max = 255
    adjusted_image = cv2.convertScaleAbs(resized_image, alpha=(new_max - new_min) / (max_intensity - min_intensity), beta=new_min - min_intensity)
    
    # We could use too, histogram equalization
#   equalized_image = cv2.equalizeHist(image)

    # Adaptive tresholding
    preprocessed_image = cv2.adaptiveThreshold(adjusted_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    cv2.imshow("Processed image", preprocessed_image)
    cv2.waitKey(0)

    # Return the preprocessed image
    return preprocessed_image

# Function to get the ground truth label for an image
def get_ground_truth_label(filename):
    # Extract the base filename without the extension
    base_filename = filename.split(".")[0]

    # Assuming the ground truth labels are stored in a separate file named "ground_truth.txt"
    ground_truth_file = "./data/labels.txt"

    # Load the ground truth labels from the file
    with open(ground_truth_file, "r") as file:
        lines = file.readlines()

        # Search for the corresponding ground truth label
        for line in lines:
            parts = line.split("-")
            if parts[0] == base_filename:
                ground_truth_label = parts[1].strip()
                return ground_truth_label

    raise Exception("Ground truth label is not found for " + filename)
    # If no ground truth label is found, return None
    return None



def prepare_data():
    dataset_directory = "./data"
    images, labels = load_images_from_directory(dataset_directory)
    
    # TODO: Further processing or saving of the images and labels as needed

    print("Number of images:", len(images))
    print("Number of labels:", len(labels))

# Step 2: Preprocessing
# TODO: Implement image preprocessing techniques like resizing, noise removal, contrast adjustment, and binarization.

# Step 3: Feature Extraction
# TODO: Implement feature extraction methods like HOG, SIFT, or CNNs to extract relevant features from the preprocessed images.

# Step 4: HMM Modeling
# TODO: Design and train Hidden Markov Models (HMMs) using the extracted features.

# Step 5: Recognition
# TODO: Given an input image, extract features from text regions.
# TODO: Use the trained HMMs and the Viterbi algorithm to decode the most probable sequence of characters or words.
# TODO: Output the recognized text.

# Step 6: Evaluation and Refinement
# TODO: Evaluate the performance of your text recognition system using appropriate evaluation metrics.
# TODO: Iterate and refine the preprocessing, feature extraction, and HMM modeling steps to improve accuracy.

# Main function to orchestrate the project
def main():
    # TODO: Implement the main workflow of your project, including loading data, preprocessing, feature extraction, HMM training, recognition, evaluation, and refinement.
    prepare_data()

if __name__ == "__main__":
    main()