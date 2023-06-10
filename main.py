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
#cv2.imshow("Original Image", image)
    # Resizing
    new_width = 400
    new_height = 400

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
    preprocessed_image = cv2.adaptiveThreshold(adjusted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 8)
    
#cv2.imshow("Processed image", preprocessed_image)
#cv2.waitKey(0)

    # Return the preprocessed image
    return preprocessed_image

# Function to get the ground truth label for an image
def get_ground_truth_label(filename):
    # Extract the base filename without the extension
    base_filename = filename.split(".")[0]

    # Assuming the ground truth labels are stored in a separate file named "ground_truth.txt"
    ground_truth_file = "./data/training_data/labels.txt"

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
    dataset_directory = "./data/training_data"
    images, labels = load_images_from_directory(dataset_directory)
    
    # TODO: Further processing or saving of the images and labels as needed

    print("Number of images:", len(images))
    print("Number of labels:", len(labels))
    return images, labels

# Step 2: Feature Extraction
# TODO: Implement feature extraction methods like HOG, SIFT, or CNNs to extract relevant features from the preprocessed images.

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    return features.flatten()

def neighbor_filter(boundaries, threshold, size_similarity_threshold):
    num_boundaries = len(boundaries)
    filtered_boundaries = []
    for i in range(num_boundaries):
        x1, y1, w1, h1 = boundaries[i]
        center1 = (x1 + w1 // 2, y1 + h1 //2)
        nearest_neighbor_distance = np.inf

        for j in range(num_boundaries):
            if i == j:
                continue
            x2, y2, w2, h2 = boundaries[j]
            center2 = (x2 + w2 // 2, y2 + h2 // 2)
            distance = np.linalg.norm(np.array(center1) - np.array(center2))

            if distance < nearest_neighbor_distance  and abs(w1 - w2) < size_similarity_threshold and abs(h1 - h2) < size_similarity_threshold:
                nearest_neighbor_distance = distance
        
        if nearest_neighbor_distance < threshold:
            filtered_boundaries.append((boundaries[i], nearest_neighbor_distance))
    return filtered_boundaries

def filter_letter_boundaries(letter_boundaries):
    filtered_boundaries = []

    min_width = 2
    max_width = 50
    min_height = 12
    max_height = 130
    min_aspect_ratio = 0.11
    max_aspect_ratio = 1.7

    min_circularity = 0.2

    for boundary in letter_boundaries:
        x, y, w, h = boundary
        aspect_ratio = float(w) / h
        if w > min_width and w < max_width and h > min_height and h < max_height and aspect_ratio > min_aspect_ratio and aspect_ratio < max_aspect_ratio:
            contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            contour_area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
            if circularity > min_circularity:
                filtered_boundaries.append(boundary)
    return filtered_boundaries

def extract_letter_boundaries(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundaries = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boundaries.append((x,y,w,h))
    return boundaries
 
def extraction():
    images, labels = prepare_data()
    for img in images:
        hog_features = extract_hog_features(img)
        letter_boundaries = extract_letter_boundaries(img)
        filtered_boundaries = filter_letter_boundaries(letter_boundaries)
        neighbor_boundaries = neighbor_filter(filtered_boundaries, 27, 20)

        print("HOG Features shape:", hog_features.shape)
        print("Letter Boundaries:", filtered_boundaries)
        
        for (x,y,w,h), dist in neighbor_boundaries:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow("Boundaries", img) 
            print(f"nearest_distance: {dist}")
            print(f"dimensions: {w}, {h}")
            print(f"aspect ratio is:", float(w)/h)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

# Step 3: HMM Modeling
# TODO: Design and train Hidden Markov Models (HMMs) using the extracted features.

# Step 4: Recognition
# TODO: Given an input image, extract features from text regions.
# TODO: Use the trained HMMs and the Viterbi algorithm to decode the most probable sequence of characters or words.
# TODO: Output the recognized text.

# Step 5: Evaluation and Refinement
# TODO: Evaluate the performance of your text recognition system using appropriate evaluation metrics.
# TODO: Iterate and refine the preprocessing, feature extraction, and HMM modeling steps to improve accuracy.

# Main function to orchestrate the project
def main():
    # TODO: Implement the main workflow of your project, including loading data, preprocessing, feature extraction, HMM training, recognition, evaluation, and refinement.
    extraction()

if __name__ == "__main__":
    main()