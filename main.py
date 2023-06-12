import numpy as np          # math
import pandas as pd         # datasets
import os                   # find files
import cv2                  # machine learning
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer

# Step 1: Data Preparation
# Function to load and preprocess images from a directory
def load_images_from_directory(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".webp"):
            image_path = os.path.join(directory, filename)
            # 0 flag will read it as a grayscale image
            image = cv2.imread(image_path, 0)
            # Preprocess the image as needed and append it to the images list
            cv2.imshow("img",image)
            cv2.waitKey(0)
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
    # If no ground truth label is found, return None0
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

            if distance < nearest_neighbor_distance and abs(h1 - h2) < h1*size_similarity_threshold:
                nearest_neighbor_distance = distance
        
        if nearest_neighbor_distance < threshold:
            filtered_boundaries.append(boundaries[i])
    return filtered_boundaries

def filter_letter_boundaries(letter_boundaries):
    filtered_boundaries = []

    min_width = 1
    max_width = 80
    min_height = 7
    max_height = 150
    min_aspect_ratio = 0
    max_aspect_ratio = 2

    min_circularity = 0.13

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

def sort_boundaries(boundaries):
    # Get unique y-positions
    y_positions = sorted(list(set(b[1] for b in boundaries)))
    if(len(boundaries) == 0):
        return []
    # Merge similar y-positions
    merged_y_positions = []
    merged_indices = []
    prev_y = y_positions[0]

    for i in range(1, len(y_positions)):
        if y_positions[i] - prev_y > 10:
            merged_y_positions.append(prev_y)
            merged_indices.append(i-1)
        prev_y = y_positions[i]
    merged_y_positions.append(prev_y)
    merged_indices.append(len(y_positions)-1)

    # Assign each boundary to its corresponding merged y-position
    merged_y_bins = []
    for i, b in enumerate(boundaries):
        y = b[1]
        for j, idx in enumerate(merged_indices):
            if y <= y_positions[idx]:
                merged_y_bins.append(merged_y_positions[j])
                break
    
    # Sort boundaries based on merged y-positions, followed by x-coordinate
    sorted_boundaries = [b for _, b in sorted(zip(merged_y_bins, boundaries), key=lambda x: (x[0], x[1][0]))]

    return sorted_boundaries

def extract_roi_hog(image, x, y, w, h):
    #print("Image shape:", image.shape)
    #print("ROI coordinates (x, y, w, h):", x, y, w, h)

    # Check if the ROI coordinates are within the image bounds
    if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
        print("ROI coordinates are outside the image bounds!")
        return None

    # Extract the region of interest (ROI) from the image based on the boundary
    roi = image[y:y+h, x:x+w]

    # Check if the ROI image is grayscale
    if len(roi.shape) == 2:
        roi_gray = roi                                    # Grayscale image
    else:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale

    # Resize ROI image to a fixed size (e.g., 64x128) compatible with HOG descriptor
    roi_resized = cv2.resize(roi_gray, (64, 128))

    # Compute the HOG features for the resized ROI
    hog = cv2.HOGDescriptor()
    features = hog.compute(roi_resized).flatten()

    return features

def show_boundaries(img, boundaries):
    length = len(boundaries)
    for (x, y, w, h) in boundaries:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #print(f"[{index}/{length}]: ({x},{y},{w},{h})")
    cv2.imshow("Boundaries", img)
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()

def get_boundaries(image, show_letter=False, show_filtered=False, show_neighbor=False, show_sorted=False):
    letter_boundaries = extract_letter_boundaries(image)
    filtered_boundaries = filter_letter_boundaries(letter_boundaries)
    neighbor_boundaries = neighbor_filter(filtered_boundaries, 50, 3)
    sorted_boundaries = sort_boundaries(neighbor_boundaries)

    if show_letter == True:
       show_boundaries(image, letter_boundaries)
    if show_filtered == True:
        show_boundaries(image, filtered_boundaries)
    if show_neighbor == True:
        show_boundaries(image, neighbor_boundaries)
    if show_sorted == True:
        show_boundaries(image, sorted_boundaries)

    return sorted_boundaries

def extraction(test=False):
    images, labels = prepare_data()

    m_training_data = []
    m_training_labels = []

    for idx, img in enumerate(images):
        boundaries = get_boundaries(img)

        # Read labels and assign characters to boundaries sequentially
        label = labels[idx].replace(" ", "")
        label_idx = 0

        for (x, y, w, h) in boundaries:
            if label_idx >= len(label):
                break

            character = label[label_idx]
            label_idx += 1

            if test:
            # Check if label index exceeds the label length
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, character, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
                cv2.imshow("Boundaries", img)
                cv2.waitKey(0)

            m_training_data.append(extract_roi_hog(img, x,y,w,h))
            m_training_labels.append(character)
            
        cv2.destroyAllWindows()
    return m_training_data, m_training_labels

# Step 3: HMM Modeling
# TODO: Design and train Hidden Markov Models (HMMs) using the extracted features.

def discretize_features(boundaries):
    discretized_features = []
    num_bins = 10
    for data in boundaries:
        bins = np.linspace(np.min(data), np.max(data), num_bins+1)
        discrete_data = np.digitize(data, bins) - 1
        discretized_features.append(discrete_data)
    
    return discretized_features

def train_hmm_model(training_data, training_labels):
    negative_count = 0
    for data in training_data:
        if np.any(data < 0):
            negative_count += 1

    print("Total arrays with negative values:", negative_count)

    print(training_labels)
    states = list(set(training_labels))

    # Discretize the observations (HOG feature vectors)
    observations = discretize_features(training_data)

    # Prepare lengths array for each sequence
    lengths = [len(obs) for obs in observations]

    # Flatten the observations and concatenate them
    flat_observations = np.concatenate(observations)

    hmm_model = hmm.CategoricalHMM(n_components=len(states), n_iter=1)
    hmm_model.fit(flat_observations.reshape(-1, 1), lengths=lengths)

    transition_matrix = hmm_model.transmat_
    emission_probabilities = hmm_model.emissionprob_

    #print("Transition Matrix:")
    #print(transition_matrix)
    #print("Emission Probabilities:")
    #print(emission_probabilities)
    return hmm_model

# Step 4: Recognition
# TODO: Given an input image, extract features from text regions.
# TODO: Use the trained HMMs and the Viterbi algorithm to decode the most probable sequence of characters or words.
# TODO: Output the recognized text.

def input_recognition(img_path):
    img = preprocess_image(cv2.imread(img_path, 0))
    boundaries = get_boundaries(img)
    return boundaries

def generate_symbol_to_label(hidden_labels):
    symbol_to_label = {i: label for i, label in enumerate(hidden_labels)}
    return symbol_to_label

def test(hmm_model, boundaries, hidden_labels):
    sym_dict = generate_symbol_to_label(hidden_labels)
    features = discretize_features(boundaries) 
    hidden_states = hmm_model.predict(features)
    print("len:",len(features))
    extracted_text = "".join(sym_dict[symbol] for symbol in hidden_states)

    return extracted_text

# Step 5: Evaluation and Refinement
# TODO: Evaluate the performance of your text recognition system using appropriate evaluation metrics.
# TODO: Iterate and refine the preprocessing, feature extraction, and HMM modeling steps to improve accuracy.

def test_training():
    directory = "./data/training_data"
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".webp"):
            image_path = os.path.join(directory, filename)
            image = preprocess_image(cv2.imread(image_path, 0))
            print(filename)
            get_boundaries(image, show_sorted=True)

# Main function to orchestrate the project
def main():
    # TODO: Implement the main workflow of your project, including loading data, preprocessing, feature extraction, HMM training, recognition, evaluation, and refinement.
    #test_training()
    t_d, t_l = extraction()
    hmm = train_hmm_model(t_d, t_l)
    test_features = input_recognition("./data/test_data/0.jpeg")
    print("test features:",test_features)
    print("Doing text extraction !")
    result = test(hmm, test_features, t_l)
    print("Result:",result)

if __name__ == "__main__":
    main()