import os
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
import random

# Function to read PGM images from a folder
def read_pgm_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pgm"):
            # Read image as grayscale
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            images.append(img)
    return images

# Function to create a data matrix from a list of images
def create_data_matrix(images):
    num_images = len(images)
    sz = images[0].shape
    # Initialize data matrix with zeros
    data = np.zeros((num_images, sz[0] * sz[1]), dtype=np.float32)
    for i in range(0, num_images):
        # Flatten the image and store it as a row in the data matrix
        image = np.array(images[i].flatten(), dtype=np.float32)
        data[i, :] = image
    return data

# Function to compute the Eigenfaces, mean face, and centered images
def eigenfaces(images, num_components):
    # Create the data matrix from the images
    data = create_data_matrix(images)
    # Compute the mean face
    mean_image = np.mean(data, axis=0)
    mean_image = mean_image.reshape(-1, 1)

    # Subtract the mean face from the original images to center the data
    centered_data = data - mean_image.T

    # Compute the covariance matrix
    covariance_matrix = np.dot(centered_data, centered_data.T)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Convert eigenvectors from the smaller matrix to the original eigenspace
    eigenvectors = np.dot(centered_data.T, eigenvectors)
    # Normalize the eigenvectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    # Sort the eigenvectors by their corresponding eigenvalues in descending order
    sorted_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Return the mean face, centered data, and top k eigenvectors
    return mean_image, centered_data, eigenvectors[:, :num_components]

# Function to project the data onto the eigenvector space
def project_data(data, eigenvectors):
    return np.dot(data, eigenvectors)

# Function to save the Eigenfaces as PGM files
def save_eigenfaces(eigenvectors, num_eigenfaces, image_shape):
    for i in range(num_eigenfaces):
        # Reshape the eigenvector to the original image shape
        eigenface = eigenvectors[:, i].reshape(image_shape)
        # Save the Eigenface as a PGM file
        cv2.imwrite(f"eigenface_{i + 1}.pgm", eigenface)

# Function to compute the nearest neighbor classifier's accuracy
def compute_accuracy(training_images, test_images, mean_face, eigenvectors):
    correct_predictions = 0
    total_predictions = len(test_images)

    # Create centered training images and project them onto the eigenvector space
    centered_training_images = create_data_matrix(training_images) - mean_face.T
    training_image_coordinates = project_data(centered_training_images, eigenvectors)

    for i, test_image in enumerate(test_images):
        # Flatten the test image and convert its data type
        test_image_vector = test_image.flatten().astype(np.float32)
        # Subtract the mean face from the test image
        centered_test_image = test_image_vector - mean_face.T
        # Project the test image onto the eigenvector space
        test_image_coordinates = project_data(centered_test_image, eigenvectors)

        min_distance = float("inf")
        min_index = -1
        # Find the closest training image in the eigenvector space
        for j, training_image_coord in enumerate(training_image_coordinates):
            distance = euclidean(training_image_coord, test_image_coordinates)
            if distance < min_distance:
                min_distance = distance
                min_index = j

        # Compare the test image index with the closest training image index
        if i == min_index:
            correct_predictions += 1

    # Calculate the accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    # Load training and test images
    training_images = read_pgm_images("attDataset/train")
    test_images = read_pgm_images("attDataset/test")

    num_eigenfaces = 20

    # Compute the Eigenfaces, mean face, and centered training images
    mean_face, centered_training_images, eigenvectors = eigenfaces(training_images, num_eigenfaces)
    # Save the mean face as a PGM file
    cv2.imwrite("average_face.pgm", mean_face.reshape(training_images[0].shape))

    # Save the top 20 Eigenfaces as PGM files
    save_eigenfaces(eigenvectors, num_eigenfaces, training_images[0].shape)

    # Compute and print the accuracy using the top 20 Eigenfaces
    accuracy = compute_accuracy(training_images, test_images, mean_face, eigenvectors)
    print(f"Accuracy using top 20 eigenfaces: {accuracy * 100:.2f}%")

    # Compute the accuracy using 20 random Eigenfaces
    random_indices = random.sample(range(eigenvectors.shape[1]), num_eigenfaces)
    print(f"Selected random eigenface indices: {random_indices}")
    random_eigenvectors = eigenvectors[:, random_indices]
    random_accuracy = compute_accuracy(training_images, test_images, mean_face, random_eigenvectors)
    print(f"Accuracy using 20 random eigenfaces: {random_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
