import os
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

# Directory path to images
images_dir = './data/images'

# Get all image file paths (assuming .jpg format)
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Verify we have at least 1000 images
if len(image_files) < 1000:
    print(f"Warning: Found only {len(image_files)} images. At least 1000 are expected.")
else:
    print(f"Found {len(image_files)} images. Proceeding with train-test split...")

# Function to load and preprocess an image (RGB to HSV, extract V channel)
def load_and_preprocess_image(file_path):
    img = io.imread(file_path) / 255.0  # Normalize to [0,1]
    img_hsv = color.rgb2hsv(img)       # Convert to HSV
    v_channel = img_hsv[:, :, 2]       # Extract V channel
    return v_channel.flatten()          # Flatten to 16,384-element vector

# Load all images
print("Loading all images...")
X_all = np.array([load_and_preprocess_image(f) for f in image_files]).T  # Shape: (16384, n_images)

# Randomly shuffle and split into train (800) and test (200) sets
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(X_all.shape[1])
train_indices = indices[:800]
test_indices = indices[800:1000]

X_train = X_all[:, train_indices]  # Shape: (16384, 800)
X_test = X_all[:, test_indices]    # Shape: (16384, 200)

# Compute the mean face from training data
mean_face = np.mean(X_train, axis=1)  # Shape: (16384,)

# Center the data by subtracting the mean face
X_train_centered = X_train - mean_face[:, np.newaxis]
X_test_centered = X_test - mean_face[:, np.newaxis]

# Perform SVD on centered training data to get eigenfaces
print("Computing SVD...")
U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)  # U: (16384, 800), eigenfaces are columns of U

# Function to normalize images for display (scale to [0,1])
def normalize_for_display(img):
    img = img - np.min(img)
    img = img / np.max(img)
    return img

# Display the first 10 eigenfaces
plt.figure(figsize=(10, 5))
for i in range(10):
    eigenface = U[:, i].reshape(128, 128)          # Reshape to 128x128
    eigenface_display = normalize_for_display(eigenface)  # Normalize for visualization
    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenface_display, cmap='gray')
    plt.title(f'Eigenface {i + 1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('results/eigenfaces.png')

# Function to reconstruct images using top K eigenfaces
def reconstruct(X_centered, U, K):
    coeffs = U[:, :K].T @ X_centered              # Project onto top K eigenfaces
    X_reconstructed = U[:, :K] @ coeffs           # Reconstruct
    return X_reconstructed

# Reconstruct test images with K=50
K = 50
X_test_reconstructed = reconstruct(X_test_centered, U, K) + mean_face[:, np.newaxis]

# Display 10 original and reconstructed test images
plt.figure(figsize=(15, 6))
for i in range(10):
    original = X_test[:, i].reshape(128, 128)          # Original image
    reconstructed = X_test_reconstructed[:, i].reshape(128, 128)  # Reconstructed image
    plt.subplot(2, 10, i + 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
plt.tight_layout()
plt.savefig('results/reconstructed_images.png')

# Compute and plot reconstruction error for K = 1, 5, 10, ..., 50
K_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
errors = []
print("Computing reconstruction errors...")
for K in K_values:
    X_test_reconstructed = reconstruct(X_test_centered, U, K) + mean_face[:, np.newaxis]
    error = np.mean(np.sum((X_test - X_test_reconstructed) ** 2, axis=0) / 16384)  # Average error per pixel
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(K_values, errors, marker='o')
plt.xlabel('Number of Eigenfaces (K)')
plt.ylabel('Average Reconstruction Error per Pixel')
plt.title('Reconstruction Error vs. Number of Eigenfaces')
plt.grid(True)

plt.savefig('results/reconstruction_error.png')