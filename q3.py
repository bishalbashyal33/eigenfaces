import os
import numpy as np
from scipy.io import loadmat
from skimage import io, color
import matplotlib.pyplot as plt
import mywarper  # Assuming mywarper.py is in the same directory with warp function

# Directory paths
images_dir = './data/images'
landmarks_dir = './data/landmarks'
output_dir = './results/3'
os.makedirs(output_dir, exist_ok=True)  # Create figures directory if it doesn't exist

# Get all image and landmark file paths
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
landmark_files = [os.path.join(landmarks_dir, f) for f in os.listdir(landmarks_dir) if f.endswith('.mat')]

# Verify we have at least 1000 pairs of files
if len(image_files) < 1000 or len(landmark_files) < 1000:
    print(f"Warning: Found {len(image_files)} images or {len(landmark_files)} landmarks. At least 1000 of each are expected.")
else:
    print(f"Found {len(image_files)} images and {len(landmark_files)} landmarks. Proceeding with train-test split...")

# Function to load and preprocess an image (RGB to HSV, extract V channel, convert to 3-channel)
def load_and_preprocess_image(file_path):
    img = io.imread(file_path) / 255.0  # Normalize to [0,1]
    img_hsv = color.rgb2hsv(img)       # Convert to HSV
    v_channel = img_hsv[:, :, 2]       # Extract V channel
    # Replicate V-channel to create a 3-channel image compatible with warp
    return np.stack([v_channel] * 3, axis=-1)  # Shape: (128, 128, 3)

# Function to load landmarks from .mat files (with y-coordinate flipping)
def load_landmarks(file_path):
    mat_data = loadmat(file_path)
    landmarks = mat_data.get('lms')
    if landmarks is None:
        raise ValueError(f"No 'lms' data found in {file_path}")
    if landmarks.shape == (68, 2):
        landmarks_flipped = landmarks.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]  # Flip y-coordinates
        return landmarks_flipped
    elif landmarks.shape[0] == 136 and landmarks.ndim == 1:
        landmarks_reshaped = landmarks.reshape(68, 2)
        landmarks_flipped = landmarks_reshaped.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]
        return landmarks_flipped
    elif landmarks.shape[0] == 68 and landmarks.shape[1] == 2:
        landmarks_flipped = landmarks.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]
        return landmarks_flipped
    else:
        raise ValueError(f"Invalid landmark data in {file_path}: expected (68, 2) or (136,) array, got {landmarks.shape}")

# Load all images and landmarks
print("Loading all images and landmarks...")
images_all = np.array([load_and_preprocess_image(f) for f in image_files])  # Shape: (n_files, 128, 128, 3)
landmarks_all = np.array([load_landmarks(f) for f in landmark_files])      # Shape: (n_files, 68, 2)

# Ensure images and landmarks are aligned by sorting by filename (assuming matching order)
image_files.sort()
landmark_files.sort()
if [os.path.basename(f) for f in image_files] != [os.path.basename(f).replace('.mat', '.jpg') for f in landmark_files]:
    raise ValueError("Image and landmark files are not aligned by name.")

# Randomly shuffle and split into train (800) and test (200) sets
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(len(image_files))
train_indices = indices[:800]
test_indices = indices[800:1000]

images_train = images_all[train_indices]      # Shape: (800, 128, 128, 3)
landmarks_train = landmarks_all[train_indices]  # Shape: (800, 68, 2)
images_test = images_all[test_indices]        # Shape: (200, 128, 128, 3)
landmarks_test = landmarks_all[test_indices]  # Shape: (200, 68, 2)

# Step 1: Compute mean landmarks and top 10 eigen-warpings
mean_landmarks = np.mean(landmarks_train, axis=0)  # Shape: (68, 2)
landmarks_train_centered = landmarks_train - mean_landmarks
U_warp, _, _ = np.linalg.svd(landmarks_train_centered.reshape(800, 136).T, full_matrices=False)
eigen_warpings = U_warp[:, :10]  # Top 10 eigen-warpings, shape: (136, 10)

# Step 2: Warp training images to mean landmark position using top 10 eigen-warpings
def warp_to_mean(landmarks, eigen_warpings, mean_landmarks):
    coeffs = eigen_warpings.T @ landmarks.flatten()  # Project to top 10 eigen-warpings
    reconstructed_landmarks = eigen_warpings @ coeffs + mean_landmarks.flatten()  # Shape: (136,)
    return reconstructed_landmarks.reshape(68, 2)

warped_landmarks_train = np.array([warp_to_mean(l, eigen_warpings, mean_landmarks) for l in landmarks_train_centered])
aligned_images_train = np.array([mywarper.warp(img, landmarks_train[i], warped_landmarks_train[i]) for i, img in enumerate(images_train)])

# Step 3: Compute mean face and top 50 eigen-faces from aligned images
mean_face = np.mean(aligned_images_train.reshape(800, -1), axis=0)  # Shape: (16384 * 3,)
aligned_images_train_centered = aligned_images_train.reshape(800, -1) - mean_face
U_face, _, _ = np.linalg.svd(aligned_images_train_centered.T, full_matrices=False)
eigen_faces = U_face[:, :50]  # Top 50 eigen-faces, shape: (49152, 50) due to 3 channels

# Function to reconstruct images
def reconstruct_image(image, landmarks, eigen_warpings, eigen_faces, mean_landmarks, mean_face, K):
    # Warp to mean position
    coeffs_warp = eigen_warpings.T @ (landmarks.flatten() - mean_landmarks.flatten())
    warped_landmarks = eigen_warpings @ coeffs_warp + mean_landmarks.flatten()
    warped_landmarks = warped_landmarks.reshape(68, 2)
    warped_image = mywarper.warp(image, landmarks, warped_landmarks)
    
    # Project to eigen-faces and reconstruct
    coeffs_face = eigen_faces[:, :K].T @ (warped_image.reshape(-1) - mean_face)
    reconstructed_flat = eigen_faces[:, :K] @ coeffs_face + mean_face
    reconstructed_image = reconstructed_flat.reshape(128, 128, 3)  # Maintain 3 channels
    
    # Warp back to original landmark position
    final_image = mywarper.warp(reconstructed_image, warped_landmarks, landmarks)
    return final_image

# Reconstruct test images with K=50 and display 20 pairs
K = 50
reconstructed_test = np.array([reconstruct_image(images_test[i], landmarks_test[i], eigen_warpings, eigen_faces, mean_landmarks, mean_face, K) for i in range(20)])

plt.figure(figsize=(20, 8))
for i in range(20):
    plt.subplot(4, 10, i + 1)
    plt.imshow(images_test[i], cmap='gray')  # Display as grayscale despite 3 channels
    plt.title('Original')
    plt.axis('off')
    plt.subplot(4, 10, i + 21)
    plt.imshow(reconstructed_test[i], cmap='gray')  # Display as grayscale despite 3 channels
    plt.title('Reconstructed')
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reconstructed_faces.png'))

# Compute and plot reconstruction error for K = 1, 5, 10, ..., 50
K_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
errors = []
print("Computing reconstruction errors...")
for K in K_values:
    reconstructed_test_k = np.array([reconstruct_image(images_test[i], landmarks_test[i], eigen_warpings, eigen_faces, mean_landmarks, mean_face, K) for i in range(len(images_test))])
    error = np.mean(np.sum((images_test.reshape(200, -1) - reconstructed_test_k.reshape(200, -1)) ** 2, axis=1) / (16384 * 3))  # Per pixel, averaged over 3 channels
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(K_values, errors, marker='o')
plt.xlabel('Number of Eigen-Faces (K)')
plt.ylabel('Average Reconstruction Error per Pixel')
plt.title('Reconstruction Error vs. Number of Eigen-Faces')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'reconstruction_error.png'))