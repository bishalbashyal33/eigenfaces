import os
import numpy as np
from scipy.io import loadmat
from skimage import io, color
import matplotlib.pyplot as plt
import mywarper  # Assuming this module is provided

# Directory paths
images_dir = './data/images'
landmarks_dir = './data/landmarks'
output_dir = './results/4'
os.makedirs(output_dir, exist_ok=True)

# Load file paths
image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
landmark_files = sorted([os.path.join(landmarks_dir, f) for f in os.listdir(landmarks_dir) if f.endswith('.mat')])

# Verify data availability
assert len(image_files) >= 1000 and len(landmark_files) >= 1000, "Insufficient images or landmarks"

# Preprocessing functions
def load_and_preprocess_image(file_path):
    img = io.imread(file_path) / 255.0  # Normalize to [0, 1]
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]  # Use V-channel
    return np.stack([v_channel] * 3, axis=-1)  # Replicate to 3 channels for mywarper

def load_landmarks(file_path):
    mat_data = loadmat(file_path)
    landmarks = mat_data.get('lms')
    assert landmarks.shape == (68, 2), f"Invalid landmark shape in {file_path}"
    landmarks[:, 1] = 128 - landmarks[:, 1]  # Flip y-coordinates
    return landmarks

# Load all data
images_all = np.array([load_and_preprocess_image(f) for f in image_files[:1000]])
landmarks_all = np.array([load_landmarks(f) for f in landmark_files[:1000]])

# Train-test split
np.random.seed(42)
indices = np.random.permutation(1000)
train_indices, test_indices = indices[:800], indices[800:]
images_train, images_test = images_all[train_indices], images_all[test_indices]
landmarks_train, landmarks_test = landmarks_all[train_indices], landmarks_all[test_indices]

# Compute eigen-warpings
mean_landmarks = np.mean(landmarks_train, axis=0)  # Shape: (68, 2)
landmarks_train_centered = landmarks_train - mean_landmarks
U_warp, _, _ = np.linalg.svd(landmarks_train_centered.reshape(800, -1).T, full_matrices=False)
eigen_warpings = U_warp[:, :10]  # Top 10 eigen-warpings, Shape: (136, 10)

# Warp training images to mean landmark position
aligned_images_train = []
for i in range(800):
    coeffs = eigen_warpings.T @ landmarks_train_centered[i].flatten()
    recon_landmarks = (eigen_warpings @ coeffs).reshape(68, 2) + mean_landmarks
    warped_img = mywarper.warp(images_train[i], landmarks_train[i], mean_landmarks)
    aligned_images_train.append(warped_img)
aligned_images_train = np.array(aligned_images_train)

# Compute eigen-faces
mean_face = np.mean(aligned_images_train.reshape(800, -1), axis=0)  # Shape: (49152,)
aligned_train_centered = aligned_images_train.reshape(800, -1) - mean_face
U_face, _, _ = np.linalg.svd(aligned_train_centered.T, full_matrices=False)
eigen_faces = U_face[:, :50]  # Top 50 eigen-faces, Shape: (49152, 50)

# Reconstruction function
def reconstruct_test_image(img, landmarks, K):
    coeffs_warp = eigen_warpings.T @ (landmarks - mean_landmarks).flatten()
    recon_landmarks = (eigen_warpings @ coeffs_warp).reshape(68, 2) + mean_landmarks
    warped_img = mywarper.warp(img, landmarks, mean_landmarks)
    coeffs_face = eigen_faces[:, :K].T @ (warped_img.flatten() - mean_face)
    recon_warped = (eigen_faces[:, :K] @ coeffs_face + mean_face).reshape(128, 128, 3)
    return mywarper.warp(recon_warped, mean_landmarks, recon_landmarks)

# Reconstruct and visualize 20 test images
K = 50
reconstructed_images = [reconstruct_test_image(images_test[i], landmarks_test[i], K) for i in range(20)]
plt.figure(figsize=(20, 8))
for i in range(20):
    plt.subplot(4, 10, i + 1)
    plt.imshow(images_test[i][:, :, 0], cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(4, 10, i + 21)
    plt.imshow(reconstructed_images[i][:, :, 0], cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task4_reconstructions.png'))

# Compute reconstruction errors
K_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
errors = []
for K in K_values:
    recon_test = np.array([reconstruct_test_image(img, lm, K) for img, lm in zip(images_test, landmarks_test)])
    error = np.mean(np.sum((images_test.reshape(200, -1) - recon_test.reshape(200, -1)) ** 2, axis=1) / (128 * 128 * 3))
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(K_values, errors, marker='o')
plt.xlabel('Number of Eigen-Faces (K)')
plt.ylabel('Average Reconstruction Error per Pixel')
plt.title('Reconstruction Error vs. K')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'task4_error_plot.png'))