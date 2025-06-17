import os
import numpy as np
from scipy.io import loadmat
from skimage import io, color
import matplotlib.pyplot as plt
import mywarper  # Assuming mywarper.py provides the warp function

# Set up directories
images_dir = './data/images'
landmarks_dir = './data/landmarks'
output_dir = './results/3'
os.makedirs(output_dir, exist_ok=True)

# Load image and landmark files
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
landmark_files = [os.path.join(landmarks_dir, f) for f in os.listdir(landmarks_dir) if f.endswith('.mat')]
image_files.sort()
landmark_files.sort()

# Load and preprocess image
def load_and_preprocess_image(file_path):
    img = io.imread(file_path) / 255.0
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]
    return np.stack([v_channel] * 3, axis=-1)

# Load landmarks with y-coordinate flip
def load_landmarks(file_path):
    mat_data = loadmat(file_path)
    landmarks = mat_data['lms']
    landmarks_flipped = landmarks.reshape(68, 2)
    landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]
    return landmarks_flipped

# Load all data
images_all = np.array([load_and_preprocess_image(f) for f in image_files[:1000]])
landmarks_all = np.array([load_landmarks(f) for f in landmark_files[:1000]])

# Split into train (800) and test (200)
np.random.seed(42)
indices = np.random.permutation(1000)
train_indices = indices[:800]
test_indices = indices[800:]
images_train = images_all[train_indices]
landmarks_train = landmarks_all[train_indices]
images_test = images_all[test_indices]
landmarks_test = landmarks_all[test_indices]

# Compute mean landmarks and eigen-warpings
mean_landmarks = np.mean(landmarks_train, axis=0)
landmarks_train_centered = landmarks_train - mean_landmarks
U_warp, _, _ = np.linalg.svd(landmarks_train_centered.reshape(800, 136).T, full_matrices=False)
eigen_warpings = U_warp[:, :10]

# Warp landmarks to mean
def warp_to_mean(landmarks, eigen_warpings, mean_landmarks):
    coeffs = eigen_warpings.T @ (landmarks.flatten() - mean_landmarks.flatten())
    return (eigen_warpings @ coeffs + mean_landmarks.flatten()).reshape(68, 2)

warped_landmarks_train = np.array([warp_to_mean(l, eigen_warpings, mean_landmarks) for l in landmarks_train])
aligned_images_train = np.array([mywarper.warp(img, landmarks_train[i], mean_landmarks) for i, img in enumerate(images_train)])

# Compute mean face and eigen-faces
mean_face = np.mean(aligned_images_train.reshape(800, -1), axis=0)
aligned_images_train_centered = aligned_images_train.reshape(800, -1) - mean_face
U_face, _, _ = np.linalg.svd(aligned_images_train_centered.T, full_matrices=False)
eigen_faces = U_face[:, :50]

# Reconstruction function with intermediates
def reconstruct_image_with_intermediates(image, landmarks, eigen_warpings, eigen_faces, mean_landmarks, mean_face, K):
    original_image = image
    original_landmarks = landmarks
    coeffs_warp = eigen_warpings.T @ (landmarks.flatten() - mean_landmarks.flatten())
    reconstructed_landmarks = (eigen_warpings @ coeffs_warp + mean_landmarks.flatten()).reshape(68, 2)
    warped_to_mean = mywarper.warp(image, landmarks, mean_landmarks)
    coeffs_face = eigen_faces[:, :K].T @ (warped_to_mean.reshape(-1) - mean_face)
    reconstructed_at_mean = (eigen_faces[:, :K] @ coeffs_face + mean_face).reshape(128, 128, 3)
    final_reconstructed = mywarper.warp(reconstructed_at_mean, mean_landmarks, reconstructed_landmarks)
    return {
        'original_image': original_image,
        'original_landmarks': original_landmarks,
        'reconstructed_landmarks': reconstructed_landmarks,
        'warped_to_mean': warped_to_mean,
        'reconstructed_at_mean': reconstructed_at_mean,
        'final_reconstructed': final_reconstructed
    }

# Visualize for 5 test images
K = 50
for i in range(5):
    intermediates = reconstruct_image_with_intermediates(
        images_test[i], landmarks_test[i], eigen_warpings, eigen_faces, mean_landmarks, mean_face, K
    )

    # Plot landmarks
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(intermediates['original_landmarks'][:, 0], intermediates['original_landmarks'][:, 1], s=10, c='b')
    plt.title(f'Test Image {i+1} - Original Landmarks')
    plt.axis('equal')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(intermediates['reconstructed_landmarks'][:, 0], intermediates['reconstructed_landmarks'][:, 1], s=10, c='r')
    plt.title(f'Test Image {i+1} - Warped Landmarks')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_image_{i+1}_landmarks.png'))
    plt.close()

    # Plot image steps
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(intermediates['original_image'], cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(intermediates['warped_to_mean'], cmap='gray')
    plt.title('Warped to Mean')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(intermediates['reconstructed_at_mean'], cmap='gray')
    plt.title('Reconstructed at Mean')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(intermediates['final_reconstructed'], cmap='gray')
    plt.title('Final Reconstructed')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_image_{i+1}_steps.png'))
    plt.close()

print("Landmark and image plots saved in the output directory.")