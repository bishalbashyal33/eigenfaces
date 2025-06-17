import os
import numpy as np
from scipy.io import loadmat
from skimage import io, color
import matplotlib.pyplot as plt
import mywarper

# Set directories
images_dir = './data/images'
landmarks_dir = './data/landmarks'
output_dir = './results/4'
os.makedirs(output_dir, exist_ok=True)

# Load image and landmark files
image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
landmark_files = sorted([os.path.join(landmarks_dir, f) for f in os.listdir(landmarks_dir) if f.endswith('.mat')])

# Define loading functions
def load_and_preprocess_image(file_path):
    img = io.imread(file_path) / 255.0
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]
    return np.stack([v_channel] * 3, axis=-1)

def load_landmarks(file_path):
    mat_data = loadmat(file_path)
    landmarks = mat_data.get('lms')
    if landmarks is None:
        raise ValueError(f"No 'lms' data found in {file_path}")
    if landmarks.shape == (68, 2):
        landmarks_flipped = landmarks.copy()
        landmarks_flipped[:, 1] = 128 - landmarks_flipped[:, 1]
        return landmarks_flipped
    else:
        raise ValueError(f"Invalid landmark shape in {file_path}")

# Load training data (first 800 samples)
np.random.seed(42)
indices = np.random.permutation(1000)
train_indices = indices[:800]
images_train = np.array([load_and_preprocess_image(image_files[i]) for i in train_indices])
landmarks_train = np.array([load_landmarks(landmark_files[i]) for i in train_indices])

# Compute mean landmarks and center the data
mean_landmarks = np.mean(landmarks_train, axis=0)  # Shape: (68, 2)
landmarks_train_centered = landmarks_train - mean_landmarks  # Shape: (800, 68, 2)
landmarks_train_centered_flat = landmarks_train_centered.reshape(800, 136)  # Shape: (800, 136)

# PCA for landmarks (eigen-warpings)
U_warp, S_warp, _ = np.linalg.svd(landmarks_train_centered_flat.T, full_matrices=False)
variances_warp = (S_warp[:10] ** 2) / (800 - 1)  # Variances for top 10 components

# Warp training images to mean landmarks and compute mean face
aligned_images_train = np.array([mywarper.warp(images_train[i], landmarks_train[i], mean_landmarks) for i in range(800)])
mean_face = np.mean(aligned_images_train.reshape(800, -1), axis=0)  # Shape: (49152,)
aligned_images_train_centered = aligned_images_train.reshape(800, -1) - mean_face  # Shape: (800, 49152)

# PCA for appearance (eigen-faces)
U_face, S_face, _ = np.linalg.svd(aligned_images_train_centered.T, full_matrices=False)
variances_face = (S_face[:50] ** 2) / (800 - 1)  # Variances for top 50 components

# Generate and display 50 synthesized faces
plt.figure(figsize=(20, 10))
for i in range(50):
    # Sample coefficients for landmarks
    c = np.random.normal(0, np.sqrt(variances_warp), size=10)
    new_landmarks_flat = mean_landmarks.flatten() + U_warp[:, :10] @ c
    new_landmarks = new_landmarks_flat.reshape(68, 2)

    # Sample coefficients for appearance
    d = np.random.normal(0, np.sqrt(variances_face), size=50)
    new_appearance_flat = mean_face + U_face[:, :50] @ d
    new_appearance = new_appearance_flat.reshape(128, 128, 3)

    # Warp the new appearance to the new landmarks
    synthesized_face = mywarper.warp(new_appearance, mean_landmarks, new_landmarks)
    synthesized_face = np.clip(synthesized_face, 0, 1)

    # Plot the synthesized face
    plt.subplot(5, 10, i + 1)
    plt.imshow(synthesized_face, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'synthesized_faces.png'))
plt.close()