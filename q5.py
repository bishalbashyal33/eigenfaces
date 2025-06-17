import os
import numpy as np
from skimage import io, color
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Define directories
data_dir = './data'
male_images_dir = os.path.join(data_dir, 'male_images')
male_landmarks_dir = os.path.join(data_dir, 'male_landmarks')
female_images_dir = os.path.join(data_dir, 'female_images')
female_landmarks_dir = os.path.join(data_dir, 'female_landmarks')
output_dir = './results/3_1'
os.makedirs(output_dir, exist_ok=True)

# Load male data
male_base_names = [os.path.splitext(f)[0] for f in os.listdir(male_images_dir) if f.endswith('.jpg')]
male_images = []
male_landmarks = []
for base_name in male_base_names:
    img_path = os.path.join(male_images_dir, base_name + '.jpg')
    landmark_path = os.path.join(male_landmarks_dir, base_name + '.mat')
    img = io.imread(img_path) / 255.0
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]
    img_3ch = np.stack([v_channel] * 3, axis=-1)
    landmarks = loadmat(landmark_path)['lms'].reshape(68, 2)
    landmarks[:, 1] = 128 - landmarks[:, 1]  # Flip y-coordinates
    male_images.append(img_3ch)
    male_landmarks.append(landmarks.flatten())

# Load female data
female_base_names = [os.path.splitext(f)[0] for f in os.listdir(female_images_dir) if f.endswith('.jpg')]
female_images = []
female_landmarks = []
for base_name in female_base_names:
    img_path = os.path.join(female_images_dir, base_name + '.jpg')
    landmark_path = os.path.join(female_landmarks_dir, base_name + '.mat')
    img = io.imread(img_path) / 255.0
    img_hsv = color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]
    img_3ch = np.stack([v_channel] * 3, axis=-1)
    landmarks = loadmat(landmark_path)['lms'].reshape(68, 2)
    landmarks[:, 1] = 128 - landmarks[:, 1]  # Flip y-coordinates
    female_images.append(img_3ch)
    female_landmarks.append(landmarks.flatten())

# Combine data
images_all = np.array(male_images + female_images)
landmarks_all = np.array(male_landmarks + female_landmarks)
labels_all = np.array([0] * len(male_images) + [1] * len(female_images))

# Shuffle and split into training (800) and testing (200)
np.random.seed(42)
indices = np.random.permutation(len(images_all))
train_indices = indices[:800]
test_indices = indices[800:]
images_train = images_all[train_indices]
landmarks_train = landmarks_all[train_indices]
labels_train = labels_all[train_indices]
images_test = images_all[test_indices]
landmarks_test = landmarks_all[test_indices]
labels_test = labels_all[test_indices]

# PCA for landmarks (reduce to 10 dimensions)
mean_landmarks = np.mean(landmarks_train, axis=0)
landmarks_train_centered = landmarks_train - mean_landmarks
U_warp, _, _ = np.linalg.svd(landmarks_train_centered.T, full_matrices=False)
eigen_warpings = U_warp[:, :10]
landmarks_train_pca = eigen_warpings.T @ landmarks_train_centered.T  # (10, 800)
landmarks_test_centered = landmarks_test - mean_landmarks
landmarks_test_pca = eigen_warpings.T @ landmarks_test_centered.T  # (10, 200)

# PCA for images (reduce to 50 dimensions)
images_train_flat = images_train.reshape(800, -1)  # (800, 49152)
mean_face = np.mean(images_train_flat, axis=0)
images_train_centered = images_train_flat - mean_face
U_face, _, _ = np.linalg.svd(images_train_centered.T, full_matrices=False)
eigen_faces = U_face[:, :50]
images_train_pca = eigen_faces.T @ images_train_centered.T  # (50, 800)
images_test_flat = images_test.reshape(200, -1)
images_test_centered = images_test_flat - mean_face
images_test_pca = eigen_faces.T @ images_test_centered.T  # (50, 200)

# Concatenate features
features_train = np.vstack([landmarks_train_pca, images_train_pca])  # (60, 800)
features_test = np.vstack([landmarks_test_pca, images_test_pca])  # (60, 200)

# Compute FLD
mu_0 = np.mean(features_train[:, labels_train == 0], axis=1)
mu_1 = np.mean(features_train[:, labels_train == 1], axis=1)
S_w = np.zeros((60, 60))
for i in range(800):
    if labels_train[i] == 0:
        S_w += np.outer(features_train[:, i] - mu_0, features_train[:, i] - mu_0)
    else:
        S_w += np.outer(features_train[:, i] - mu_1, features_train[:, i] - mu_1)
W = np.linalg.solve(S_w, mu_1 - mu_0)
W = W / np.linalg.norm(W)

# Project data
projections_train = W.T @ features_train
projections_test = W.T @ features_test

# Determine threshold
mean_proj_0 = np.mean(projections_train[labels_train == 0])
mean_proj_1 = np.mean(projections_train[labels_train == 1])
threshold = (mean_proj_0 + mean_proj_1) / 2

# Classify and compute error rate
predictions = (projections_test > threshold).astype(int)
error_rate = np.mean(predictions != labels_test)
print(f"Error rate: {error_rate * 100:.2f}%")

# Visualize projections
plt.figure()
plt.hist(projections_train[labels_train == 0], bins=20, alpha=0.5, label='Male (train)')
plt.hist(projections_train[labels_train == 1], bins=20, alpha=0.5, label='Female (train)')
plt.axvline(threshold, color='k', linestyle='--', label='Threshold')
plt.legend()
plt.title('FLD Projections (Training Set)')
plt.savefig(os.path.join(output_dir, 'fld_projections_train.png'))
plt.close()

plt.figure()
plt.hist(projections_test[labels_test == 0], bins=20, alpha=0.5, label='Male (test)')
plt.hist(projections_test[labels_test == 1], bins=20, alpha=0.5, label='Female (test)')
plt.axvline(threshold, color='k', linestyle='--', label='Threshold')
plt.legend()
plt.title('FLD Projections (Test Set)')
plt.savefig(os.path.join(output_dir, 'fld_projections_test.png'))
plt.close()

# Visualize Fisher Face
# Extract the appearance component (50 dimensions) from W
W_face = W[10:]  # Skip the 10 landmark dimensions

# Reconstruct the Fisher face
fisher_face_flat = mean_face + (eigen_faces @ W_face)
fisher_face = fisher_face_flat.reshape(128, 128, 3)
fisher_face = np.clip(fisher_face, 0, 1)  # Ensure values are in [0, 1]

plt.figure(figsize=(5, 5))
plt.imshow(fisher_face, cmap='gray')
plt.title('Fisher Face')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'fisher_face.png'))
plt.close()