import os
import numpy as np
from scipy.io import loadmat
from skimage import io, color
import matplotlib.pyplot as plt
import mywarper  # Assuming mywarper.py is provided

# Directory paths
images_dir = './data/images'
male_landmarks_dir = './data/male_landmarks'
female_landmarks_dir = './data/female_landmarks'
output_dir = './figures'
os.makedirs(output_dir, exist_ok=True)

# Load file paths
image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
male_landmark_files = sorted([os.path.join(male_landmarks_dir, f) for f in os.listdir(male_landmarks_dir) if f.endswith('.mat')])
female_landmark_files = sorted([os.path.join(female_landmarks_dir, f) for f in os.listdir(female_landmarks_dir) if f.endswith('.mat')])

# Verify data availability
assert len(image_files) >= 1000, "Insufficient images"
print(f"Found {len(image_files)} images, {len(male_landmark_files)} male landmarks, {len(female_landmark_files)} female landmarks.")

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

# Load all data and align by filename
images_all = np.array([load_and_preprocess_image(f) for f in image_files[:1000]])  # Limit to 1000
male_landmarks_all = np.array([load_landmarks(f) for f in male_landmark_files])
female_landmarks_all = np.array([load_landmarks(f) for f in female_landmark_files])

# Align images with landmarks by matching filenames
def get_image_index(landmark_file):
    base = os.path.basename(landmark_file).split('.')[0]  # e.g., '000001'
    index = int(base) - 1  # Convert to 0-based index
    if index >= len(image_files):
        print(f"Warning: Index {index} out of bounds for {landmark_file}, skipping")
        return None
    return index

male_indices = [get_image_index(f) for f in male_landmark_files]
female_indices = [get_image_index(f) for f in female_landmark_files]
valid_male_indices = [i for i in male_indices if i is not None]
valid_female_indices = [i for i in female_indices if i is not None]
images_male = images_all[valid_male_indices]
images_female = images_all[valid_female_indices]
landmarks_male = male_landmarks_all[[i is not None for i in male_indices]]
landmarks_female = female_landmarks_all[[i is not None for i in female_indices]]

# Train-test split for each gender
np.random.seed(42)
male_indices = np.random.permutation(len(images_male))
female_indices = np.random.permutation(len(images_female))
male_train_indices = male_indices[:int(0.8 * len(male_indices))]
male_test_indices = male_indices[int(0.8 * len(male_indices)):]
female_train_indices = female_indices[:int(0.8 * len(female_indices))]
female_test_indices = female_indices[int(0.8 * len(female_indices)):]

images_male_train = images_male[male_train_indices]
images_male_test = images_male[male_test_indices]
landmarks_male_train = landmarks_male[male_train_indices]
landmarks_male_test = landmarks_male[male_test_indices]
images_female_train = images_female[female_train_indices]
images_female_test = images_female[female_test_indices]
landmarks_female_train = landmarks_female[female_train_indices]
landmarks_female_test = landmarks_female[female_test_indices]

# Compute gender-specific models
def compute_model(images, landmarks):
    mean_landmarks = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - mean_landmarks
    U_warp, _, _ = np.linalg.svd(landmarks_centered.reshape(len(landmarks), -1).T, full_matrices=False)
    eigen_warpings = U_warp[:, :10]
    
    aligned_images = np.array([mywarper.warp(img, lm, mean_landmarks) for img, lm in zip(images, landmarks)])
    mean_face = np.mean(aligned_images.reshape(len(aligned_images), -1), axis=0)
    aligned_centered = aligned_images.reshape(len(aligned_images), -1) - mean_face
    U_face, _, _ = np.linalg.svd(aligned_centered.T, full_matrices=False)
    eigen_faces = U_face[:, :50]
    return mean_landmarks, eigen_warpings, mean_face, eigen_faces

mean_male_landmarks, eigen_warpings_male, mean_male_face, eigen_faces_male = compute_model(images_male_train, landmarks_male_train)
mean_female_landmarks, eigen_warpings_female, mean_female_face, eigen_faces_female = compute_model(images_female_train, landmarks_female_train)

# Reconstruction function (assuming Task 5 extends Task 4 with a new metric, e.g., PSNR)
def reconstruct_image(img, landmarks, mean_landmarks, eigen_warpings, mean_face, eigen_faces, K):
    coeffs_warp = eigen_warpings.T @ (landmarks.flatten() - mean_landmarks.flatten())
    recon_landmarks = (eigen_warpings @ coeffs_warp).reshape(68, 2) + mean_landmarks
    warped_img = mywarper.warp(img, landmarks, mean_landmarks)
    coeffs_face = eigen_faces[:, :K].T @ (warped_img.flatten() - mean_face)
    recon_warped = (eigen_faces[:, :K] @ coeffs_face + mean_face).reshape(128, 128, 3)
    return mywarper.warp(recon_warped, mean_landmarks, recon_landmarks)

# Reconstruct test images with K=50
K = 50
reconstructed_male = [reconstruct_image(images_male_test[i], landmarks_male_test[i], mean_male_landmarks, eigen_warpings_male, mean_male_face, eigen_faces_male, K) for i in range(len(images_male_test))]
reconstructed_female = [reconstruct_image(images_female_test[i], landmarks_female_test[i], mean_female_landmarks, eigen_warpings_female, mean_female_face, eigen_faces_female, K) for i in range(len(images_female_test))]

# Compute PSNR as a new metric (assuming Task 5 requests this)
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Normalized images
    return 20 * np.log10(max_pixel / np.sqrt(mse))

male_psnr = [psnr(images_male_test[i], reconstructed_male[i]) for i in range(len(images_male_test))]
female_psnr = [psnr(images_female_test[i], reconstructed_female[i]) for i in range(len(images_female_test))]
avg_male_psnr = np.mean(male_psnr)
avg_female_psnr = np.mean(female_psnr)
print(f"Average PSNR - Male: {avg_male_psnr:.2f} dB, Female: {avg_female_psnr:.2f} dB")

# Visualize 10 samples per gender
plt.figure(figsize=(20, 8))
for i in range(10):
    plt.subplot(4, 10, i + 1)
    plt.imshow(images_male_test[i][:, :, 0], cmap='gray')
    plt.title('Male Orig')
    plt.axis('off')
    plt.subplot(4, 10, i + 11)
    plt.imshow(reconstructed_male[i][:, :, 0], cmap='gray')
    plt.title('Male Recon')
    plt.axis('off')
for i in range(10):
    plt.subplot(4, 10, i + 21)
    plt.imshow(images_female_test[i][:, :, 0], cmap='gray')
    plt.title('Female Orig')
    plt.axis('off')
    plt.subplot(4, 10, i + 31)
    plt.imshow(reconstructed_female[i][:, :, 0], cmap='gray')
    plt.title('Female Recon')
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task5_reconstructions.png'))

# Compute and plot reconstruction errors (MSE) and PSNR
K_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
male_errors = []
female_errors = []
male_psnrs = []
female_psnrs = []

for K in K_values:
    recon_male = np.array([reconstruct_image(img, lm, mean_male_landmarks, eigen_warpings_male, mean_male_face, eigen_faces_male, K) for img, lm in zip(images_male_test, landmarks_male_test)])
    recon_female = np.array([reconstruct_image(img, lm, mean_female_landmarks, eigen_warpings_female, mean_female_face, eigen_faces_female, K) for img, lm in zip(images_female_test, landmarks_female_test)])
    male_error = np.mean(np.sum((images_male_test.reshape(len(images_male_test), -1) - recon_male.reshape(len(recon_male), -1)) ** 2, axis=1) / (128 * 128 * 3))
    female_error = np.mean(np.sum((images_female_test.reshape(len(images_female_test), -1) - recon_female.reshape(len(recon_female), -1)) ** 2, axis=1) / (128 * 128 * 3))
    male_psnr_k = np.mean([psnr(images_male_test[i], recon_male[i]) for i in range(len(images_male_test))])
    female_psnr_k = np.mean([psnr(images_female_test[i], recon_female[i]) for i in range(len(images_female_test))])
    male_errors.append(male_error)
    female_errors.append(female_error)
    male_psnrs.append(male_psnr_k)
    female_psnrs.append(female_psnr_k)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(K_values, male_errors, marker='o', label='Male MSE')
plt.plot(K_values, female_errors, marker='o', label='Female MSE')
plt.xlabel('Number of Eigen-Faces (K)')
plt.ylabel('Mean Squared Error')
plt.title('Reconstruction Error vs. K')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(K_values, male_psnrs, marker='o', label='Male PSNR')
plt.plot(K_values, female_psnrs, marker='o', label='Female PSNR')
plt.xlabel('Number of Eigen-Faces (K)')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs. K')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task5_error_metrics.png'))