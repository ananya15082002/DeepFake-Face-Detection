import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


#FISHERFACE LOCAL BINARY PATTERN HISTOGRAM(LBPH) Features and LINEAR DISCRIMNANT ANALYSIS (LDA)
# Function to load preprocessed images
def load_preprocessed_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        images.append(image)
        labels.append(1 if 'real' in filename else 0)  # Assuming 'real' in filename indicates real image
    return images, labels

# Function to perform Fisherface-LBPH feature extraction
def fisherface_lbph(images, labels):
    X = []
    y = []
    radius = 3
    n_points = 8 * radius
    for image, label in zip(images, labels):
        lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
        lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        X.append(lbp_histogram)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    # Dimensionality reduction using Fisherface
    lda = LinearDiscriminantAnalysis()
    X_fisherface = lda.fit_transform(X, y)

    return X_fisherface, y

# Paths to the preprocessed dataset directories
preprocessed_real_dir = 'preprocesseddata/real'
preprocessed_fake_dir = 'preprocesseddata/fake'

# Load preprocessed dataset
real_images, real_labels = load_preprocessed_data(preprocessed_real_dir)
fake_images, fake_labels = load_preprocessed_data(preprocessed_fake_dir)

# Perform feature extraction using Fisherface-LBPH
X_real, y_real = fisherface_lbph(real_images, real_labels)
X_fake, y_fake = fisherface_lbph(fake_images, fake_labels) 

# Combine real and fake features and labels
X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

# Save the features and labels
np.save('features.npy', X)
np.save('labels.npy', y)
print("done")