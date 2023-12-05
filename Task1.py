import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_sample_image

# Load sample images of cats and dogs (you should replace this with your dataset)
cats = load_sample_image("cat.jpg")
dogs = load_sample_image("dog.jpg")

# Flatten and normalize the images
cats_flat = cats.reshape((len(cats), -1))
dogs_flat = dogs.reshape((len(dogs), -1))

# Create labels for the images (1 for cats, 0 for dogs)
cat_labels = np.ones(len(cats_flat))
dog_labels = np.zeros(len(dogs_flat))

# Combine data and labels
data = np.vstack((cats_flat, dogs_flat))
labels = np.hstack((cat_labels, dog_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Predictions
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(cats.shape), cmap=plt.cm.gray)
    ax.set_title(f"True: {int(y_test[i])}, Predicted: {int(y_pred[i])}")
    ax.axis('off')

plt.show()
