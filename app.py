from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility.knn import KNN

digits = load_digits()

_, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

model = KNN(k=3, distance=KNN.EUCLIDEAN, x_train=X_train, y_train=y_train)

y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
