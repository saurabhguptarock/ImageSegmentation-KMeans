import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('elephant.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

all_pixels = img.reshape((330 * 500, 3))

dominant_color = 4

km = KMeans(n_clusters=dominant_color)
km.fit(all_pixels)

centers = np.array(km.cluster_centers_, dtype='uint8')

new_img = np.zeros((330 * 500, 3), dtype='uint8')

for ix in range(new_img.shape[0]):
    new_img[ix] = centers[km.labels_[ix]]
new_img = new_img.reshape((img.shape))
plt.imshow(new_img)
# plt.imshow(img)
plt.show()
