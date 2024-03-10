from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage
from matplotlib import pyplot as plt
img = cv2.imread("1.jpeg")
# plt.imshow(img)
# plt.show()

newImg = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

gray = rgb2gray(newImg)
plt.subplot(2, 1, 1)
plt.hist(gray.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
#plt.show()
#plt.imshow(gray, cmap='gray')
#plt.show()



gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > 0.5:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0], gray.shape[1])
#plt.imshow(gray, cmap='gray')
#plt.show()
plt.subplot(2, 1, 2)
plt.hist(gray.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()


