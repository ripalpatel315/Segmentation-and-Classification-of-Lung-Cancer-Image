import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from PIL import Image
import skimage.filters as sp
import matplotlib.pyplot as plt

# Load image, grayscale, Otsu's threshold
#image = cv2.imread('1.jpeg')
image = cv2.imread('C:/Users/lenovo/Downloads/Nethra/DATASET/DATASET/NC/NC/009.bmp')
image= cv2.resize(image, (0,0), fx=0.25, fy=0.25)
original = image
plt.subplot(1, 4, 1)
#cv2.imshow(' Original image', image)
plt.imshow(image)
#convert color to grayscale image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('B&W image', gray)


#THresholding
#img, thresh1) = cv2.threshold(gray, 0, 255,  cv2.THRESH_OTSU)

# Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
# Use a bimodal image as an input.
# Optimal threshold value is determined automatically.
otsu_threshold, image_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
print("Obtained threshold: ", otsu_threshold)


"""
#cv2.imshow('Otsu Threshold', thresh1)
img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.subplot(1, 4, 2)
#cv2.imshow('Threshold image', img)
plt.imshow(img)

#Morphological
kernel = np.ones((10,15), np.uint8)
#kernel1 = np.ones((15,15), np.uint8)

dilation =   cv2.dilate(img, kernel, iterations=1)
erosion = cv2.erode(dilation,kernel, iterations=1) # refines all edges in the binary image

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)


plt.subplot(1, 4, 3)
#cv2.imshow('After Morphological image',closing)
plt.imshow(closing)
# Find contours, obtain bounding box, extract and save ROI


ROI_number = 0
cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# focus on only the largest outline by area
areas = [] #list to hold all areas

for contour in cnts:
    ar = cv2.contourArea(contour)
    areas.append(ar)

max_area = max(areas)
max_area_index = areas.index(max_area)  # index of the list element with largest area

cnt = cnts[max_area_index ] # largest area contour is usually the viewing window itself, why?

#print('Contour',cnt)
cv2.drawContours(image, [cnt], 0, (0,0,255), 2)
(x1,y1,w1,h1) = cv2.boundingRect(cnt)
ROI_G = gray[y1:y1+h1, x1:x1+w1]
plt.subplot(1, 4, 4)
#cv2.imshow('segmented image', image)
plt.imshow(image)
plt.show()
cv2.waitKey()

glcm = greycomatrix(ROI_G, distances=[5], angles=[0], levels=256,symmetric=True, normed=True)
#data = im.fromarray(glcm)
#img1 = Image.fromarray((glcm * 255).astype(np.uint8))
#img1 = Image.fromarray(glcm)
#print('glcm',glcm)
#np_img = np.squeeze(glcm, axis=2)  # axis=2 is channel dimension
#pil_img = Image.fromarray(np_img)


#data.show()
#print(type(glcm))
D=greycoprops(glcm, 'dissimilarity')
C=greycoprops(glcm, 'correlation')
Cr = greycoprops(glcm, 'contrast')
H = greycoprops(glcm, 'homogeneity')
E = greycoprops(glcm, 'energy')
A = greycoprops(glcm, 'ASM')

print('Contrast=',Cr, 'Dissimilarity=', D, 'Homogeneity=', H, 'Energy=', E, 'Correlation=',C, 'ASM=',A )
cv2.waitKey()
"""