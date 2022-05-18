# Edge-Detection
## Aim:
To perform edge detection using Sobel, Laplacian, and Canny edge detectors.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the necessary modules.
### Step2:
For performing edge detection on a image.
Sobel
```
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,5)
sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,5)
```
Laplacian
```
Laplacian=cv2.Laplacian(img,cv2.CV_64F)
Canny
canny=cv2.Canny(img,120,150)
```
### Step3:
Display all the images with their respective edge detected images.
 
## Program:

``` Python
# Import the packages

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image, Convert to grayscale and remove noise


ip_img=cv2.imread("car.jpg")
gray_img=cv2.cvtColor(ip_img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Canny edge detector",gray_img)
cv2.waitKey(0)
img=cv2.GaussianBlur(gray_img,(3,3),0)


# SOBEL EDGE DETECTOR

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobelx'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobely'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(sobelxy,cmap = 'gray')
plt.title('Sobelxy'), plt.xticks([]), plt.yticks([])
plt.show()

# LAPLACIAN EDGE DETECTOR


laplacian = cv2.Laplacian(img,cv2.CV_64F)

cv2.imshow("Laplacian edge detector",laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()


# CANNY EDGE DETECTOR

canny = cv2.Canny(img, 70, 150)
cv2.imshow("Canny edge detector",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Output:
### SOBEL EDGE DETECTOR

![car1](https://user-images.githubusercontent.com/75241177/169005068-0e58a2b9-e8f0-4cc1-b695-9161cd862f60.jpg)


### LAPLACIAN EDGE DETECTOR

![car 2](https://user-images.githubusercontent.com/75241177/169005101-653c9dd7-6cc1-44e1-8a31-2d4ee9bf03ba.jpg)


### CANNY EDGE DETECTOR
)
![car 3](https://user-images.githubusercontent.com/75241177/169005138-5c5a8a87-2f31-4866-a199-ba02a4d7a856.jpg)



## Result:
Thus the edges are detected using Sobel, Laplacian, and Canny edge detectors.
