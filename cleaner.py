import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt

print("Loading images")

imgs_paths = ["1.pgm","3.pgm","4.pgm","11.pgm","12.pgm"]
imgs = []

for i in range(len(imgs_paths)):
    img = cv.imread(imgs_paths[i],0)
    cv.normalize(img.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
    imgs.append(img)

print("Computing gradients")

gradx = []
grady = []
for i in range(len(imgs)):
    sobelx = cv.Sobel(imgs[i],cv.CV_64F,1,0,ksize=5)
    gradx.append(sobelx)
    sobely = cv.Sobel(imgs[i],cv.CV_64F,0,1,ksize=5)
    grady.append(sobely)

print("Computing median gradient")

img = cv.imread(imgs_paths[0],0)
cv.normalize(img.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
cv.imwrite("out.png",img)

rows = img.shape[0]
cols = img.shape[1]

medimgx = np.zeros((rows,cols),np.float64)
medimgy = np.zeros((rows,cols),np.float64)

for i in range(rows):
  for j in range(cols):
    pixelsx = []
    pixelsy = []
    for k in range(len(imgs)):
        pixelsx.append(gradx[k][i,j])
        pixelsy.append(grady[k][i,j])
    medimgx[i,j] = sorted(pixelsx)[math.ceil(len(pixelsx)/2)-1]
    medimgy[i,j] = sorted(pixelsy)[math.ceil(len(pixelsy)/2)-1]

print("Solving poisson problem")

medimgxx = cv.Sobel(medimgx,cv.CV_64F,1,0,ksize=5)
medimgyy = cv.Sobel(medimgy,cv.CV_64F,0,1,ksize=5)

u = np.zeros((rows,cols),np.float64)
u0 = np.zeros((rows,cols),np.float64)

ITER = 100
for k in range(ITER):
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            #u[i,j] = (u0[i+1,j]+u[i-1,j]+u0[i,j+1]+u[i,j-1] -medimgxx[i,j]-medimgyy[i,j])/4
            u[i,j] = ((2/3)*(u0[i+1,j]+u[i-1,j]+u0[i,j+1]+u[i,j-1])+(1/6)*(u0[i+1,j+1]+u0[i+1,j-1]+u0[i-1,j+1]+u[i-1,j-1]) -medimgxx[i,j]-medimgyy[i,j])*(3/10) 
    
    u[0,:] = u[1,:]
    u[rows-1,:] = u[rows-2,:]
    u[:,0] = u[:,1]
    u[:,cols-1] = u[:,cols-2]
    
    print(np.linalg.norm(u-u0))
    
    for i in range(rows):
        for j in range(cols):
            u0[i,j] = u[i,j]
    
    print(k, ITER-1)

print("Done!!")

cv.imwrite("out.png",u)

print(u)

plt.subplot(1,2,1),plt.imshow(imgs[0],cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(u,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

