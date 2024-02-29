import cv2
import numpy as np

img = cv2.imread('imgs/1.jpg')
img2 = cv2.imread('imgs/2.jpg')

img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

matches12 = bf.knnMatch(des1, des2, k=2)
matches21 = bf.knnMatch(des2, des1, k=2)

ratio_check = 0.25
matches = []
indices = []
for (m, n) in matches12:
        
    if m.distance <= ratio_check * n.distance:
        continue
    
    q1, t1 = m.queryIdx, m.trainIdx
    q2, t2 = matches21[t1][0].queryIdx, matches21[t1][0].trainIdx
    
    if q1 != t2:
        continue
    
    matches.append(m)
    indices.append((q1, t1))
    
img3 = cv2.drawMatches(img, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

indices = np.array(indices)
kp1_ = np.array([kp.pt for kp in kp1])
kp2_ = np.array([kp.pt for kp in kp2])
kp1_ = kp1_[indices[:, 0]]
kp2_ = kp2_[indices[:, 1]]

F, mask = cv2.findFundamentalMat(kp1_, kp2_, method=cv2.FM_RANSAC, ransacReprojThreshold=1, confidence=0.99999, maxIters=1000000)

update_matches = [m for m, id in zip(matches, mask) if id != 0]

img4 = cv2.drawMatches(img, kp1, img2, kp2, update_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('imgs/matches1.jpg', img3)
cv2.imwrite('imgs/matches2.jpg', img4)

# setup non-liear optization problem of the fundamental matrix
from scipy.optimize import least_squares

# Define the error function
def error_func(params, x, y):
    F = params.reshape(3, 3)  # reshape the parameters into a 3x3 matrix
    x_prime = np.dot(F, x)  # estimate the correspoding points
    residuals = y - x_prime  # calculate the residuals
    return residuals.ravel()  # return the residuals as a 1D array

def optimize_fundamental_matrix(x, y, F_initial):
    params_initial = F_initial.ravel()  # flatten the initial fundamental matrix into a 1D array
    result = least_squares(error_func, params_initial, args=(x, y))  # run the optimization
    F_optimized = result.x.reshape(3, 3)  # reshape the optimized parameters into a 3x3 matrix
    return F_optimized


