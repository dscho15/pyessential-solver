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

F, mask = cv2.findFundamentalMat(kp1_, kp2_, method=cv2.FM_RANSAC, ransacReprojThreshold=3, confidence=0.9999, maxIters=10000)
F, _ = cv2.findFundamentalMat(kp1_[mask.flatten() == 1], kp2_[mask.flatten() == 1], method=cv2.FM_LMEDS)

update_matches = [m for m, id in zip(matches, mask) if id != 0]

img4 = cv2.drawMatches(img, kp1, img2, kp2, update_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('imgs/matches1.jpg', img3)
cv2.imwrite('imgs/matches2.jpg', img4)

# setup non-liear optization problem of the fundamental matrix
from scipy.optimize import least_squares

def rotm_x(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def rotm_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def convert_fundamental_matrix(F):
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    S[1] = S[1] / S[0]
    S[0] = 1
    return U, S, Vt.T

U_, S_, V_ = convert_fundamental_matrix(F)

x = np.array(kp1_[mask.flatten() == 1])
y = np.array(kp2_[mask.flatten() == 1])

# pad with a column of ones
x = np.hstack([x, np.ones((x.shape[0], 1))])
y = np.hstack([y, np.ones((y.shape[0], 1))])

# Define the error function
def error_func(params, x, y):    
    U = U_ @ rotm_x(params[0])
    V = V_ @ rotm_y(params[1])
    S = S_[1] + params[2]
    F = U[:, 0].reshape(3, 1) @ V[:, 0].reshape(1, 3) + S * U[:, 1].reshape(3, 1) @ V[:, 1].reshape(1, 3)
    
    F = F / F[2, 2]
    residuals = np.sum(((F @ x.T).T * y), axis=1)    
    
    return residuals.ravel()  # return the residuals as a 1D array

def optimize_fundamental_matrix(x, y):
    params_initial = [0, 0, 0]
    result = least_squares(error_func, params_initial, args=(x, y), loss='cauchy', verbose=True)
    F_optimized = result.x.reshape(3, )
    return F_optimized

optimize_fundamental_matrix(x, y)