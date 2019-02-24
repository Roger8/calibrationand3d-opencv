
import numpy as np
import cv2
#import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('imgs/*.jpg')[:15]
#images = [os.path.join('imgs',x) for x in os.listdir('imgs')[:15]]
images = ['imgs/'+x for x in os.listdir('imgs')[:15]]
print images

pattern = (5,7)
for fname in images:
    print fname
    img = cv2.imread(fname)
    print 'img',img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print gray.shape

    help(cv2.findChessboardCorners)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,7),None)
    print ret,corners

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        help(cv2.cornerSubPix)
        cv2.cornerSubPix(gray,corners,(30,30),(-1,-1),criteria)
        corners2 = corners
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (5,7), corners2,ret)
        print img.shape
        cv2.imshow('img',img)
        cv2.waitKey(5)
cv2.destroyAllWindows()

help(cv2.calibrateCamera)

print type(objpoints[0])
print type(imgpoints[0])
#print objpoints
#print imgpoints
print gray.shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape)
print 'ret:', ret
print 'mtx:', mtx
print 'dist:', dist
print 'rvecs:',rvecs
print 'tvecs:',tvecs

# 
imx = cv2.imread('imgs/7.jpg')
h,w,_ = imx.shape
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist, (w,h),1,(w,h))
#### 1 
#undistort
dst = cv2.undistort(imx, mtx, dist, None, newcameramtx)
# crop
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
#cv2.imwrite('al.png',dst)

#### 2
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(imx,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('a2.png',dst)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print "total error: ", mean_error/len(objpoints)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
    
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# draw cube
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    help(cv2.line)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    help(cv2.drawContours)
    print imgpts
    cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


def projet(fname):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,7),None)

    if ret == True:
        cv2.cornerSubPix(gray,corners,(30,30),(-1,-1),criteria)
        corners2  = corners 
        print corners2.shape, objp.shape

        # Find the rotation and translation vectors.
        help(cv2.solvePnPRansac)
        _rvecs, _tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, _rvecs, _tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        print k
        if k == ord('s'):
            cv2.imwrite(fname[:-4]+'_.png', img)
        cv2.destroyAllWindows()
projet('imgs/19.jpg')
