import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
img1 = cv2.imread('./src/orig.jpg',0)          # queryImage
img2 = cv2.imread('./src/mod.jpg',0)           # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

a = np.array([x.pt for x in kp1], np.float32)
b = np.array([x.pt for x in kp2], np.float32)
print cv2.findHomography(a, b, method=cv2.RANSAC)[0]
exit()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img1 = cv2.imread('./src/orig.jpg')          # queryImage
img2 = cv2.imread('./src/mod.jpg')           # trainImage

img1 = img1[...,::-1]
img2 = img2[...,::-1]

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3),plt.show()
'''

def interpo(img, n, dsize):
    scale_lst = np.linspace(1, 1.5, n)
    [h ,w] = dsize
    #cv2.namedWindow('ggg')
    out = []
    for scale in scale_lst:
        new_img = cv2.resize(img, None, fx=scale, fy=scale)

        [new_h, new_w, _] = new_img.shape
        
        offset_h = (new_h - h ) // 2
        offset_w = (new_w - w ) // 2
        new_img = new_img[offset_h:offset_h+h, offset_w:offset_w+w]
        
        #cv2.imshow('ggg', new_img)
        #cv2.waitKey(0)
        out.append(new_img)
    
    return out

def getHomography(lst):
    kp_lst = []
    des_lst = []
    #orb = cv2.ORB_create()
    #orb = cv2.xfeatures2d.SIFT_create()
    orb = cv2.xfeatures2d.SURF_create()
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    for one in lst:
        kp, d = orb.detectAndCompute(one, None)
        #kp_lst.append(np.array([x.pt for x in kp], np.float32))
        #print ([x.pt for x in kp])
        #exit()
        kp_lst.append(kp)
        des_lst.append(d)

    homo_lst = [np.eye(3).astype(np.float32)]
    [h, w, c] = lst[0].shape
    tmp = lst[0].copy()
    aaa = []
    for i in range(1, len(lst)):
        #print i
        #print des_lst[i]

        matches = bf.match(des_lst[i - 1], des_lst[i])

        kp1 = [kp_lst[i - 1][x.queryIdx] for x in matches]
        kp2 = [kp_lst[i][x.trainIdx] for x in matches]

        kp1 = np.array([x.pt for x in kp1], np.float32)
        kp2 = np.array([x.pt for x in kp2], np.float32)

        H = cv2.findHomography(kp2, kp1, method=cv2.RANSAC)[0].astype(np.float32)
        homo_lst.append(H)
        #new_img = cv2.warpPerspective(lst[i], np.dot(homo_lst[i-1], H), (w, h))
        new_img = cv2.warpPerspective(lst[i], H, (w, h))
        '''
        plt.subplot('131')
        plt.imshow(lst[i-1])
        plt.subplot('132')
        plt.imshow(lst[i])
        plt.subplot('133')
        plt.imshow(new_img)
        plt.show()
        '''
        mask = np.tile((new_img.mean(2) > 0)[:, :, None], [1, 1, 3])
        tmp[mask] = ((new_img[mask].astype(float) + tmp[mask].astype(float)) / 2).astype(np.uint8)
        aaa += interpo(tmp, 10, (h, w))        
        tmp = lst[i].copy()

    return aaa
lst = ['./src/%s' %
       x for x in sorted(os.listdir('./src')) if x.endswith('.JPG')]
img_lst = [cv2.imread(x, cv2.IMREAD_COLOR) for x in lst]
h, w, c = img_lst[0].shape
aaa = getHomography(img_lst[:])
video=cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc('P','I','M','1'), frameSize=(w, h), fps=8)
for one in aaa:
    video.write(one)
video.release()
