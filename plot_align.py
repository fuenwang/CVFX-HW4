import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./src/DSC01.JPG', cv2.IMREAD_COLOR)
img2 = cv2.imread('./src/DSC02.JPG', cv2.IMREAD_COLOR)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

matches = bf.match(des1, des2)
kp1 = [kp1[x.queryIdx] for x in matches]
kp2 = [kp2[x.trainIdx] for x in matches]

kp1 = np.array([x.pt for x in kp1], np.float32)
kp2 = np.array([x.pt for x in kp2], np.float32)
H = cv2.findHomography(kp2, kp1, method=cv2.RANSAC)[0].astype(np.float32)
[h, w, c] = img1.shape
new_img = cv2.warpPerspective(img2, H, (w, h))

mask = np.tile((new_img.mean(2) > 0)[:, :, None], [1, 1, 3])
fuse = img1.copy()
fuse[mask] = ((new_img[mask].astype(float) + img1[mask].astype(float)) / 2).astype(np.uint8)


plt.subplot('141')
plt.title('Fig. 1')
plt.imshow(img1)
plt.axis('off')
plt.subplot('142')
plt.title('Fig. 2')
plt.imshow(img2)
plt.axis('off')
plt.subplot('143')
plt.title('Aligned')
plt.imshow(new_img)
plt.axis('off')
plt.subplot('144')
plt.title('Combined')
plt.imshow(fuse)
plt.axis('off')
#plt.show()
plt.savefig('./src/aligned.png', dpi=300, bbox_inches='tight')




