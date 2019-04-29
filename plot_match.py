import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('./src/DSC01.JPG', cv2.IMREAD_COLOR)
img2 = cv2.imread('./src/DSC02.JPG', cv2.IMREAD_COLOR)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

matches = bf.match(des1, des2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)


plt.imshow(img3)
plt.axis('off')
#plt.show()
plt.savefig('./src/match.png', dpi=300, bbox_inches='tight')




