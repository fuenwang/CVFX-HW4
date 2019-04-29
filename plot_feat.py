import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('./src/DSC01.JPG', cv2.IMREAD_COLOR)
img2 = cv2.imread('./src/DSC02.JPG', cv2.IMREAD_COLOR)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

kp1 = [x.pt for x in kp1]
kp2 = [x.pt for x in kp2]

for one in kp1:
    cv2.circle(img1, (int(one[0]), int(one[1])), 3, (255, 0, 0), 2)
for one in kp2:
    cv2.circle(img2, (int(one[0]), int(one[1])), 3, (255, 0, 0), 2)

plt.subplot('121')
plt.title('Fig. 1')
plt.imshow(img1)
plt.axis('off')
plt.subplot('122')
plt.title('Fig. 2')
plt.imshow(img2)
plt.axis('off')
#plt.show()
plt.savefig('./src/feat.png', dpi=300, bbox_inches='tight')




