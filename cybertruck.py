import cv2
import numpy as np
img=cv2.imread("images/dog.png",0)
edges=cv2.Canny(img,75,150)
edges=cv2.dilate(edges,None,iterations=1)
edges=cv2.erode(edges,None,iterations=1)
cnt,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
z=np.zeros(img.shape,dtype="uint8")
cyber=np.zeros(img.shape,dtype="uint8")
contours=[cv2.contourArea(c) for c in cnt]
max_index=np.argmax(contours)
cv2.drawContours(z,cnt,max_index,(255,0,0),-1)
peri=cv2.arcLength(cnt[max_index],True)
points=cv2.approxPolyDP(cnt[max_index],0.008*peri,True)
cv2.drawContours(cyber,[points],-1,255,-1)
cv2.imwrite("cyberdog.jpg",cyber)
cv2.imshow("cyber truck",cyber)
cv2.waitKey(0)


'''
Probably this is how tesla made their cyber truck. some tweaks are required though.
Also the dog looks much better
'''
