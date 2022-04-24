
import cv2

bodyis = cv2.VideoCapture("4f.jpg")

gray= cv2.cvtColor(bodyis, cv2.COLOR_BGR2GRAY)

body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

bodyis = body_cascade.detectMultiScale(gray,1.2,3)

for (x,y,w,h) in bodyis:
       cv2.rectangle(bodyis,(x,y),(x+w,y+h),(255,0,0),2)
       roi_color=bodyis[y:y+h,x:x+w]
       cv2.imwrite("face.jpg",roi_color)
             
cv2.imshow('body dtetin',bodyis)
cv2.waitKey(0)



