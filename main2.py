import cv2
body_classifier=cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')
cap=cv2.VideoCapture("bb3.mp4") 
while (True): 
    ret,frame= cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=body_classifier.detectMultiScale(gray,1.1,3)
    for(x,y,w,h) in bodies: 
        cv2.rectangle(frame,(x,y),(x+w,x+y),(0,255,0))
        cv2.imshow("body detection", frame) 
    if cv2.waitKey(25) == 32:
        break