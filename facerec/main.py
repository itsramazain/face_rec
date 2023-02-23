import cv2 as cv
import numpy as np
rec_cascade=cv.CascadeClassifier('C:\\Users\\sawa\\Desktop\\faces_rec.xml')
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_train2.yml')
people=['inaam','rama']
capture=cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    if isTrue:
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        face_rect=rec_cascade.detectMultiScale(gray, 1.1, minNeighbors=5)
        for (x,y,w,h) in face_rect:
            facern=gray[y:y+h,x:x+w]
            cv.rectangle(gray,(x,y),(x+w,y+h),(0,0,0),thickness=1)
            labell,sure=face_recognizer.predict(facern)
            cv.putText(gray,f'{people[labell]} with sure {sure}',(x,y+h),cv.FONT_HERSHEY_SIMPLEX,1.0,1)
        cv.imshow('who',gray)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()
