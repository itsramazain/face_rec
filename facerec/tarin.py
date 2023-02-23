import cv2 as cv
import os
import numpy as np
rec_cascade=cv.CascadeClassifier('C:\\Users\\sawa\\Desktop\\faces_rec.xml')
pepole=['inaam','rama']
featuers=[]
labels=[]
photo_dir='C:\\Users\\sawa\\Desktop\\opencv-course-master\\Resources\\Faces\\new'
def create_tarin():
    for folder in os.listdir(photo_dir):
        person_folder=os.path.join(photo_dir,folder)
        label=pepole.index(folder)
        for pic in os.listdir(person_folder):
            pic_path=os.path.join(person_folder,pic)
            img=cv.imread(pic_path)
            gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            race_rect=rec_cascade.detectMultiScale(gray, 1.1, minNeighbors=5)
            for (x,y,w,h) in race_rect:
                featuers.append(gray[y:y+h,x:x+w])
                labels.append(label)
create_tarin()
features=np.array(featuers,dtype='object')
labels=np.array(labels)
np.save('featuers2.npy',features)
np.save('labels2.npy',labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save('face_train2.yml')


