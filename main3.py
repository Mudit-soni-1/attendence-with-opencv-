import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime
#imported all needed module

path = 'images'
images = []
person_name = []
myList = os.listdir(path)
print(myList)
#to show that the list is created from dir to path of image
for c in myList:
    curret_Ing = cv2.imread(f'{path}/{c}') 
    images.append(curret_Ing)
    person_name.append(os.path.splitext(c)[0])
print(person_name)
#to print only name of student
def faceEncodings(images):
    Encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #TO CONVERT BGR FORMAT INTO RGB FORMAT AS WE USED CV2
        encode = face_recognition.face_encodings(img)[0]
        Encode_list.append(encode)
    return Encode_list    

encodelistknown =faceEncodings(images)
# dlib as dlib used to distingused face in 128 variable/factors
#so matrix is formed by hog algorithm
print("ALL ENCODING IS COMPLETE")

#for attendence marking
def att(name):
    with open('att1.csv','r+')as f:
        myDatalist = f.readlines()
        namelist = []
        for i in myDatalist:
            entry = i.split(':')
            namelist.append(entry[0])

        if name not in namelist:
            time_now = datetime.now()
            time_Str = time_now.strftime('%H:%M:%S')
            date_Str = time_now.strftime('%j/%m/%Y')
        f.writelines(f'\n{name}, {time_Str} , {date_Str}')


cap = cv2.VideoCapture(0)
#for laptop cam is 0 so use it

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame , (0,0) , None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    face_current = face_recognition.face_locations(faces)
    encodeCurrentframe = face_recognition.face_encodings(faces, face_current)

    for faceloc,encodeface in zip(face_current,encodeCurrentframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedis = face_recognition.face_distance(encodelistknown, encodeface)
        # for distance and encoding for matching 
        # for min facedistance it match
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = person_name[matchIndex].upper()
            #print(name)

            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            #as we rsize it by dividing by 4 so to create a box we need to mul by 4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame, name, (x1 +6 ,y2 -6) , cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 4)
            #green color
            #calling attendence method
            att(name)

    cv2.imshow("webcam" , frame)
    if cv2.waitKey(1)==13:
        #(enter)key asci
        break
   
   
cap.release()
cv2.destroyAllWindows()       



