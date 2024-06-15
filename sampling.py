import cv2
import numpy as np
import pyttsx3

k = pyttsx3.init()
sound = k.getProperty('voices')
k.setProperty('voice', sound[0].id)
k.setProperty('rate', 130)
k.setProperty('pitch', 200)


def speak(text):
    k.say(text)
    # k.startLoop()


face_classifier = cv2.CascadeClassifier("C:/Users/DELL/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]

    return cropped_face



cap = cv2.VideoCapture(0)
count=0

while True:
    ret,frame = cap.read()
    if face_extractor(frame) is not None:
        # speak("wait sample pic is processing")
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path = "D:/kavach/Face-Recognition-Project-master/Face-Recognition-Project-master/sample/user"+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)

    else:
        print("Face not found !!")
        
        pass

    if cv2.waitKey(1) == 13 or count == 250:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting samples completed *_*")
