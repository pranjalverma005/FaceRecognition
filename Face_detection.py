#Read a video stream from Camera(Frame By Frame)
import cv2

#step1) capture the device from which you'll read image
#       '0' means the default webcam, otherwise specify for multiple webcam
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#cap.read()->returns 2 things
#1) boolean value->whether frame has been captured or not(eg. webcam not started)
#2) frame itself
while True:
  ret,frame=cap.read()
  gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  if ret==False:
    continue

  #argument 2) scaling factor: how much image size is reduced at each image scale
  #arguemnt 3) minNeighbors: how many neighbors each candidate rectangle should have to retain it
  #it returns coordinate(x,y,width,height),for multiple faces, it returns a tuple of coordinate 
  faces=face_cascade.detectMultiScale(gray_frame,1.3,5)

  #for rectangle construction
  #!)frame 2)first coordinate 3)diagonally opposite coordinate 4)color of border 5)
  for(x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,0,255),3)

  cv2.imshow('Video Frame',frame)
  cv2.imshow('Gray Frame',gray_frame)

  #wait for user input-q, then you will stop the loop
  #ord->returns ascii value
  #0xFF->8 numbers of 1's

  key_pressed=cv2.waitKey(1) & 0xFF
  if key_pressed==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()