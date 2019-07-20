#Write a python script that captures images from your webcam video stream
#Extracts all faces from the image frame(using haarcascades)
#stores thqe face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np

#initialize the camera
cap=cv2.VideoCapture(0)

#Face Detection
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
face_data=[]
dataset_path='./data/'

file_name=input("Enter the name of the person")

while True:
	ret,frame=cap.read()

	if ret==False:
		continue;

	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces=face_cascade.detectMultiScale(frame,1.3,5)
	#for multiple faces, sort the faces based upon area,(x,y,w,h) therefore (w*h) would yield the area
	faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)

	for(x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
		#extract (crop out required face) : Region of Interest
		#offset is amount of padding we want around all edges
		offset=10
		face_section= frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		#store every 10th face
		skip+=1
		if(skip%10==0):
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)
	
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break

#convert face list array into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
#(number of images ,30000) since size was 100*100 and it was a RGB image
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved at ",dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()

