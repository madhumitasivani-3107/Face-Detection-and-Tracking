import cv2

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg) # Load the algorithm

cam = cv2.VideoCapture(0) # Initializing the camera
flag = 0
while True:
    _,img = cam.read() # reading frame from the camera
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting gray scale image
    
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) # gives the parameter of the face->src, scalefactor,minNeighbors
    
    for(x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) # Draw rectangle with the face-> src, coordinate, color, thickness
        flag = 1
        
    if flag == 1:
        print("Person Detected")
        
    else:
        print("No person is detected")
    
    cv2.imshow("FaceDetection", img) #Display the face rectangle
    
    key = cv2.waitKey(1) & 0xFF # For closing the window
    if key == ord("q"):
        break
        
cam.release()  
cv2.destroyAllWindows()
