import cv2 as cv

# Getting the classifier for detecing the faces
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initiating webcam
webcam = cv.VideoCapture(0)

while True:
    succesful_frame_load, frame = webcam.read()

    # Converting the frame to grayscale for detecting the faces through brightness variation.
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if succesful_frame_load:

        # Getting the faces
        faces = face_detector.detectMultiScale(grey_frame, minSize = (40,40), minNeighbors = 8)  

        # Drawing rectangles on every face identified.      
        for x,y,w,h in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv.imshow('Video',frame)

    # Press 'q' to quit.
    if cv.waitKey(1) == ord('q'):
        break 

# Releasing the captured webcam and destroying the indow.
webcam.release()
cv.destroyAllWindows()