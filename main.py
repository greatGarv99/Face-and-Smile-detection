import cv2 as cv

# Getting the classifiers for detecing the faces and smiles.
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv.CascadeClassifier('haarcascade_smile.xml')

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

            # Get the current face, to detect the smile.
            current_face = frame[y:y+h, x:x+h]
            grey_face = cv.cvtColor(current_face, cv.COLOR_BGR2GRAY)

            smile = smile_detector.detectMultiScale(grey_face, scaleFactor = 1.7, minNeighbors = 20)

            # Comment out the next follwing lines to draw rectangle around the smile.
            # for x_,y_,w_,h_ in smile:
            #     cv.rectangle(current_face, (x_,y_), (x_+w_,y_+h_), (0,0,255), 2)

            # Indicating if a smile indicated.
            if len(smile)>0:
                cv.putText(
                    img = frame, 
                    text = 'smiling', 
                    org = (x, y+w+40),
                    fontFace = cv.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color = (255,255,255),
                )

    cv.imshow('Video',frame)

    # Press 'q' to quit.
    if cv.waitKey(1) == ord('q'):
        break 

# Releasing the captured webcam and destroying the indow.
webcam.release()
cv.destroyAllWindows()