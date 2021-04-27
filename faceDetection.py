import cv2 as cv

# in case you want to rescale the frame i created it for you but im not going to use this
# create a rescale frame function to rescale the size of frame
def rescaledFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#add a OpenCV open source haar_cascade file in your current directory 
#helps to detect faces
haar_cascade = cv.CascadeClassifier("haar_face.xml")

def face_detects_photos():
    img_src = "group of faces.jpg"

    #reads original image
    image = cv.imread(img_src)
    cv.imshow("Original", image)

    #now detects faces from the photos
    face_detects = haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=8)

    #check the total number of faces we found in that image
    print(f"{len(face_detects)} faces were found in this image.")

    for (x,y,w,h) in face_detects:
        cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0))
    
    cv.imshow("Face Detects", image)

    cv.waitKey(0)

def realDetection():
    # capture your webcame using VideoCapture function
    capture = cv.VideoCapture(0)


    #use while true it will take frames one by one continues until you 
    # terminate yourself
    while True:
        istrue, frame = capture.read()

        face_detects = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=8)

        for (x,y,w,h) in face_detects:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=4)
        
        cv.imshow("FACE DETECTION", frame)

        if cv.waitKey(100) & 0xFF==ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()

# first function will detects faces from photos
face_detects_photos()

#this function will detect real time faces using your webcam
#If you want to use comment out it. 
#realDetection()