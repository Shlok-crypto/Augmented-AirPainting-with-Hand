import cv2 as cv
import numpy as np
import time
import os
import HandDetectionModule as htm


##################
BruhThickness = 15
EraserThickness = 100
##################

# Extrating all images from Header Folder
folderpath = "Header"
mylist = os.listdir(folderpath)
overlayList = []
for imgPath in mylist:
    image = cv.imread(f'{folderpath}/{imgPath}')
    overlayList.append(image)

# default overlay image and Draw Color, xp ,yp
header = overlayList[0]
drawColor = (156, 118, 237)
xp,yp = 0, 0

capture = cv.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)

detector = htm.handDetector(detectionCon=0.85)

# Drawing Canvas

canvas = np.zeros((720,1280,3),np.uint8)
preTime = 0
curTime = 0

while True:
    _,frame = capture.read()
    frame = cv.flip(frame,1)
    # find Landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0 :
        #print(lmList[8])

    #Tip of point index finger and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2,= lmList[12][1:]

    # how many fingers are up
        fingers = detector.fingersUp()
    # selection mode
        if fingers[1] and fingers[2]:
            print("Selection mode")
        # Making Selection
            if y1 < 125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor= (156, 118, 237)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)

            cv.rectangle(frame,(x1,y1-20),(x2,y2+20),drawColor,-1)
            xp, yp = x1, y1

        # drawing mode
        if fingers[1] and fingers[2]==False:
            cv.circle(frame, (x1, y1), 5, drawColor, 10)
            print("Drawing mode")

            if xp == 0 and yp == 0:
                xp,yp = x1, y1

            if drawColor == (0,0,0):
                cv.line(frame, (xp, yp), (x1, y1), drawColor, EraserThickness)
                cv.line(canvas, (xp, yp), (x1, y1), drawColor, EraserThickness)
            else:
                cv.line(frame,(xp,yp),(x1,y1),drawColor,BruhThickness)
                cv.line(canvas,(xp,yp),(x1,y1),drawColor,BruhThickness)

            xp,yp = x1, y1

    canvasGray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, canvasInv = cv.threshold(canvasGray, 50, 255, cv.THRESH_BINARY_INV)

    canvasInv = cv.cvtColor(canvasInv,cv.COLOR_GRAY2BGR)

    frame = cv.bitwise_and(frame,canvasInv)
    frame = cv.bitwise_or(frame,canvas)



# Setting the Header Image
    frame[0:125,0:1280]=header
    cv.imshow("image",frame)
    cv.imshow("canvas",canvas)
    cv.imshow("canvasINV",canvasInv)

    if cv.waitKey(1) == ord('q'):
        break
capture.release()
cv.destroyAllWindows()

