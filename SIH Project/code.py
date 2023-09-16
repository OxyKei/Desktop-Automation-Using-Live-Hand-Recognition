import cv2
import mediapipe as mp
import time
import pyautogui
import winsound
import os
import sys
import face_recognition
import numpy as np
import math
import ctypes

# Global variables
# For Face-recognition
path = "F:\HandTracking(HACKLATHON)\Images"
images = []
classNames = []
mylist = os.listdir(path)
decide = bool
# for hand-recognition
handNo = 0
draw = True
tipIds = [4, 8, 12, 16, 20]
t = True
xList = []
yList = []
bbox = []
#Appending of all images in the folder to a list
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
a=True


###########################
# HandRecognition function with gestures
###################################################################################################
def HandRecognition():
    global t
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    while True:
        success, img = cam.read()
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        # re = hands.process(imgRGB)
        # Drawing the circles and connecting the dots on the landmarks
        # imgRGB=detector.findHands(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # if id ==0:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        lmlist = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)

                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        # print(lmlist,bbox)

        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]
            x3, y3 = lmlist[16][1:]
            x4, y4 = lmlist[20][1:]
            x5, y5 = lmlist[4][1:]
            # print(x1,y1,x2,y2)

        if len(lmlist) != 0:
            fingers = []
            for id in range(0, 5):
                if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            print(fingers)

            if len(lmlist) != 0:
                x1, y1 = lmlist[4][1], lmlist[4][2]
                x2, y2 = lmlist[8][1], lmlist[8][2]
                x3, y3 = lmlist[12][1], lmlist[12][2]
                x4, y4 = lmlist[16][1], lmlist[16][2]
                x5, y5 = lmlist[20][1], lmlist[20][2]
                '''
                ca1, ca2 = (x1 + x2) // 2, (y1 + y2) // 2
                ca1, ca2 = (x3 + x2) // 2, (y3 + y2) // 2

                cv2.circle(imgRGB, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(imgRGB, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(imgRGB, (x1, x2), (x2, y2), (255, 0, 255), 3)
                cv2.circle(imgRGB, (ca1, ca2), 15, (255, 0, 255), cv2.FILLED)
                '''
                length = math.hypot(x2 - x1, y2 - y1)
                length2 = math.hypot(x3 - x2, y3 - y2)
                length3 = math.hypot(x5 - x1, y5 - y1)
                length4 = math.hypot(x3 - x1, y3 - y1)
                length5 = math.hypot(x4 - x1, y4 - y1)

            ############################################################
            # Gestures start here
            #############################
            #for opening a new word file
            if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                while t:
                        os.startfile("C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE")
                        os.system("C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE")
                        t = False

            #for selecting all text/files
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

                while t:
                    pyautogui.hotkey('altleft', 'Tab')
                    winsound.PlaySound("Selectall.wav", winsound.SND_ASYNC)
                    pyautogui.hotkey('ctrlleft', 'a')
                    time.sleep(2)
                    t = False
            # for ctrlx(cut)
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and length2 < 40:

                 while t:
                    pyautogui.hotkey('ctrlleft', 'x')
                    t = False
                    winsound.PlaySound("Cut.wav", winsound.SND_ASYNC)
                    time.sleep(2)

            # For Ctrlv(paste)
            elif fingers[0] == 0 and  fingers[1] == 1and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:

                while t:
                    pyautogui.hotkey('ctrlleft', 'v')
                    t = False
                    winsound.PlaySound("paste.wav", winsound.SND_ASYNC)
                    time.sleep(2)

            # For Ctrls(save)
            elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and length < 30:

                while t:
                    pyautogui.hotkey('ctrlleft', 's')
                    t = False
                    winsound.PlaySound("save.wav", winsound.SND_ASYNC)
                    time.sleep(2)

            # for altf4(closing)
            elif fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

                while t:
                    pyautogui.hotkey('altleft', 'F4')
                    t = False
                    winsound.PlaySound("Exit.wav", winsound.SND_ASYNC)
                    time.sleep(2)

            elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

                while t:
                    winsound.PlaySound("Exit.wav", winsound.SND_ASYNC)
                    ctypes.windll.user32.LockWorkStation()
                    t = False
                    time.sleep(2)

            # for resetting/confirming
            if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                t = True
        ###################################
        # Gestures end here
        ###############################################################

        ################################################################
        # To show Fps
        ######################
            ptime = 0
            cTime = time.time()
            fps = 1 / (cTime - ptime)
            ptime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        ######################
        # Fps ends here
        ################################################################
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break



#####################################################################################################
# HandRecognition function ends
################################################
wcam, hcam = 420, 420
cam = cv2.VideoCapture(0)
cam.set(3, wcam)
cam.set(4, hcam)

# HandRecognition()
########################################
# Face Recognition function starts
######################################################################################################
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # for i in range(0, len(encodeList)):
        #     print(encodeList[i])
        try:
            encoded_face = face_recognition.face_encodings(img)[0]
        except IndexError as e:
            print(e)
            sys.exit(1)
        encodeList.append(encoded_face)
    return encodeList

def face_recog():
    encoded_face_train = findEncodings(images)
    print(mylist)
    while True:
        success, img = cam.read()
        img = cv2.flip(img, 1)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            name = "Not Recognized"
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)

            print(matches[matchIndex])
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

            ################################################################
            # To show Name box
            ######################
            if name=="Not Recognized":
                cnt = pyautogui.alert('Face not Recognized click ok to get Access')
                if cnt == "OK":
                    password = pyautogui.password("Enter the password", "BTC", mask='*')
                    if password == "BTC":
                        os.startfile("E:\HandTracking(HACKLATHON)\Images")
                        os.system("E:\HandTracking(HACKLATHON)\Images")
                    else:

                        pyautogui.alert("password is incorrect")
                        quit()
            y1, x2, y2, x1 = faceloc
            # since we scaled down by 4 times
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            ######################
            # Name box ends here
            ################################################################
            return matches[matchIndex]

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break



#######################################################################################################
# face recognition function ends
#########################################
x=face_recog()
while a:
    if x==True:
        pyautogui.alert("Face has been recognised.Implementing Hand Recognition.")
        HandRecognition()
    elif(x!=True):
        pyautogui.alert("The face isn't recognised...trying again!")
        x=face_recog()