count=0
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import timedelta
#import csv




path = 'images'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # input("Press Enter to continue...")
print(classNames)
# input("Press Enter to continue...")



def findEncodings(images):
    encodelist = []
    i = 0
    for imgs in images:
        i += 1
        try:
            # img = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(imgs)[0]
            encodelist.append(encode)
            # print(f"Face encoding added for image {i}")
        except Exception as e:
            print(f"Error while processing image {imgs}: {e}")
    return encodelist




def updatecount():
    global count
    count = count + 1
    return count

# def checktimeandmarkAttendance(name):
#     s = ""
#     file = open("scheduledtime.txt", "r")
#     f2 = open("preprocessedtime.txt", "w")
#     for line in file:
#         for character in line:
#             if (character == "T"):
#                 s = s + " "
#             else:
#                 s = s + character
#         f2.write(s)
#         s = ""
#     # print(datetime.strptime(s[0],"%y/%m/%d %H:%M"))
#     print(s)
#     file.close()
#     f2.close()

#     f2 = open("preprocessedtime.txt", "r")
#     for line in f2:
#         # print("helloooo")
#         s = line
#         # print(s)
#         s = s.rstrip("\n")
#         tim = datetime.strptime(s, "%Y-%m-%d %H:%M")
#         timfinal = tim + timedelta(minutes=10)
#         if (datetime.now() > tim and datetime.now() < timfinal):
#             with open('Attendance.csv', 'r+') as f:
#                 s = "Session starting at " + str(tim) + " and ending " + str(timfinal)
#                 flg = 0
#                 for line in f:
#                     line=line.strip("\n")
#                     print(line)
#                     if line == s:
#                         flg = 1

#                 if flg == 0:
#                     f.writelines(f'\n{s}')
#                     header = "Sr no, Name. Entry Time"
#                     f.writelines(f'\n{"Sr no"},{"Name"},{"Time"}')
#                 markAttendance(name)


#         else:
#             print("nop", tim, " ", timfinal)
#         s = ""

#         print(datetime.now())



def markAttendance(name):
    with open('Attendance.csv', 'w+') as f:
        insession = 0
        for line in f:
            if "Session" in line:
                insession = 0
            if name in line:
                insession = 1

        if insession == 1:
            return

        #for line in myDataList:
        #    entry = line.split(',')   #for line in myDataList:
        #         #    entry = line.split(',')
        #         #    namelist.append(entry[0])
        #         #if name not in namelist:
        #    namelist.append(entry[0])
        #if name not in namelist:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        x = updatecount()
        f.writelines(f'\n{(x)},{name},{dtString}')
            #s = ""
            #s = str(x) + ", " + name + ", " + str({dtString}) + "\n"
            #f.write(s)



# ... (Previous code remains the same)
print(images)
# input("printed images")
encodeListKnown = findEncodings(images)

print('Encoding Complete', encodeListKnown)
# input("encoded images")


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        print("Not Success")
        continue  # Skip this iteration if the frame is not captured successfully
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    if len(encodesCurFrame) == 0 or len(encodeListKnown) == 0:
        continue

    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) == 0:
            continue

        matchIndex = np.argmin(faceDis)
        mindis = np.amin(faceDis)

        if mindis < 0.39:
            name = classNames[matchIndex].upper()
            name = name.split(" ")
            name = name[0]
        else:
            name = "Unknown"

        if name[0][0] == "0" or name[0][0] == "1" or name == "Unknown":
            name = "Unrecognized"

        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if name == "Unrecognized":
            nothin = 0
        else:
            cv2.imshow('Webcam', img)
            cv2.waitKey(0)  # Pause screen until a key is pressed
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit the loop
        break
















