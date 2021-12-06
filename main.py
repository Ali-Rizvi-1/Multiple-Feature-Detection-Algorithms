import cv2
import numpy

# path to the video
video_path = 'C:/Habib University/OneDrive - Habib University/University/7th Semester/Digital Image Processing/Project/FD in Dynamic Environment/Dynamic1.mp4'

capture = cv2.VideoCapture(video_path)

# initial the frame number of the video frames 
frameNr = 0

while(True):

    flag, frame = capture.read()

    if flag:

        print(frameNr)

    else:
        break # exist the loop when frames are finished
 
    frameNr = frameNr+1 # increment in frame number

