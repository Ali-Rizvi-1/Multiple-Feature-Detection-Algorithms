import cv2
import numpy

# path to the video
video_path = 'C:/Habib University/OneDrive - Habib University/University/7th Semester/Digital Image Processing/Project/FD in Dynamic Environment/Dynamic1.mp4'

capture = cv2.VideoCapture(video_path)

# initial the frame number of the video frames 
frameNr = 0

# total keypoints in the video via SIFT
total_kp_sift = 0
# total keypoints in the video via ORB
total_kp_orb = 0

while(True):

    flag, frame = capture.read()

    if flag:

        print(frameNr)

        # convert image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints with SIFT
        kp_sift = sift.detect(gray,None)

        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp_orb = orb.detect(gray,None)

        total_kp_sift += len(kp_sift)
        total_kp_orb += len(kp_orb)

        print('Dimensions of Frame ', str(frameNr),frame.shape)
        print('SIFT Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_sift))
        print('ORB Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_orb))
        
    else:
        break # exist the loop when frames are finished
 
    frameNr = frameNr+1 # increment in frame number

print(" = "*12)
print('\n'*5)
print('Total SIFT keypoints in the video', total_kp_sift)
print('Total ORB keypoints in the video', total_kp_orb)
