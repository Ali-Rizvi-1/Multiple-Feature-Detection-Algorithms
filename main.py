import cv2
import numpy as np

# path to the video
video_path = 'C:/Habib University/OneDrive - Habib University/University/7th Semester/Digital Image Processing/Project/FD in Dynamic Environment/Dynamic1.mp4'

capture = cv2.VideoCapture(video_path)

# initial the frame number of the video frames 
frameNr = 0

# total keypoints in the video via SIFT
total_kp_sift = 0
# total keypoints in the video via ORB
total_kp_orb = 0
# total keypoints in the video via Harris cornor detector
total_kp_hcd = 0
# total keypoints in the video via FAST
total_kp_fast = 0
# total keypoints in the video via BRIEF
total_kp_brief = 0
# total keypoints in the video via BRISK
total_kp_brisk = 0


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

        # find the keypoints using Harris cornor detector
        gray_hcd = np.float32(gray)
        kp_hcd = cv2.cornerHarris(gray_hcd,2,3,0.04)

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        # find and draw the keypoints
        kp_fast = fast.detect(gray,None)

        # Initiate BRISK object with default values
        brisk = cv2.BRISK_create()
        # find and draw the keypoints
        kp_brisk = brisk.detect(gray,None)

        # # Initiate STAR detector 
        # star = cv2.xfeatures2d.StarDetector_create()
        # # Initiate BRIEF extractor
        # brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        # # find the keypoints with STAR
        # kp = star.detect(gray,None)
        # # compute the descriptors with BRIEF
        # kp_brief, des = brief.compute(gray, kp)

        total_kp_sift += len(kp_sift)
        total_kp_orb += len(kp_orb)
        total_kp_hcd += len(kp_hcd)
        total_kp_fast += len(kp_fast)
        #total_kp_brief += len(kp_brief)
        total_kp_brief += len(kp_brisk)

        print('Dimensions of Frame ', str(frameNr),frame.shape)
        print('SIFT Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_sift))
        print('ORB Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_orb))
        print('HCD Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_hcd))
        print('FAST Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_fast))
        print('BRISK Keypoints in Frame ', str(frameNr) , ' length of keypoints ', len(kp_brisk))

    else:
        break # exist the loop when frames are finished
 
    frameNr = frameNr+1 # increment in frame number

print(" = "*12)
print('\n'*5)
print('Total SIFT keypoints in the video', total_kp_sift)
print('Total ORB keypoints in the video', total_kp_orb)
print('Total HCD keypoints in the video', total_kp_hcd)
print('Total FAST keypoints in the video', total_kp_fast)
print('Total BRIEF keypoints in the video', total_kp_brief)
print('Total BRIEF keypoints in the video', total_kp_brisk)
 
capture.release()