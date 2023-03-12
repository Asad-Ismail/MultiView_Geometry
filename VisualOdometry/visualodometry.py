import cv2
import numpy as np

# Load the video frames
cap = cv2.VideoCapture("video.mp4")

# Initialize the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize the trajectory
trajectory = []

# Create a feature detector
feature_detector = cv2.ORB_create()

# Create a feature matcher
feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Loop through the video frames
while True:
    # Read the next frame
    ret, curr_frame = cap.read()
    if not ret:
        break
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Detect and extract features from the previous and current frames
    prev_keypoints, prev_descriptors = feature_detector.detectAndCompute(prev_gray, None)
    curr_keypoints, curr_descriptors = feature_detector.detectAndCompute(curr_gray, None)

    # Match the features between the previous and current frames
    matches = feature_matcher.match(prev_descriptors, curr_descriptors)

    # Estimate the motion between the previous and current frames using RANSAC
    prev_points = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    E, mask = cv2.findEssentialMat(curr_points, prev_points, focal=1.0, pp=(0, 0))
    _, R, t, mask = cv2.recoverPose(E, curr_points, prev_points)

    # Compute the camera pose
    pose = np.hstack((R, t))
    if len(trajectory) == 0:
        trajectory.append(pose)
    else:
        last_pose = trajectory[-1]
        curr_pose = last_pose @ pose
        trajectory.append(curr_pose)

    # Update the previous frame and keypoints
    prev_gray = curr_gray
    prev_keypoints = curr_keypoints

    # Visualize the trajectory and current frame
    for i in range(len(trajectory)):
        cv2.circle(curr_frame, (int(trajectory[i][0, 3]), int(trajectory[i][2, 3])), 1, (0, 0, 255), 2)
    cv2.imshow("

