# Webcam initialization
import cv2 as cv
import mediapipe as mp
import time
import numpy as np

print(cv.__version__)

# Webcam initialization
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

previousTime = 0
currentTime = 0

# Frame skipping for performance optimization
frame_count = 0
skip_frames = 3  # Process every 3rd frame for MediaPipe analysis
last_results = None  # Store the last MediaPipe results

# Playing with properties
print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Grab Pose Model from MediaPipe (instead of Holistic for better performance)
# Also initialize it.
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Load T-pose image
tpose_image = cv.imread('/home/allnats/Hackathon/stupid-hackathon/motion-demo/images/tpose.jpg')
tpose_detected = False

def detect_tpose(landmarks):
    """
    Detect T-pose based on arm angles and body posture
    Returns True if T-pose is detected
    """
    if not landmarks:
        return False

    # Get key landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Calculate arm angles (shoulder-elbow-wrist)
    def calculate_angle(p1, p2, p3):
        """Calculate angle between three points"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])

        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norms == 0:
            return 0

        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    # Check if arms are extended horizontally (T-pose criteria)
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Check if arms are roughly horizontal
    left_arm_horizontal = abs(left_elbow.y - left_shoulder.y) < 0.1
    right_arm_horizontal = abs(right_elbow.y - right_shoulder.y) < 0.1

    # T-pose conditions:
    # 1. Arms are roughly straight (angle close to 180 degrees)
    # 2. Arms are horizontal
    # 3. Both arms are visible and confident
    is_tpose = (
        left_arm_angle > 150 and right_arm_angle > 150 and  # Arms extended
        left_arm_horizontal and right_arm_horizontal and    # Arms horizontal
        left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
        left_elbow.visibility > 0.5 and right_elbow.visibility > 0.5 and
        left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5
    )

    return is_tpose


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Resize the frame
    frame = cv.resize(frame, (800, 600))

    # Frame skipping logic - only process MediaPipe on certain frames
    frame_count += 1

    if frame_count % skip_frames == 0:
        # Convert color format to the default BGR to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Making predictions using pose model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose_model.process(image)
        image.flags.writeable = True

        # Store results for reuse
        last_results = results

        # Converting back the RGB image to BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        # Reuse last results and just convert frame to display format
        image = frame.copy()
        results = last_results

    # Only draw landmarks if we have results
    if results is not None and results.pose_landmarks:
        # Drawing Pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(245,117,66),
                thickness=2,
                circle_radius=2
            ),
            mp_drawing.DrawingSpec(
                color=(245,66,230),
                thickness=2,
                circle_radius=2
            )
        )

        # Detect T-pose
        current_tpose = detect_tpose(results.pose_landmarks)

        if current_tpose:
            tpose_detected = True
            # Display T-pose status on main image
            cv.putText(image, "T-POSE DETECTED!", (10, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            # Show T-pose image
            if tpose_image is not None:
                cv.imshow("T-Pose Reference", tpose_image)
        else:
            tpose_detected = False
            # Remove T-pose image window if it exists
            try:
                cv.destroyWindow("T-Pose Reference")
            except cv.error:
                pass
    else:
        # No pose detected, make sure T-pose window is closed
        tpose_detected = False
        try:
            cv.destroyWindow("T-Pose Reference")
        except cv.error:
            pass

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv.putText(image, str(int(fps))+" FPS", (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    # Displays the frame.
    cv.imshow(
        winname="Live Camera Feed",
        mat=image
    )

    # Let's exit on 'q' key press
    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()