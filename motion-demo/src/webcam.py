# Webcam initialization
import cv2 as cv
import mediapipe as mp
import time

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