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

# Grab Holistic Model from MediaPipe
# Also initialize it.
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
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

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_model.process(image)
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
    if results is not None:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(
                color=(255,0,255),
                thickness=1,
                circle_radius=1
            ),
            mp_drawing.DrawingSpec(
                color=(0,255,255),
                thickness=1,
                circle_radius=1
            )
        )

        # Drawing Right hand Land Marks
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        # Drawing Left hand Land Marks
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
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