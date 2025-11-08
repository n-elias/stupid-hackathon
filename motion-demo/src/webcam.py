# Webcam initialization
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import random
import pygame

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

# Grab Pose, Hand, and Face Models from MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Initialize models
pose_model = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Initialize pygame mixer for sound effects
pygame.mixer.init()

# Load sound effects
pose_sounds = {
    'tpose': pygame.mixer.Sound('./sfx/vine-boom.mp3'),
    'armcross': pygame.mixer.Sound('./sfx/i-i-i-be-poppin-bottles.mp3'),
    'monkeyfinger': pygame.mixer.Sound('./sfx/ding.mp3'),
    'shocked_face': pygame.mixer.Sound('./sfx/faaah.mp3'),
    'spiderman': pygame.mixer.Sound('./sfx/spiderman-meme-song.mp3')
}
failure_sound = pygame.mixer.Sound('./sfx/bo-womp.mp3')

# Game state variables
game_state = {
    'current_pose': None,
    'pose_start_time': None,
    'game_phase': 'startup',  # 'startup', 'waiting', 'challenge', 'result'
    'challenge_duration': 5.0,  # 5 seconds to complete pose
    'wait_duration': 3.0,      # 3 seconds between challenges
    'startup_duration': 5.0,   # 5 seconds startup delay
    'score': 0,
    'round': 0,
    'app_start_time': time.time()  # Record when app started
}

# Available poses for the game
available_poses = ['tpose', 'armcross', 'monkeyfinger', 'shocked_face', 'spiderman']

# Load pose reference images
pose_images = {
    'tpose': cv.imread('./images/tpose.jpg'),
    'armcross': cv.imread('./images/armcross.jpg'),
    'monkeyfinger': cv.imread('./images/monkeyfinger.png'),
    'shocked_face': cv.imread('./images/shocked_face.png'),
    'spiderman': cv.imread('./images/spiderman.png')
}

# Dictionary to store detected poses
detected_poses = {
    'tpose': False,
    'armcross': False,
    'monkeyfinger': False,
    'shocked_face': False,
    'spiderman': False
}

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

def detect_armcross(landmarks):
    """
    Detect arm cross pose based on arm positions
    Returns True if arm cross pose is detected
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
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Check if arms are crossed (wrists are closer to opposite shoulders)
    left_wrist_to_right_shoulder = abs(left_wrist.x - right_shoulder.x)
    right_wrist_to_left_shoulder = abs(right_wrist.x - left_shoulder.x)
    shoulder_distance = abs(left_shoulder.x - right_shoulder.x)

    # Check if elbows are at reasonable height (not too low)
    left_elbow_height = left_elbow.y < left_shoulder.y + 0.2
    right_elbow_height = right_elbow.y < right_shoulder.y + 0.2

    # Arm cross conditions:
    # 1. Arms are bent (angle less than 120 degrees)
    # 2. Wrists are crossed over to opposite sides
    # 3. Elbows are at reasonable height
    # 4. All landmarks are visible
    is_armcross = (
        left_arm_angle < 120 and right_arm_angle < 120 and  # Arms bent
        left_wrist_to_right_shoulder < shoulder_distance * 1.5 and  # Left wrist near right side
        right_wrist_to_left_shoulder < shoulder_distance * 1.5 and  # Right wrist near left side
        left_elbow_height and right_elbow_height and  # Elbows at reasonable height
        left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
        left_elbow.visibility > 0.5 and right_elbow.visibility > 0.5 and
        left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5
    )

    return is_armcross

def detect_monkeyfinger(landmarks, hand_results=None):
    """
    Detect monkey finger pose based on right hand position near mouth
    Enhanced with hand landmarks for better finger detection
    Returns True if monkey finger pose is detected
    """
    if not landmarks:
        return False

    # Get key landmarks
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    mouth_left = landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
    mouth_right = landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]

    # Calculate mouth center position
    mouth_center_x = (mouth_left.x + mouth_right.x) / 2
    mouth_center_y = (mouth_left.y + mouth_right.y) / 2

    # Enhanced detection using hand landmarks if available
    finger_to_mouth_distance = float('inf')
    hand_detected = False

    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]

            # Calculate distance from index finger to mouth
            finger_distance = np.sqrt(
                (index_finger_tip.x - mouth_center_x)**2 +
                (index_finger_tip.y - mouth_center_y)**2
            )

            if finger_distance < finger_to_mouth_distance:
                finger_to_mouth_distance = finger_distance
                hand_detected = True

    # Fallback to wrist position if no hand landmarks
    if not hand_detected:
        finger_to_mouth_distance = np.sqrt(
            (right_wrist.x - mouth_center_x)**2 +
            (right_wrist.y - mouth_center_y)**2
        )

    # Calculate arm angle (shoulder-elbow-wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Check if right hand is raised (wrist above elbow)
    hand_raised = right_wrist.y < right_elbow.y

    # Check if elbow is bent and hand is near mouth
    elbow_bent = right_arm_angle < 150  # Arm should be bent

    # Enhanced monkey finger conditions with better threshold for finger detection
    threshold = 0.08 if hand_detected else 0.15  # Stricter threshold with hand landmarks

    is_monkeyfinger = (
        finger_to_mouth_distance < threshold and  # Finger/hand close to mouth
        elbow_bent and  # Arm bent
        hand_raised and  # Hand raised
        right_shoulder.visibility > 0.5 and
        right_elbow.visibility > 0.5 and
        right_wrist.visibility > 0.5 and
        nose.visibility > 0.5 and
        mouth_left.visibility > 0.5 and
        mouth_right.visibility > 0.5
    )

    return is_monkeyfinger

def detect_shocked_face(face_results):
    """
    Detect shocked face expression based on facial landmarks
    Returns True if shocked face is detected
    """
    if not face_results or not face_results.multi_face_landmarks:
        return False

    # Get the first face landmarks
    face_landmarks = face_results.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark

    # Key facial landmark indices for shocked expression
    # Eyes - upper and lower eyelids
    left_eye_upper = landmarks[159]  # Left eye upper eyelid
    left_eye_lower = landmarks[145]  # Left eye lower eyelid
    right_eye_upper = landmarks[386] # Right eye upper eyelid
    right_eye_lower = landmarks[374] # Right eye lower eyelid

    # Mouth - corners and center points
    mouth_left = landmarks[61]       # Left mouth corner
    mouth_right = landmarks[291]     # Right mouth corner
    mouth_top = landmarks[13]        # Upper lip center
    mouth_bottom = landmarks[14]     # Lower lip center

    # Eyebrows - inner and outer points
    left_eyebrow_inner = landmarks[70]   # Left eyebrow inner
    left_eyebrow_outer = landmarks[107]  # Left eyebrow outer
    right_eyebrow_inner = landmarks[300] # Right eyebrow inner
    right_eyebrow_outer = landmarks[336] # Right eyebrow outer

    # Calculate eye openness (shocked eyes are wide open)
    left_eye_height = abs(left_eye_upper.y - left_eye_lower.y)
    right_eye_height = abs(right_eye_upper.y - right_eye_lower.y)
    avg_eye_height = (left_eye_height + right_eye_height) / 2

    # Calculate mouth openness (shocked mouth is open)
    mouth_height = abs(mouth_top.y - mouth_bottom.y)
    mouth_width = abs(mouth_left.x - mouth_right.x)

    # Calculate eyebrow elevation (shocked eyebrows are raised)
    left_eyebrow_height = abs(left_eyebrow_inner.y - left_eye_upper.y)
    right_eyebrow_height = abs(right_eyebrow_inner.y - right_eye_upper.y)
    avg_eyebrow_elevation = (left_eyebrow_height + right_eyebrow_height) / 2

    # Shocked face conditions:
    # 1. Wide open eyes (above threshold)
    # 2. Open mouth (height relative to width)
    # 3. Raised eyebrows (above eyes)
    # 4. Symmetrical expression

    wide_eyes = avg_eye_height > 0.015  # Eyes wide open
    open_mouth = mouth_height > 0.02 and mouth_height > mouth_width * 0.3  # Mouth open in surprise
    raised_eyebrows = avg_eyebrow_elevation > 0.025  # Eyebrows raised
    symmetric_eyes = abs(left_eye_height - right_eye_height) < 0.008  # Eyes similarly open

    is_shocked = (
        wide_eyes and
        open_mouth and
        raised_eyebrows and
        symmetric_eyes
    )

    return is_shocked

def detect_spiderman(landmarks):
    """
    Detect Spiderman pose based on body landmarks only
    Classic web-slinging pose: one arm extended forward, crouched stance, dynamic leg position
    Returns True if Spiderman pose is detected
    """
    if not landmarks:
        return False

    # Get key landmarks for body pose
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # Calculate arm angles and positions
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Calculate leg angles (hip-knee-ankle)
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Check for one arm extended forward (Spiderman web-shooting pose)
    # One arm should be more extended than the other
    left_arm_extended = left_arm_angle > 140 and left_wrist.x > left_shoulder.x
    right_arm_extended = right_arm_angle > 140 and right_wrist.x < right_shoulder.x

    # At least one arm should be extended forward
    one_arm_forward = left_arm_extended or right_arm_extended

    # Check for crouched/action stance
    # Body should be leaning forward (nose ahead of hips)
    hip_center_x = (left_hip.x + right_hip.x) / 2
    body_leaning_forward = nose.x > hip_center_x - 0.05  # Slight forward lean tolerance

    # Check leg positioning - one or both legs should be bent (action stance)
    left_leg_bent = left_leg_angle < 160  # Leg bent for dynamic pose
    right_leg_bent = right_leg_angle < 160
    dynamic_legs = left_leg_bent or right_leg_bent

    # Check for asymmetric arm positioning (characteristic of Spiderman pose)
    arm_height_diff = abs(left_wrist.y - right_wrist.y) > 0.1  # Arms at different heights

    # Check body crouch - hips should be lower than shoulders (crouched stance)
    hip_center_y = (left_hip.y + right_hip.y) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    crouched_stance = hip_center_y > shoulder_center_y + 0.05  # Hips lower than shoulders

    # Spiderman pose conditions:
    # 1. One arm extended forward (web-shooting)
    # 2. Dynamic leg positioning (bent/action stance)
    # 3. Body leaning forward or crouched
    # 4. Asymmetric arm positioning
    # 5. All key landmarks visible

    is_spiderman = (
        one_arm_forward and
        dynamic_legs and
        (body_leaning_forward or crouched_stance) and
        arm_height_diff and
        # Visibility checks
        left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
        left_elbow.visibility > 0.5 and right_elbow.visibility > 0.5 and
        left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5 and
        left_hip.visibility > 0.5 and right_hip.visibility > 0.5 and
        left_knee.visibility > 0.5 and right_knee.visibility > 0.5 and
        nose.visibility > 0.5
    )

    return is_spiderman

def detect_poses(pose_landmarks, hand_results=None, face_results=None):
    """
    Unified pose detection function that checks all poses
    Now includes hand and face data for enhanced detection accuracy
    Returns dictionary with detected poses
    """
    poses = {
        'tpose': detect_tpose(pose_landmarks),
        'armcross': detect_armcross(pose_landmarks),
        'monkeyfinger': detect_monkeyfinger(pose_landmarks, hand_results),
        'shocked_face': detect_shocked_face(face_results),
        'spiderman': detect_spiderman(pose_landmarks)
    }
    return poses

def start_new_challenge():
    """Start a new pose challenge"""
    game_state['current_pose'] = random.choice(available_poses)
    game_state['pose_start_time'] = time.time()
    game_state['game_phase'] = 'challenge'
    game_state['round'] += 1
    print(f"Round {game_state['round']}: Show me a {game_state['current_pose'].upper()}!")

def check_challenge_completion(detected_poses):
    """Check if current challenge is completed"""
    if game_state['game_phase'] != 'challenge':
        return

    current_time = time.time()
    elapsed_time = current_time - game_state['pose_start_time']

    # Check if pose was detected
    if detected_poses.get(game_state['current_pose'], False):
        # Success! Play success sound and update score
        pose_sounds[game_state['current_pose']].play()
        game_state['score'] += 1
        game_state['game_phase'] = 'result'
        game_state['pose_start_time'] = current_time
        print(f"SUCCESS! Score: {game_state['score']}")
        return True

    # Check if time ran out
    elif elapsed_time >= game_state['challenge_duration']:
        # Failure! Play failure sound
        failure_sound.play()
        game_state['game_phase'] = 'result'
        game_state['pose_start_time'] = current_time
        print(f"TIME'S UP! Score: {game_state['score']}")
        return False

    return None

def update_game_state():
    """Update game state based on current phase"""
    current_time = time.time()

    if game_state['game_phase'] == 'startup':
        # Check if startup period is over
        if current_time - game_state['app_start_time'] >= game_state['startup_duration']:
            game_state['game_phase'] = 'waiting'
            game_state['pose_start_time'] = current_time
            print("Game starting! Get ready for your first challenge!")

    elif game_state['game_phase'] == 'waiting':
        # Start first challenge or check if wait period is over
        if game_state['pose_start_time'] is None:
            start_new_challenge()
        elif current_time - game_state['pose_start_time'] >= game_state['wait_duration']:
            start_new_challenge()

    elif game_state['game_phase'] == 'result':
        # Wait before starting next challenge
        if current_time - game_state['pose_start_time'] >= game_state['wait_duration']:
            game_state['game_phase'] = 'waiting'
            game_state['pose_start_time'] = current_time

def draw_game_ui(image):
    """Draw game UI elements on the image"""
    current_time = time.time()

    if game_state['game_phase'] == 'startup':
        # Draw startup countdown
        elapsed_time = current_time - game_state['app_start_time']
        remaining_time = max(0, game_state['startup_duration'] - elapsed_time)

        cv.putText(image, "BRAINROT ROULETTE", (150, 250), cv.FONT_HERSHEY_SIMPLEX, 1.5, (28, 28, 28), 3)
        cv.putText(image, f"starting in: {remaining_time:.1f}s", (200, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (76, 96, 166), 3)
        cv.putText(image, "how chronically online are you?", (150, 350), cv.FONT_HERSHEY_SIMPLEX, 0.8, (39, 41, 143), 2)

    elif game_state['game_phase'] == 'challenge':
        # Draw countdown timer
        elapsed_time = current_time - game_state['pose_start_time']
        remaining_time = max(0, game_state['challenge_duration'] - elapsed_time)

        # Draw challenge info
        pose_name = game_state['current_pose'].upper().replace('_', ' ')
        cv.putText(image, f"TIME: {remaining_time:.1f}s", (10, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (93, 54, 145), 3)

        # Show reference image
        if pose_images[game_state['current_pose']] is not None:
            cv.imshow(f"meme: {pose_name}", pose_images[game_state['current_pose']])

    elif game_state['game_phase'] == 'result':
        elapsed_time = current_time - game_state['pose_start_time']
        remaining_wait = max(0, game_state['wait_duration'] - elapsed_time)
        cv.putText(image, f"next meme: {remaining_wait:.1f}s", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (28, 28, 28), 2)

        # Close challenge window
        try:
            cv.destroyWindow(f"Challenge: {game_state['current_pose'].upper().replace('_', ' ')}")
        except cv.error:
            pass

    elif game_state['game_phase'] == 'waiting':
        cv.putText(image, "Get Ready!", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (28, 28, 28), 3)

    # Show score and round (except during startup)
    if game_state['game_phase'] != 'startup':
        cv.putText(image, f"Score: {game_state['score']}", (10, 280), cv.FONT_HERSHEY_SIMPLEX, 1, (54, 54, 54), 2)
        cv.putText(image, f"Round: {game_state['round']}", (10, 320), cv.FONT_HERSHEY_SIMPLEX, 1, (54, 54, 54), 2)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Resize the frame
    frame = cv.resize(frame, (1920, 1080))

    # Frame skipping logic - only process MediaPipe on certain frames
    frame_count += 1

    if frame_count % skip_frames == 0:
        # Convert color format to the default BGR to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Making predictions using pose, hand, and face models
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        pose_results = pose_model.process(image)
        hand_results = hands_model.process(image)
        face_results = face_model.process(image)
        image.flags.writeable = True

        # Store results for reuse (combine all results)
        results = {
            'pose': pose_results,
            'hands': hand_results,
            'face': face_results
        }
        last_results = results

        # Converting back the RGB image to BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        # Reuse last results and just convert frame to display format
        image = frame.copy()
        results = last_results

    # Only process landmarks if we have results (no visual output for performance)
    if results is not None and results['pose'] is not None and results['pose'].pose_landmarks:
        # Skip drawing landmarks to save FPS - just detect poses
        # Drawing Pose landmarks - COMMENTED OUT FOR PERFORMANCE
        # mp_drawing.draw_landmarks(
        #     image,
        #     results['pose'].pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     mp_drawing.DrawingSpec(
        #         color=(245,117,66),
        #         thickness=2,
        #         circle_radius=2
        #     ),
        #     mp_drawing.DrawingSpec(
        #         color=(245,66,230),
        #         thickness=2,
        #         circle_radius=2
        #     )
        # )

        # Detect all poses using pose, hand, and face data
        current_poses = detect_poses(results['pose'].pose_landmarks, results['hands'], results['face'])
        detected_poses.update(current_poses)

        # Game logic: Check challenge completion
        check_challenge_completion(current_poses)

        # Show detected poses (for debugging/feedback)
        y_offset = 40
        for pose_name, detected in current_poses.items():
            if detected:
                pose_display = pose_name.upper().replace('_', ' ')
                cv.putText(image, f"{pose_display} DETECTED!", (400, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y_offset += 30
    else:
        # No pose detected
        detected_poses = {'tpose': False, 'armcross': False, 'monkeyfinger': False, 'shocked_face': False, 'spiderman': False}
        # Game logic still needs to run even without pose detection
        check_challenge_completion(detected_poses)

    # Update game state
    update_game_state()

    # Draw game UI
    draw_game_ui(image)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv.putText(image, str(int(fps))+" FPS", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

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
pygame.mixer.quit()