import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Precise indices for eyes, lips, forehead, and cheeks (based on MediaPipe landmark data)
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388]
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 310, 311, 312, 13, 82, 81, 80, 191]
LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FOREHEAD_INDICES = [151, 68, 54, 103, 67, 109, 10, 338, 297, 332, 284]
LEFT_CHEEK_INDICES = [118,50,93,101,36,205,123,234]#[142,234, 93, 132]
RIGHT_CHEEK_INDICES = [454, 323, 361, 288, 397, 365, 379, 400, 378, 152]

# Function to display facial landmarks and collect indices for eyes, lips, forehead, and cheeks
def display_facial_landmarks(image, landmarks):
    eyes_indices = set()
    lips_indices = set()
    forehead_indices = set()
    left_cheek_indices = set()
    right_cheek_indices = set()

    # Loop over the facial landmarks
    for i, landmark in enumerate(landmarks.landmark):
        # Convert normalized landmark coordinates to pixel values
        x = int(landmark.x * image.shape[1])  # Multiply by image width
        y = int(landmark.y * image.shape[0])  # Multiply by image height

        # Draw circles on the image for each landmark
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Check for eyes, lips, forehead, and cheeks based on known indices
        if i in LEFT_EYE_INDICES or i in RIGHT_EYE_INDICES:
            eyes_indices.add(i)
            cv2.putText(image, "Eye", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        elif i in UPPER_LIP_INDICES or i in LOWER_LIP_INDICES:
            lips_indices.add(i)
            cv2.putText(image, "Lip", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        elif i in FOREHEAD_INDICES:
            forehead_indices.add(i)
            cv2.putText(image, "Forehead", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        elif i in LEFT_CHEEK_INDICES:
            left_cheek_indices.add(i)
            cv2.putText(image, "Left Cheek", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 105, 180), 1)
        elif i in RIGHT_CHEEK_INDICES:
            right_cheek_indices.add(i)
            cv2.putText(image, "Right Cheek", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 105, 180), 1)

    # Print the indices for eyes, lips, forehead, and cheeks
    print("Eye Indices: ", sorted(list(eyes_indices)))
    print("Lip Indices: ", sorted(list(lips_indices)))
    print("Forehead Indices: ", sorted(list(forehead_indices)))
    print("Left Cheek Indices: ", sorted(list(left_cheek_indices)))
    print("Right Cheek Indices: ", sorted(list(right_cheek_indices)))

# Function to process the webcam feed and display landmarks in real time
def process_webcam_feed():
    cap = cv2.VideoCapture(0)  # Use the default webcam (0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Display the landmarks and their indexes on the frame
                display_facial_landmarks(frame, face_landmarks)

        # Show the frame with landmarks and indexes
        cv2.imshow('Live Face Landmark Indices (Eyes, Lips, Forehead, Cheeks)', frame)

        # Press 'q' to quit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the webcam feed and display face landmarks with indexes
process_webcam_feed()
