import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Precise indices for cheeks
LEFT_CHEEK_INDICES = [118, 50, 93, 101, 36, 205, 123, 234]

# Function to display facial landmarks
def display_facial_landmarks(image, landmarks):
    # Loop over the facial landmarks
    for i, landmark in enumerate(landmarks.landmark):
        # Convert normalized landmark coordinates to pixel values
        x = int(landmark.x * image.shape[1])  # Multiply by image width
        y = int(landmark.y * image.shape[0])  # Multiply by image height

        # Draw circles on the image for each landmark
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Label the landmarks
        if i in LEFT_CHEEK_INDICES:
            cv2.putText(image, "Cheek", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

# Function to apply blush effect on the cheeks
def apply_blush(image, landmarks, blush_color, transparency):
    # Calculate the points for the left cheek
    cheek_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                      int(landmarks.landmark[i].y * image.shape[0])) for i in LEFT_CHEEK_INDICES]

    # Create an overlay with the blush color
    overlay = image.copy()
    cv2.fillPoly(overlay, [np.array(cheek_points, np.int32)], blush_color)

    # Apply Gaussian blur to the overlay to soften the edges
    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

# Function to process the webcam feed and display landmarks in real-time
def process_webcam_feed():
    cap = cv2.VideoCapture(0)  # Use the default webcam (0)

    # Blush color (light blush)
    blush_color = (251, 234, 239)  # Define as a tuple
    light_blush_color = tuple((np.array(blush_color) + np.array([255, 255, 255])) // 2)  # Mix with white

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
                # Display the landmarks
                display_facial_landmarks(frame, face_landmarks)

                # Apply the blush effect
                apply_blush(frame, face_landmarks, light_blush_color, 0.5)  # Set transparency as needed

        # Show the frame with landmarks and blush effect
        cv2.imshow('Live Face Landmark with Blush Effect', frame)

        # Press 'q' to quit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the webcam feed and display face landmarks with blush effect
process_webcam_feed()
