import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Define realistic color options for lipstick in BGR format
color_options = {
    'Classic Red': (34, 34, 178),
    'Soft Pink': (180, 105, 255),
    'Nude': (63, 133, 205),
    'Coral': (128, 128, 240),
    'Berry': (139, 0, 139),
    'Wine': (0, 0, 128),
    'Mauve': (133, 21, 199),
    'Peach': (122, 160, 255)
}

# Updated indices for face landmarks
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 310, 311, 312, 13, 82, 81, 80, 191]
LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FOREHEAD_INDICES = [151, 68, 54, 103, 67, 109, 10, 338, 297, 332, 284]

# Function to apply lipstick to lips
def apply_lipstick(image, landmarks, lip_indices, color, transparency):
    lip_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                   int(landmarks.landmark[i].y * image.shape[0])) for i in lip_indices]

    if lip_points:
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(lip_points, np.int32)], color)
        cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

# Function to smooth only the cheek and forehead regions, excluding the eyes
def smooth_cheeks_and_forehead(image, landmarks):
    cheek_indices_left = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152]
    cheek_indices_right = [454, 323, 361, 288, 397, 365, 379, 400, 378, 152]

    cheek_points_left = [(int(landmarks.landmark[i].x * image.shape[1]),
                          int(landmarks.landmark[i].y * image.shape[0])) for i in cheek_indices_left]
    cheek_points_right = [(int(landmarks.landmark[i].x * image.shape[1]),
                           int(landmarks.landmark[i].y * image.shape[0])) for i in cheek_indices_right]
    forehead_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                        int(landmarks.landmark[i].y * image.shape[0])) for i in FOREHEAD_INDICES]

    # Create mask for cheeks and forehead
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.array(cheek_points_left, np.int32)], (255, 255, 255))
    cv2.fillPoly(mask, [np.array(cheek_points_right, np.int32)], (255, 255, 255))
    cv2.fillPoly(mask, [np.array(forehead_points, np.int32)], (255, 255, 255))

    # Exclude the eye regions
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388]
    left_eye_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                        int(landmarks.landmark[i].y * image.shape[0])) for i in LEFT_EYE_INDICES]
    right_eye_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                         int(landmarks.landmark[i].y * image.shape[0])) for i in RIGHT_EYE_INDICES]

    # Exclude the eye regions
    cv2.fillPoly(mask, [np.array(left_eye_points, np.int32)], (0, 0, 0))
    cv2.fillPoly(mask, [np.array(right_eye_points, np.int32)], (0, 0, 0))

    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    image = np.where(mask == (255, 255, 255), blurred_image, image)

    return image

# Function to process the image and apply effects
def process_image(image, lipstick_color, transparency):
    # Convert the image to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Apply lipstick
            apply_lipstick(image, face_landmarks, UPPER_LIP_INDICES, lipstick_color, transparency)
            apply_lipstick(image, face_landmarks, LOWER_LIP_INDICES, lipstick_color, transparency)

            # Smooth cheeks and forehead
            image = smooth_cheeks_and_forehead(image, face_landmarks)

    return image

# Streamlit UI
#st.title("Makeup Effects Application")

# Sidebar for options
st.sidebar.header("Makeup Effects Application")
mode = st.sidebar.radio("Choose Input Mode:", ("Upload Image", "Webcam"))

# Lipstick Color Selector
lipstick_color_name = st.sidebar.selectbox("Select Lipstick Color:", list(color_options.keys()))
current_lipstick_color = color_options[lipstick_color_name]

# Transparency Slider
current_transparency = st.sidebar.slider("Adjust Transparency:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

if mode == "Upload Image":
    # Upload image section
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Convert to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Process the image
        processed_image = process_image(image, current_lipstick_color, current_transparency)

        # Convert back to Image for display
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image, caption='Processed Image', use_column_width=True)

elif mode == "Webcam":
    # Webcam section
    st.header("Webcam Feed")

    # Create a placeholder for the webcam output
    placeholder = st.empty()

    # State to control webcam
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    # Start and Stop Webcam buttons on the main page
    if st.button("Start Webcam", key="start_webcam"):
        st.session_state.webcam_running = True

    if st.button("Stop Webcam", key="stop_webcam"):
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Could not read from webcam.")
                    break

                # Process the frame
                processed_frame = process_image(frame, current_lipstick_color, current_transparency)

                # Display the processed frame in Streamlit
                placeholder.image(processed_frame, channels="BGR", use_column_width=True)




            cap.release()
=======
import streamlit as st
import dlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Streamlit app title
st.title("AI Makeup Application")
st.write("Upload an image and choose a makeup style")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Makeup options
option_makeup = st.selectbox("Choose your makeup style:", ("Deep Gray", "Brown", "Hot Pink", "Crimson"))

# Apply makeup if an image is uploaded
if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    
    # Detect faces in the image
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            
            # Convert the image to RGBA
            img_with_makeup = img.convert("RGBA")
            overlay = Image.new('RGBA', img_with_makeup.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Define makeup colors based on the selected option
            if option_makeup == "Deep Gray":
                eyebrow_color = (68, 54, 39, 90)
                lip_color = (150, 0, 0, 128)
                blush_color = (255, 0, 0, 100)
            elif option_makeup == "Brown":
                eyebrow_color = (110, 38, 14, 70)
                lip_color = (199, 21, 133, 128)
                blush_color = (210, 180, 140, 100)
            elif option_makeup == "Hot Pink":
                eyebrow_color = (68, 54, 39, 90)
                lip_color = (255, 105, 180, 60)
                blush_color = (255, 20, 147, 100)
            elif option_makeup == "Crimson":
                eyebrow_color = (68, 54, 39, 90)
                lip_color = (220, 20, 1, 60)
                blush_color = (255, 99, 71, 100)

            # Get landmark points for different facial regions
            landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            # Define landmark indices for key face parts
            left_eyebrow_indices = [17, 18, 19, 20, 21]
            right_eyebrow_indices = [22, 23, 24, 25, 26]
            top_lip_indices = [48, 49, 50, 51, 52, 53, 54]
            bottom_lip_indices = [54, 55, 56, 57, 58, 59, 48]
            cheek_left_indices = [36, 1, 2, 3, 31]
            cheek_right_indices = [45, 15, 14, 13, 35]

            # Draw eyebrows
            left_eyebrow_points = [landmark_points[i] for i in left_eyebrow_indices]
            right_eyebrow_points = [landmark_points[i] for i in right_eyebrow_indices]
            draw.polygon(left_eyebrow_points, fill=eyebrow_color)
            draw.polygon(right_eyebrow_points, fill=eyebrow_color)

            # Draw lips
            top_lip_points = [landmark_points[i] for i in top_lip_indices]
            bottom_lip_points = [landmark_points[i] for i in bottom_lip_indices]
            draw.polygon(top_lip_points, fill=lip_color)
            draw.polygon(bottom_lip_points, fill=lip_color)

            # Draw blush
            left_cheek_points = [landmark_points[i] for i in cheek_left_indices]
            right_cheek_points = [landmark_points[i] for i in cheek_right_indices]
            draw.polygon(left_cheek_points, fill=blush_color)
            draw.polygon(right_cheek_points, fill=blush_color)

            # Overlay the makeup onto the original image
            img_with_makeup = Image.alpha_composite(img_with_makeup, overlay)

            # Display the result
            st.image(img_with_makeup, caption='Image with Makeup', use_column_width=True)
    else:
        st.write("No face detected. Please upload a clear image with a visible face.")

