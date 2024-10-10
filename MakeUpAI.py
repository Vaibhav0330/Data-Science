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
