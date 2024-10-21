import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Define realistic color options for lipstick and blush in BGR format
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

blush_color_options = {
    'Soft Pink': (180, 105, 255),
    'Peach': (122, 160, 255),
    'Rose': (183, 110, 255),
    'Coral': (128, 128, 240)
}

# Default lipstick, blush color, and transparency
current_lipstick_color = color_options['Classic Red']
current_blush_color = blush_color_options['Soft Pink']
current_transparency = 0.7  # Default transparency level


# Function to apply lipstick to lips
def apply_lipstick(image, landmarks, lip_indices, color, transparency):
    lip_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                   int(landmarks.landmark[i].y * image.shape[0])) for i in lip_indices]

    if lip_points:
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(lip_points, np.int32)], color)
        cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)


# Function to apply blush on cheeks
def apply_blush(image, landmarks, cheek_indices, color, transparency):
    cheek_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                     int(landmarks.landmark[i].y * image.shape[0])) for i in cheek_indices]

    if cheek_points:
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(cheek_points, np.int32)], color)
        cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)


# Function to smooth only the cheek and forehead regions, excluding the eyes
def smooth_cheeks_and_forehead(image, landmarks):
    # Cheek and forehead landmark indices
    cheek_indices_left = [118,50,93,101,36,205,123,234]
    cheek_indices_right = [454, 323, 361, 288, 397, 365, 379, 400, 378, 152]

    # Forehead indices (covers the upper part of the face)
    forehead_indices = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58]

    # Left and right eye indices to exclude from smoothing
    left_eye_indices = [33, 133, 7, 163, 144]
    right_eye_indices = [362, 263, 249, 390, 373]

    # Create mask for cheeks and forehead
    mask = np.zeros_like(image)

    cheek_points_left = [(int(landmarks.landmark[i].x * image.shape[1]),
                          int(landmarks.landmark[i].y * image.shape[0])) for i in cheek_indices_left]
    cheek_points_right = [(int(landmarks.landmark[i].x * image.shape[1]),
                           int(landmarks.landmark[i].y * image.shape[0])) for i in cheek_indices_right]

    forehead_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                        int(landmarks.landmark[i].y * image.shape[0])) for i in forehead_indices]

    # Fill in the cheek and forehead regions on the mask
    cv2.fillPoly(mask, [np.array(cheek_points_left, np.int32)], (255, 255, 255))
    cv2.fillPoly(mask, [np.array(cheek_points_right, np.int32)], (255, 255, 255))
    cv2.fillPoly(mask, [np.array(forehead_points, np.int32)], (255, 255, 255))

    # Exclude the left and right eyes from the mask
    left_eye_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                        int(landmarks.landmark[i].y * image.shape[0])) for i in left_eye_indices]
    right_eye_points = [(int(landmarks.landmark[i].x * image.shape[1]),
                         int(landmarks.landmark[i].y * image.shape[0])) for i in right_eye_indices]

    # Set both eye regions to 0 (no smoothing)
    cv2.fillPoly(mask, [np.array(left_eye_points, np.int32)], (0, 0, 0))
    cv2.fillPoly(mask, [np.array(right_eye_points, np.int32)], (0, 0, 0))

    # Apply Gaussian blur to the cheek and forehead region only
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # Combine blurred image with original, applying the effect only to the cheeks and forehead
    image = np.where(mask == (255, 255, 255), blurred_image, image)

    return image


# Function to update lipstick color
def update_lipstick_color(event):
    global current_lipstick_color
    selected_color = lipstick_color_var.get()
    current_lipstick_color = color_options[selected_color]


# Function to update blush color
def update_blush_color(event):
    global current_blush_color
    selected_color = blush_color_var.get()
    current_blush_color = blush_color_options[selected_color]


# Function to update transparency
def update_transparency(value):
    global current_transparency
    current_transparency = float(value)


# Function to process the video feed
def video_loop():
    global current_lipstick_color, current_blush_color, current_transparency
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lip indices for applying lipstick
                upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 310, 311, 312, 13, 82, 81, 80, 191]
                lower_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87,
                                     178, 88, 95]

                # Apply lipstick to lips with current transparency level
                apply_lipstick(frame, face_landmarks, upper_lip_indices, current_lipstick_color, current_transparency)
                apply_lipstick(frame, face_landmarks, lower_lip_indices, current_lipstick_color, current_transparency)

                # Apply blush to cheeks
                cheek_indices_left = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152]
                cheek_indices_right = [454, 323, 361, 288, 397, 365, 379, 400, 378, 152]
                apply_blush(frame, face_landmarks, cheek_indices_left, current_blush_color, current_transparency)
                apply_blush(frame, face_landmarks, cheek_indices_right, current_blush_color, current_transparency)

                # Apply smoothing effect to the cheeks and forehead only, excluding both eyes
                frame = smooth_cheeks_and_forehead(frame, face_landmarks)

        # Convert the frame to a format that can be displayed in Tkinter
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_bgr)  # Convert to Image
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, video_loop)


# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create GUI window using Tkinter
root = Tk()
root.title("Makeup Effects Application")

# Create a label to display the video feed
video_label = Label(root)
video_label.pack()

# Lipstick Color Selector
Label(root, text="Select Lipstick Color:").pack()
lipstick_color_var = StringVar(value='Classic Red')
lipstick_selector = ttk.Combobox(root, textvariable=lipstick_color_var, values=list(color_options.keys()))
lipstick_selector.pack()
lipstick_selector.bind("<<ComboboxSelected>>", update_lipstick_color)

# Blush Color Selector
Label(root, text="Select Blush Color:").pack()
blush_color_var = StringVar(value='Soft Pink')
blush_selector = ttk.Combobox(root, textvariable=blush_color_var, values=list(blush_color_options.keys()))
blush_selector.pack()
blush_selector.bind("<<ComboboxSelected>>", update_blush_color)

# Transparency Slider
Label(root, text="Adjust Transparency:").pack()
transparency_slider = Scale(root, from_=0.1, to=1.0, resolution=0.1, orient=HORIZONTAL, command=update_transparency)
transparency_slider.set(current_transparency)
transparency_slider.pack()

# Start the video processing loop
video_loop()

# Run the Tkinter main loop
root.mainloop()

# Release resources after the GUI is closed
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
