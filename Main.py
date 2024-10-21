import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

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

# Default lipstick color and transparency
current_lipstick_color = color_options['Classic Red']
current_transparency = 0.7  # Default transparency level

# Updated indices
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388]
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

# Function to update lipstick color
def update_lipstick_color(event):
    global current_lipstick_color
    selected_color = lipstick_color_var.get()
    current_lipstick_color = color_options[selected_color]

# Function to update transparency
def update_transparency(value):
    global current_transparency
    current_transparency = float(value)

# Function to process the video feed
def video_loop():
    global current_lipstick_color, current_transparency
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Apply lipstick
                apply_lipstick(frame, face_landmarks, UPPER_LIP_INDICES, current_lipstick_color, current_transparency)
                apply_lipstick(frame, face_landmarks, LOWER_LIP_INDICES, current_lipstick_color, current_transparency)

                # Smooth cheeks and forehead
                frame = smooth_cheeks_and_forehead(frame, face_landmarks)

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
