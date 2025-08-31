import cv2
import numpy as np
import time
import mediapipe as mp
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Function to capture the background
def create_background(cap, num_frames=30):
    print("Capturing background. Please move out of frame.")
    backgrounds = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            backgrounds.append(frame)
        else:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
        time.sleep(0.1)
    if backgrounds:
        return np.median(backgrounds, axis=0).astype(np.uint8)
    else:
        raise ValueError("Could not capture any frames for background")

# Function to create a mask based on color range
def create_mask(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return mask

# Function to apply the cloak effect
def apply_cloak_effect(frame, mask, background):
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)
    return cv2.add(fg, bg)

# Function to apply real-time filters and effects
def apply_filter(frame, filter_type):
    if filter_type == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif filter_type == "pixelate":
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif filter_type == "grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == "sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_filter)
    else:
        return frame

# Main application class
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Create a canvas to display the video
        self.canvas = Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Initialize background
        try:
            self.background = create_background(self.cap)
        except ValueError as e:
            print(f"Error: {e}")
            self.cap.release()
            return

        # Load the predefined background image
        self.predefined_background = cv2.imread("Background1.jpg")  # Replace with your custom background image path
        if self.predefined_background is None:
            print("Warning: Could not load predefined background image. Using default background.")
            self.predefined_background = np.zeros((480, 640, 3), dtype=np.uint8)  # Default black background
        else:
            print("Predefined background image loaded successfully.")

        # Initialize custom background
        self.custom_background = self.predefined_background  # Start with the predefined background

        # Define color ranges for masking
        self.color_ranges = {
            "Red": (np.array([0, 120, 70]), np.array([10, 255, 255])),  # Red
            "Blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),  # Blue
            "Green": (np.array([35, 50, 50]), np.array([85, 255, 255])),  # Green
            "Pink": (np.array([140, 50, 50]), np.array([170, 255, 255])),  # Pink
            "Yellow": (np.array([15, 50, 50]), np.array([35, 255, 255]))  # Yellow
        }

        # Initialize filter type
        self.filter_type = None

        # Initialize mode
        self.mode = "normal"  # Options: "normal", "invisibility", "filters", "custom_background", "user_background"

        # Initialize color for invisibility
        self.invisibility_color = "Blue"  # Default to blue

        # Create buttons for mode selection
        self.btn_normal = Button(window, text="Normal Mode", command=self.set_normal_mode, bg="gray",activebackground='brown')
        self.btn_normal.pack(side=LEFT, padx=10, pady=10)

        self.btn_invisibility = Button(window, text="Invisibility Mode", command=self.set_invisibility_mode,bg="gray",activebackground='brown')
        self.btn_invisibility.pack(side=LEFT, padx=10, pady=10)

        # Create a button to change invisibility color
        self.btn_change_color = Button(window, text="Change Invisibility Color", command=self.change_invisibility_color,bg="gray",activebackground='cyan')
        self.btn_change_color.pack(side=LEFT, padx=10, pady=10)

        self.btn_filters = Button(window, text="Filters Mode", command=self.set_filters_mode, bg="gray",activebackground='brown')
        self.btn_filters.pack(side=LEFT, padx=10, pady=10)

        # Create a button to cycle through filters
        self.btn_cycle_filter = Button(window, text="Cycle Filter", command=self.cycle_filter, bg="gray",activebackground='cyan')
        self.btn_cycle_filter.pack(side=LEFT, padx=10, pady=10)

        self.btn_custom_bg = Button(window, text="Custom Background Mode", command=self.set_custom_bg_mode,bg="gray",activebackground='brown')
        self.btn_custom_bg.pack(side=LEFT, padx=10, pady=10)

        self.btn_user_bg = Button(window, text="User-Defined Background", command=self.set_user_bg_mode,bg="gray",activebackground='brown')
        self.btn_user_bg.pack(side=LEFT, padx=10, pady=10)

        # Start the video update loop
        self.update()
        self.window.mainloop()

    def set_normal_mode(self):
        self.mode = "normal"
        print("Mode: Normal")

    def set_invisibility_mode(self):
        self.mode = "invisibility"
        print("Mode: Invisibility")

    def set_filters_mode(self):
        self.mode = "filters"
        print("Mode: Filters")

    def set_custom_bg_mode(self):
        self.mode = "custom_background"
        self.custom_background = self.predefined_background  # Use the predefined background
        print("Mode: Custom Background")

    def set_user_bg_mode(self):
        self.mode = "user_background"
        print("Mode: User-Defined Background")
        # Open a file dialog to select a custom background image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.custom_background = cv2.imread(file_path)
            if self.custom_background is None:
                print("Error: Could not load the image. Please check the path and try again.")
                self.mode = "normal"  # Switch back to normal mode
            else:
                print("Custom background image loaded successfully.")

    def cycle_filter(self):
        if self.mode == "filters":
            filters = [None, "blur", "pixelate", "grayscale", "sepia"]
            current_index = filters.index(self.filter_type) if self.filter_type in filters else 0
            self.filter_type = filters[(current_index + 1) % len(filters)]
            print(f"Filter applied: {self.filter_type}")

    def change_invisibility_color(self):
        if self.mode == "invisibility":
            colors = ["Red", "Blue", "Green", "Pink", "Yellow"]
            current_index = colors.index(self.invisibility_color)
            self.invisibility_color = colors[(current_index + 1) % len(colors)]
            print(f"Invisibility Color: {self.invisibility_color}")

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize custom background to match frame dimensions
            if self.custom_background is not None and not np.array_equal(self.custom_background, frame):
                self.custom_background = cv2.resize(self.custom_background, (frame.shape[1], frame.shape[0]))

            # Apply effects based on the selected mode
            if self.mode == "normal":
                result = frame  # Normal mode: display the frame as is
            elif self.mode == "invisibility":
                # Get the color range for the selected invisibility color
                lower_color, upper_color = self.color_ranges[self.invisibility_color]
                # Create mask and apply cloak effect
                mask = create_mask(frame, lower_color, upper_color)
                result = apply_cloak_effect(frame, mask, self.background)
            elif self.mode == "filters":
                # Apply selected filter
                result = apply_filter(frame, self.filter_type)
            elif self.mode == "custom_background" or self.mode == "user_background":
                # Use MediaPipe Selfie Segmentation to create a mask for the person
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = selfie_segmentation.process(rgb_frame)
                mask = results.segmentation_mask

                # Threshold the mask to create a binary mask
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Resize the mask to match the frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Invert the mask to get the background
                bg_mask = cv2.bitwise_not(mask)

                # Extract the person from the frame
                person = cv2.bitwise_and(frame, frame, mask=mask)

                # Extract the custom background
                bg = cv2.bitwise_and(self.custom_background, self.custom_background, mask=bg_mask)

                # Combine the person and custom background
                result = cv2.add(person, bg)

            # Convert the result to RGB format for displaying in Tkinter
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(result_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new image
            self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

        # Call the update function again after 10 milliseconds
        self.window.after(10, self.update)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Create a Tkinter window and pass it to the App class
root = Tk()
BASE_DIR = os.path.dirname(__file__)
icon_img = PhotoImage(file=os.path.join(BASE_DIR, "tech_srujan.PNG")) 
root.iconphoto(False, icon_img)
app = App(root, "CamCraze")

