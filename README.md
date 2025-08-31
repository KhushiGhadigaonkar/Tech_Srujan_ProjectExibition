**CamCraze 🎥✨**

CamCraze is a fun and interactive real-time camera effects application built using Python, OpenCV, Tkinter, and MediaPipe.
It allows users to experience invisibility cloaks, live filters, and custom background replacement in real time.

**Features**
* Normal Mode → Displays your live webcam feed without any changes.
* Invisibility Cloak Mode → Wear a colored cloth (Red, Blue, Green, Pink, Yellow), and the app makes it invisible using color masking.
* Filters Mode → Apply fun filters to your video feed:
 --> Blur
 --> Pixelate
 --> Grayscale
 --> Sepia
* Custom Background Mode → Replace your background with a predefined image (Background1.jpg).
* User-Defined Background Mode → Select your own background image from your system.
* MediaPipe Segmentation → Accurately detects people for background replacement.
* Tkinter GUI → Simple buttons to switch between modes and filters.

**Tech Stack**
* Python 3.8+
* OpenCV (cv2) → For video processing and image effects
* MediaPipe → For person segmentation
* Tkinter → GUI for mode switching
* PIL (Pillow) → For rendering frames in Tkinter
