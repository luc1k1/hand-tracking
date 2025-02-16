# ✫ Hand Tracking & Finger Bend Detection 

A Python application that uses **MediaPipe** and **OpenCV** to detect hands, track fingers, and determine whether a finger is bent or straight.

##  Features
-  **Real-time hand tracking** with OpenCV & MediaPipe.
-  **Finger bend detection** using joint angles.
-  **Color selection panel** for interactive selection.
-  **Multi-hand support** (up to 2 hands).
-  **Enable/disable tracking with a key press (`T`)**.

##  How It Works
- Detects **hand landmarks** using MediaPipe.
- Calculates **joint angles** to check if fingers are bent.
- Displays **tracking status and bent fingers** in real-time.
- Includes a **color selection UI**, allowing interaction.

## 🛠️ Installation
Make sure you have **Python 3.7+** installed.

### Install Dependencies# Hand Tracking & Drawing with OpenCV & MediaPipe

An interactive real-time hand tracking and drawing application powered by **OpenCV** & **MediaPipe**.

---

## Features
- **Real-Time Hand Tracking** – Detects hands and landmarks instantly.  
- **Draw with Your Finger** – Use your index finger to draw on the screen.  
- **Color Selection** – Choose from multiple colors by hovering over the color bar.  
- **Finger Bend Detection** – Recognizes bent fingers and displays alerts.  
- **Keyboard Controls** – Switch modes and clear the canvas with hotkeys.  

---

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install opencv-python mediapipe numpy
```

---

## Usage
Run the script to start the application:

```sh
python hand_tracking.py
```

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `T` | Toggle hand tracking |
| `D` | Toggle drawing mode |
| `C` | Clear the canvas |
| `Q` | Exit the application |

---

## How It Works
1. **Hand Tracking:** Uses MediaPipe Hands to detect hand landmarks in real time.  
2. **Drawing Mode:** The index finger tip acts as a brush for drawing.  
3. **Color Selection:** Hover over the color palette at the top to change colors.  
4. **Finger Bending Detection:** Calculates angles to identify bent fingers.


---

## License
This project is open-source and free for educational and personal use.

Developed using OpenCV & MediaPipe.


```sh
pip install opencv-python mediapipe numpy
```

### ▶️ Run the Application
```sh
python hand_tracking.py
```

## 🎮 Controls
| Key | Action |
|----|--------|
| `T` | Toggle tracking on/off |
| `Q` | Quit the application |

##  Future Enhancements
-  Improve accuracy of finger bending detection.
-  Add gesture-based **color selection & drawing**.
-  Implement **saving & loading gestures**.

##  License
This project is licensed under the **MIT License**.

---

💡 **Like this project?** Feel free to ⭐ **star** the repo!

