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

### Install Dependencies
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

