# Military Object Recognition System

## Description

This project is a Python desktop application for **real-time object detection in video streams** using a trained YOLO model. The program provides a graphical user interface that allows users to select a video source, choose a detection model, filter detected object classes, and adjust detection confidence.

The application processes frames from a webcam, RTSP stream, or video file and detects objects such as military vehicles. Detected objects are highlighted with bounding boxes and labels in the video stream.

The system also logs detection results and performance metrics such as **FPS and inference time**.

---

## Features

* Real-time object detection using YOLO
* Graphical user interface built with Tkinter
* Support for multiple video sources:
  * USB webcam
  * RTSP stream
  * Local video file
* Model selection (Light / Heavy)
* Adjustable confidence threshold
* Object class filtering
* Bounding box visualization
* Overlapping detection merging
* Detection logging to file
* Real-time FPS monitoring
* Performance statistics generation

---

## Technologies Used

* Python
* Tkinter
* OpenCV
* Ultralytics YOLO
* Pillow (PIL)
* NumPy
* Multithreading

---

## Supported Object Classes

The system is designed to detect several types of military vehicles, for example:

* **AFV** — Armored Fighting Vehicle
* **APC** — Armored Personnel Carrier
* **MEV** — Military Engineering Vehicle
* **LAV** — Light Armored Vehicle

Each class is displayed using a unique color in the bounding box.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Master-s.git
cd military-recognition
```

### 2. Install dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install pillow
pip install numpy
```

---

## Download Model Weights

Due to GitHub file size limitations, the trained YOLO model weights are **not included in this repository**.

Download the models from Google Drive:

**Google Drive link:**

```
https://drive.google.com/your-model-link
```

After downloading, place the files in the **root directory of the project**.

Required files:

```
lightmodel.pt
heavymodel.pt
```

Your project folder should look like this:

```
project/
│
├── Main.py
├── lightmodel.pt
├── heavymodel.pt
└── README.md
```

---

## Running the Application

Start the program with:

```bash
python Main.py
```

The graphical interface will open automatically.

---

## Usage

1. Select the **video source**:

   * `0-USB` for webcam
   * RTSP stream address
   * local video file name

2. Select the **model type**:

   * **Light** – faster detection
   * **Heavy** – higher accuracy

3. Choose the **object class filter** or select **All**.

4. Adjust the **confidence threshold** using the slider.

5. Click **Start** to begin detection.

6. Detected objects will appear with bounding boxes and labels.

7. Click **Stop** to end the detection process.

---

## Output

### Real-time detection display

The GUI shows:

* Video stream
* Bounding boxes around detected objects
* Detection labels with confidence values
* FPS information
* Current processing time

### Detection log file

All detections are saved to:

```
detections.txt
```

Log format:

```
time_s    class    conf    bbox(x1,y1,x2,y2)
```

Example:

```
2.14   APC   0.87   120,300,250,410
3.15   AFV   0.91   400,280,520,430
```

If no objects are detected:

```
4.00   NO_OBJECTS   -   -
```

At the end of processing, performance statistics are appended:

```
=== SUMMARY ===
Average FPS: 22.5
Average inference time per frame, ms: 35.2
```

---

## Project Structure

```
project/
│
├── Main.py                # Main application
├── lightmodel.pt          # Lightweight detection model
├── heavymodel.pt          # High accuracy detection model
└── README.md
```

---

## Possible Improvements

Potential future improvements include:

* Exporting detection logs to CSV or JSON
* Saving processed video output
* Adding GPU/CPU runtime selection
* Supporting multiple cameras
* Real-time alerts for specific object types
* Detection statistics visualization

---

## License

This project is intended for educational and research purposes.
