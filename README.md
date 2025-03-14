# MOT-B34-2023
The Final Year project of Batch 34 at DSCE for the year 2022-23.

Link for the dataset used - https://github.com/VisDrone/VisDrone-Dataset

# Multi-Object Tracking with OCSORT and YOLO-NAS

This repository implements a multi-object tracking system using the OCSORT (Observation-Centric SORT) algorithm combined with YOLO-NAS (You Only Look Once - Neural Architecture Search) for object detection. The system is designed to track multiple objects in a video stream, providing real-time tracking with high accuracy.

## Features

- **Object Detection**: Utilizes YOLO-NAS for detecting objects in each frame of the video.
- **Object Tracking**: Implements the OCSORT algorithm for robust and efficient multi-object tracking.
- **Visualization**: Tracks are visualized with bounding boxes and unique IDs for each object.
- **GUI**: Includes a graphical user interface (GUI) for video playback and tracking visualization.
- **Performance Metrics**: Tracks the number of frames each object is detected (hits) and provides FPS (frames per second) metrics.

## Components

### 1. **Object Detection (`MainApp.py`)**
   - Uses YOLO-NAS (`yolo_nas_s` model) for detecting objects in each frame.
   - Processes the video frame-by-frame, resizing frames for efficient inference.
   - Outputs bounding boxes and confidence scores for detected objects.

### 2. **Object Tracking (`ocsort.py`)**
   - Implements the OCSORT algorithm for associating detections across frames.
   - Uses Kalman filters for predicting object trajectories.
   - Supports multiple association cost functions (IoU, GIoU, DIoU, CIoU).
   - Handles object re-identification and track management.

### 3. **Kalman Filter (`kalmanfilter.py`)**
   - Implements a Kalman filter for state estimation of tracked objects.
   - Predicts and updates object positions based on detections.

### 4. **Association Logic (`association.py`)**
   - Contains functions for calculating association costs between detections and tracks.
   - Supports IoU, GIoU, DIoU, and CIoU for bounding box matching.

### 5. **Graphical User Interface (`gui.py`)**
   - Provides a GUI for video playback, pause, forward, and backward controls.
   - Displays tracked objects with unique IDs and the number of frames they have been detected.
   - Saves the output video with tracking annotations.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Combust10/MOT-B34-2023.git
cd MOT-B34-2023
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download YOLO-NAS weights:
Ensure you have the YOLO-NAS model weights (yolo_nas_s_coco.pth) in the appropriate directory.

4. Run the application:
```bash
python MainApp.py
```

## Usage

1. **Video Input**:
   - Place your video file in the project directory and update the `video_path` variable in `MainApp.py` to point to your video.

2. **Run the Tracker**:
   - Execute `MainApp.py` to start the tracking process. The script will process the video, track objects, and save the output video with tracking annotations.

3. **GUI**:
   - After processing, the GUI will automatically launch, allowing you to view the tracked video with object IDs and frame counts.

## Configuration

- **Detection Threshold**: Adjust the detection threshold in `MainApp.py` to control the sensitivity of object detection.
- **Tracking Parameters**: Modify the OCSORT parameters in `MainApp.py` (e.g., `max_age`, `min_hits`, `iou_threshold`) to fine-tune tracking performance.
- **Association Function**: Change the association function in `ocsort.py` (e.g., `iou`, `giou`, `diou`, `ciou`) to experiment with different matching strategies.

## Output

- **Tracked Video**: The output video (`output_video.mp4`) will be saved in the project directory, showing the tracked objects with bounding boxes and IDs.
- **Track IDs and Hits**: The `track_ids.txt` and `track_hits.txt` files will contain the IDs of tracked objects and the number of frames they were detected, respectively.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- PyTorch
- Super-Gradients (for YOLO-NAS)
- MoviePy
- CustomTkinter (for GUI)
- TkVideoPlayer (for video playback in GUI)
