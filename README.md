# Target Tracker using OpenCV and Kalman Filter
![Demo Video](https://github.com/maazjamshaid123/Target-Tracker-using-OpenCV-and-Kalman-Filter/blob/main/demo.mp4)
## Overview
This project implements an object tracker using OpenCV and a Kalman filter. The tracker allows a user to select a point in a video stream, then tracks the selected object using the MIL (MedianFlow, KCF, CSRT, GOTURN) tracker provided by OpenCV. Additionally, the Kalman filter is used to predict the state of the object being tracked, providing a more stable and accurate tracking experience.

## Architechture of the Project
![flowchart](https://github.com/maazjamshaid123/Target-Tracker-using-OpenCV-and-Kalman-Filter/assets/81762527/4e7d724c-0dd2-45af-8061-7c8c64be8baa)

## Features

- **Object Selection**: Click on the object in the video stream to initialize tracking.
- **Object Tracking**: Uses OpenCV's MIL tracker for real-time object tracking.
- **State Prediction**: Employs a Kalman filter for predicting the state of the tracked object.
- **Velocity Estimation**: Calculates and displays the velocity of the tracked object.

## Dependencies

- OpenCV
- NumPy

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the necessary libraries using pip:

```
pip install opencv-python-headless numpy
```

## Usage
**Download the project:**
```
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

## Select an object:
Click on the object you want to track in the video window. The tracker will initialize and start tracking the selected object.

## View tracking:
The tracked object will be highlighted with a bounding box, and its predicted position will be shown with a circle. The object's coordinates and velocity will also be displayed.

## Stop the tracker:
Press `q` to quit the application.

## Explanation
**Initialization**
The `Tracker` class initializes with the following key components:
**Selected Point:** Stores the point clicked by the user.
**Tracker:** Uses OpenCV's MIL tracker for tracking the object.
**Kalman Filter:** Configured for state prediction, helping in smoothing the tracking.
**Video Stream:** Reads from a video file or RTSP stream.

## Mouse Callback
The `mouse_callback` function handles the mouse click event to select the point to start tracking. It initializes the tracker and sets the initial state of the Kalman filter.

## Main Loop
The `run` method handles the main loop:

- Reads frames from the video stream.
- Updates the tracker and predicts the object's state.
- Draws the bounding box around the tracked object and displays the predicted state.
- Calculates and displays the velocity of the tracked object.
- Provides a way to quit the loop and release resources.

## Additional Details
- **Bounding Box Size:** The size of the bounding box around the tracked object can be adjusted with the `box_size` parameter.
- **Window Size:** The video display window dimensions are set to half of the 1080p resolution by default but can be adjusted.

## Troubleshooting
- **Video Stream Not Opening:** Ensure that the video file or RTSP stream URL is correct and accessible.
- **Dependencies Not Installed:** Verify that all required libraries are installed correctly.
- **Performance Issues:** For better performance, ensure your system meets the required specifications and try running on a lower resolution video.

## Contribution
Feel free to fork the repository and submit pull requests for improvements or bug fixes. For major changes, please open an issue first to discuss what you would like to change.
