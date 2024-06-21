import cv2
import numpy as np

class Tracker:
    def __init__(self):
        '''
        Initialize the Tracker class, setting up necessary variables and configurations.

        selected_point: Tuple[int, int] or None
            Stores the selected point from mouse click.

        tracker: cv2.Tracker or None
            Initialized tracker object for object tracking.

        kalman_filter: cv2.KalmanFilter
            Kalman filter for state estimation.

        predicted_state: np.ndarray or None
            Predicted state of the object being tracked.

        video_stream: cv2.VideoCapture
            Video capture object for accessing the camera.

        rtsp_stream: str
            RTSP stream URL for video streaming.

        box_size: int
            Size of the bounding box around the tracked object.

        window_width: int
            Width of the video display window.
        '''
        self.selected_point = None
        self.tracker = None
        self.kalman_filter = cv2.KalmanFilter(6, 2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman_filter.processNoiseCov = 0.1 * np.eye(6, dtype=np.float32)
        self.kalman_filter.measurementNoiseCov = 1 * np.eye(2, dtype=np.float32)
        self.predicted_state = None
        # self.video_stream = cv2.VideoCapture(0)
        self.video_stream = cv2.VideoCapture("demo.mp4")
        self.rtsp_stream = 'rtsp://....'
        self.tracker = cv2.TrackerMIL_create()
        self.box_size = 10
        self.window_width = 1920 // 2
        self.window_height = 1080 // 2
        
    def mouse_callback(self, event, x, y, flags, param):
        '''
        The mouse_callback function is a callback function that is called whenever a mouse event occurs on the video frame. 
        It is used in this context to handle the selection of a point by the user for object tracking.

        event: int
            The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN for left button down).

        x: int
            The x-coordinate of the mouse event.

        y: int
            The y-coordinate of the mouse event.

        flags: int
            Additional flags from cv2.

        param: Any
            Additional parameters.

        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)
            ok, frame = self.video_stream.read()
            if ok:
                bbox = (x - self.box_size, y - self.box_size, 2 * self.box_size, 2 * self.box_size)
                self.tracker.init(frame, bbox)
                self.kalman_filter.statePost = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
                self.predicted_state = self.kalman_filter.statePost[:4]
                
    def run(self):
        '''
        Main loop for the object tracker. Reads frames from the video stream, processes them for object tracking,
        and displays the result in a window. Allows the user to select an object for tracking and visualize the
        tracking process in real-time.
        '''

        if not self.video_stream.isOpened():
            print("Error opening video stream")
            return
        
        while True:
            ret, frame = self.video_stream.read()
            if ret:
                if self.selected_point is not None:
                    ok, bbox = self.tracker.update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        center_x = int(x + w / 2)
                        center_y = int(y + h / 2)
                        self.predicted_state = self.kalman_filter.predict()[:4]
                        predicted_x, predicted_y = self.predicted_state[0], self.predicted_state[1]
                        cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (0, 255, 0), -1)
                        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
                        self.kalman_filter.correct(measurement)
                        cv2.putText(frame, f'Target: ({center_x}, {center_y})', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        predicted_vx, predicted_vy = self.predicted_state[2], self.predicted_state[3]
                        pixels_per_meter = 100
                        velocity_mps = (predicted_vx / pixels_per_meter, predicted_vy / pixels_per_meter)
                        velocity_text = f'Velocity: ({velocity_mps[0][0]:.2f} m/s, {velocity_mps[1][0]:.2f} m/s)'
                        text_x = 10
                        text_y = frame.shape[0] - 10
                        cv2.putText(frame, velocity_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                frame_height, frame_width, _ = frame.shape
                center_x = int(frame_width / 2)
                center_y = int(frame_height / 2)
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 0), -1)
                cv2.putText(frame, f'({center_x}, {center_y})', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Frame', self.window_width, self.window_height)
                cv2.namedWindow('Frame')
                cv2.setMouseCallback('Frame', self.mouse_callback)
                cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()
