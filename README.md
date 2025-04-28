Requirements.txt file contains all the requirements or libraries required for execution. The command for installing all the requirements from requirements.txt file is: pip install -r requirements.txt.
The main command for the execution of the project is: python main.py. A video is taken as input, the frames are extracted i.e. one frame at a time. The YOLOv5 model is used for detection of vehicles and license plates.
Bounding box is created over vehicles and license plates. Deep SORT alogorithm is used for tracking of vehicles. And the coordinates of the tracked vehicles are maintained. Speed of the vehicle is calculated from the
coordinates and the distance. Output video and the captured frames are saved.
The uploaded video is taken as input. The yolo model is loaded and the video is passed as arguments to  detect.py module. The detect.py file splits the whole video into frames and the yolo model for detecting the 
license plate is loaded. The frames are passed one by one for detection. After the detection of vehicles in the frame, the frames and the coordinates of the vehicle are passed to deep sort algorithm. The Deep SORT
algorithm tracks vehicles from one frame to another. The yolo model for license plate detection is loaded. The yolo model and the frames are passed to license_plate.py.
license_plate.py detects the license plate in the frame. The speed of the vehicle is calculated when vehicle crosses the two horizontal lines. The speed of the vehicle is calculated from the coordinates and distance.
Once the speed of the vehicle exceeds the maximum speed,  the frame is captured. The output video and the captured frame are saved
