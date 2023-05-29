from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time



app = Flask(__name__)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(0) 

#load custom model
path_model = r"C:\Users\nidam\Downloads\buat kodingan orbit\deployment\ujian paktek\deploy PA\app\model\model ANN.sav"
path_minmax = r"C:\Users\nidam\Downloads\buat kodingan orbit\deployment\ujian paktek\deploy PA\app\model\minmax.pkl"

def gen_frames() :
    # Initializing mediapipe pose class.
    mp_pose = mp.solutions.pose

    # Initializing mediapipe drawing class, useful for annotation.
    mp_drawing = mp.solutions.drawing_utils 

    # used to record the time when we processed last frame 
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0

    model = pickle.load(open(path_model, "rb"))
    sc = pickle.load(open(path_minmax, "rb"))

    # Initialize a list to store the detected landmarks.
    landmarks = []

    while True:
        success, frame = video.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            # Setup Pose function for video.
            with mp_pose.Pose(static_image_mode=False, 
            min_detection_confidence=0.5, 
            model_complexity=1) as pose_video:
                frame.flags.writeable = False
            
                # Convert the image from BGR into RGB format.
                imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Perform the Pose Detection.
                results = pose_video.process(imageRGB)
                
                # Retrieve the height and width of the input image.
                height, width, _ = frame.shape
                

                # Check if any landmarks are detected.
                if results.pose_landmarks:
                
                    # Draw Pose landmarks on the output image.
                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks,
                                            connections=mp_pose.POSE_CONNECTIONS)
                    
                    # # Iterate over the detected landmarks.
                    # for landmark in results.pose_landmarks.landmark:
                        
                    #     # Append the landmark into the list.
                    #     landmarks.append((int(landmark.x * width), int(landmark.y * height),
                    #                         (landmark.z * width)))


                try:
                    if results.pose_landmarks:

                        image_height, image_width, _ = frame.shape
                        landmarks = results.pose_landmarks.landmark
                        
                        # mencari landmark bagian tubuh kiri
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height]
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]

                        # mencari engle bagian tubuh kiri
                        angle_left_elbow = calculateAngle(left_wrist, left_elbow, left_shoulder)
                        angle_left_shoulder= calculateAngle(left_elbow, left_shoulder, left_hip)
                        angle_left_hip= calculateAngle(left_shoulder, left_hip, left_knee)
                        angle_left_knee = calculateAngle(left_hip, left_knee, left_ankle)

                        # mencari landmark bagian tubuh kanan
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height]
                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height]
                        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height]
                        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height]

                        # mencari engle bagian tubuh kiri
                        angle_right_elbow = calculateAngle(right_wrist, right_elbow, right_shoulder)
                        angle_right_shoulder = calculateAngle(right_elbow, right_shoulder, right_hip)
                        angle_right_hip = calculateAngle(right_shoulder, right_hip, right_knee)
                        angle_right_knee = calculateAngle(right_hip, right_knee, right_ankle)


                        # konsultasi 
                        data = []
                        data.append([angle_left_elbow, angle_left_shoulder, angle_left_hip, angle_left_knee, 
                            angle_right_elbow, angle_right_shoulder, angle_right_hip, angle_right_knee])

                        # if len(data)>=8:
                        #     print(data)
                        x = sc.transform(data)
                        y_result = model.predict(x)
                        prediction_result = y_result[0]
                        cv2.putText(frame,f'{prediction_result}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)


                except Exception as e:
                    continue

                finally:
                    # font which we will be using to display FPS
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # time when we finish processing for this frame
                    new_frame_time = time.time()

                    fps = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time

                    # converting the fps into integer
                    fps = int(round(fps))

                    # converting the fps to string so that we can display it on frame
                    # by using putText function
                    fps = str(fps)
                    # puting the FPS count on the frame
                    cv2.putText(frame, fps, (550, 50), font, 2, (100, 255, 0), 3, cv2.LINE_AA)

                    # Show the result
                    cv2.imshow('Result', frame)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def calculateAngle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('index.html')

# @app.route('/treePose')
# def treePose():
#     """Video streaming home page."""
#     return render_template('index2.html')

# @app.route('/plankPose')
# def plankPose():
#     """Video streaming home page."""
#     return render_template('index3.html')

# @app.route('/warrior2Pose')
# def warrior2Pose():
#     """Video streaming home page."""
#     return render_template('index4.html')

# @app.route('/dandasanaPose')
# def dandasanaPose():
#     """Video streaming home page."""
#     return render_template('index5.html')

# @app.route('/goddesPose')
# def goddesPose():
#     """Video streaming home page."""
#     return render_template('index6.html')


# if __name__ == '__main__':
#     app.run(debug=True)