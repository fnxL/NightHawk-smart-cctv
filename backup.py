# Python program to implement
# Webcam Motion Detector
import cv2, time
# importing datetime class from datetime library
import datetime
import numpy as np
import os

 
# Assigning our static_back to None
static_back = None
 
 
# Time of movement
time = []
 
print("[INFO] warming up...")
avg = None
lastUploaded = datetime.datetime.now()
motion_counter = 0
non_motion_timer = 200
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # a little hacky, but works for now
writer = None
(h, w) = (None, None)
zeros = None
output = None
made_recording = False

# Capturing video
video = cv2.VideoCapture(0)
 
# Infinite while loop to treat stack of image as video
while True:
    # Reading frame(image) from video
    grabbed, frame = video.read()
    timestamp = datetime.datetime.now()
    # Initializing motion = 0(no motion)
    motion_detected = False
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print("[INFO] Frame couldn't be grabbed. Breaking - " +
              datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
        break
    
    # resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # In first iteration we assign the value
    # of static_back to our first frame
    if static_back is None:
        static_back = gray
        continue
 
    # Difference between static background
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(static_back, gray)
 
    # If change in between static background and
    # current frame is greater than 30 it will show white color(21000055)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
 
    # Finding contour of moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(),
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 

    # loop over contours
    for contour in cnts:
        # if the contour is too small, ignore it.
        if cv2.contourArea(contour) < 10000:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    

    fps = int(round(video.get(cv2.CAP_PROP_FPS)))
    record_fps = 24
    ts = timestamp.strftime("%Y-%m-%d_%H_%M_%S")
    time_and_fps = ts + " - fps: " + str(fps)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Motion Detected: {}".format(motion_detected), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, time_and_fps, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
 
    


    def record_video():
        output = np.zeros((h2, w2, 3), dtype="uint8")
        output[0:h2, 0:w2] = frame
        # write the output frame to file
        writer.write(output)
        # print("[DEBUG] Recording....")


    if motion_detected:

        motion_counter += 1
        # check to see if the number of frames with motion is high enough
 
        if motion_counter >= 12:
            if writer is None:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                file_path = ('/home/fnx/Videos/' + "{filename}.mp4")
                file_path = file_path.format(filename=filename)

                (h2, w2) = frame.shape[:2]
                writer = cv2.VideoWriter(file_path, fourcc, record_fps, (w2, h2), True)
                zeros = np.zeros((h2, w2), dtype="uint8")
                
            record_video()
            made_recording = True
            non_motion_timer = 200
    
    # If there is no motion, continue recording until timer reaches 0
    # Else clean everything up
    else:
        if made_recording is True and non_motion_timer > 0:
            non_motion_timer -= 1
            print("[DEBUG] first else and timer: " + str(non_motion_timer))
            record_video()
        else:
            print("[DEBUG] hit else")
            motion_counter = 0
            if writer is not None:
                print("[DEBUG] hit if 1")
                writer.release()
                writer = None

            made_recording = False
            non_motion_timer = 200
     
    # Displaying image in gray_scale
    cv2.imshow("Gray Frame", gray)
 
    # Displaying the difference in currentframe to
    # the staticframe(very first_frame)
    cv2.imshow("Difference Frame", diff_frame)
 
    # Displaying the black and white image in which if
    # intensity difference greater than 30 it will appear white
    cv2.imshow("Threshold Frame", thresh_frame)
 
    # Displaying color frame with contour of motion of object
    cv2.imshow("Security Feed", frame)
 
    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        break
 
 

# cleanup the camera and close any open windows
print("[INFO] cleaning up...")

video.release()
 
# Destroying all the windows
cv2.destroyAllWindows()
