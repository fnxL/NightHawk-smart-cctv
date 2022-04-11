# Python program to implement
# Webcam Motion Detector
import cv2, time
# importing datetime class from datetime library
import datetime
import numpy as np
import os
 

class SmartCCTV():
    def __init__(self,record_fps=30):
        self.writer = None #CV2 VideoWriter
        self.static_background = None
        self.motion_counter = 0
        self.non_motion_timer = 200
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.height = None
        self.width = None
        self.isRecording = False
        self.output = None
        self.motion_detected = False
        self.record_fps = record_fps
        self.zeros = None
        self.filename = "opencv/haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(self.filename)
        # Capture feed from camera
        self.video = cv2.VideoCapture(0)

    
    def record_video(self, h2, w2, frame):
        output = np.zeros((h2, w2, 3), dtype="uint8")
        output[0:h2, 0:w2] = frame
        # write the output frame to file
        self.writer.write(output)
        # print("[DEBUG] Recording....")

  
    def getContours(self,gray):
        # Difference between static background
        # and current frame(Gaussian blur)
        diff_frame = cv2.absdiff(self.static_background, gray)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(21000055)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

        # Finding contour of moving object
        cnts,_ = cv2.findContours(thresh_frame.copy(),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return cnts


    def drawText(self,frame, time_and_fps = "NAN"):
        cv2.putText(frame, "Motion Detected: {}".format(self.motion_detected), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, time_and_fps, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    def run(self):
        # Prepare face regonizer
        paths = [os.path.join("persons", im) for im in os.listdir("persons")]
        labelslist = {}
        for path in paths:
            labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

        print(labelslist)
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read('model.yml')

        

        while True:
            # Reading frames(images) from video
            grabbed, frame = self.video.read()
            timestamp = datetime.datetime.now()
            self.motion_detected = False

            if not grabbed:
                print("[INFO] Frame couldn't be grabbed. Breaking - " +
                    datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
                break

            # convert frame to grayscale, and blur it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.cascade.detectMultiScale(gray, 1.3,2)

            processedFrame = cv2.GaussianBlur(gray, (21,21), 0)

            for x,y,w,h in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                roi = gray[y:y+h, x:x+w]

                label = recognizer.predict(roi)

                if label[1] < 100:
                    cv2.putText(frame, f"{labelslist[str(label[0])]} + {int(label[1])}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                else:
                    cv2.putText(frame, "unkown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)


            # Assign first frame as static background for comparision
            if self.static_background is None:
                self.static_background = processedFrame
                continue

            cnts = self.getContours(processedFrame)

            # loop over contours
            for contour in cnts:
                if cv2.contourArea(contour) < 10000:
                    continue
                self.motion_detected = True

                (x, y, w, h) = cv2.boundingRect(contour)
                # making green rectangle around the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            camera_fps = int(round(self.video.get(cv2.CAP_PROP_FPS)))
            ts = timestamp.strftime("%Y-%m-%d_%H_%M_%S")
            time_and_fps = ts + " - fps: " + str(camera_fps)
            
            self.drawText(frame,time_and_fps)

            if self.motion_detected:
                self.motion_counter += 1

                if self.motion_counter >= 12:
                    if self.writer is None:
                        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                        file_path = ('/home/fnx/Videos/' + "{filename}.mp4")
                        file_path = file_path.format(filename=filename)


                        (h2, w2) = frame.shape[:2]
                        self.writer = cv2.VideoWriter(file_path, self.fourcc, self.record_fps, (w2, h2), True)
                        self.zeros = np.zeros((h2, w2), dtype="uint8")

                    self.record_video(h2,w2,frame)
                    self.isRecording = True
                    self.non_motion_timer = 200

            else:
                if self.isRecording is True and self.non_motion_timer > 0:
                    self.non_motion_timer -= 1
                    self.record_video(h2,w2, frame)
                else:
                    self.motion_counter = 0
                    if self.writer is not None:
                        self.writer.release()
                        self.writer = None
                    
                    self.isRecording = False
                    self.non_motion_timer = 200

           
            cv2.imshow("Security Feed", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # cleanup the camera and close any open windows
        print("[INFO] cleaning up...")

        self.video.release()
 
        # Destroying all the windows
        cv2.destroyAllWindows()

