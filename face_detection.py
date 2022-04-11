import cv2
import os
import numpy as np


class FaceDetection():
    def __init__(self,name='', userId=-1):
        self.video = cv2.VideoCapture(0)
        self.filename = "opencv/haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(self.filename)
        self.image_count = 1
        self.name = name
        self.userId = userId
 
    def run(self):
        paths = [os.path.join("persons", im) for im in os.listdir("persons")]
        labelslist = {}
        for path in paths:
            labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

        print(labelslist)
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read('model.yml')

        while True:
            _, frame = self.video.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.cascade.detectMultiScale(gray, 1.3, 2)

            for x,y,w,h in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                roi = gray[y:y+h, x:x+w]

                label = recognizer.predict(roi)

                if label[1] < 100:
                    cv2.putText(frame, f"{labelslist[str(label[0])]} + {int(label[1])}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                else:
                    cv2.putText(frame, "unkown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            cv2.imshow("identify", frame)

            key = cv2.waitKey(1)

            if key == ord('q') :
                cv2.destroyAllWindows()
                self.video.release()
                break

    def collect_dataset(self):
        while True:
            _, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.4, 1)

            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2) 
                roi = gray[y:y+h, x:x+w]

                cv2.imwrite(f'persons/{self.name}-{self.image_count}-{self.userId}.jpg', roi)
                self.image_count += 1 
                cv2.putText(frame, f'{self.image_count}', (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),3)
                cv2.imshow("new", roi)

            cv2.imshow("identify", frame)

            if cv2.waitKey(1) == 27 or self.image_count > 100 or cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                self.video.release()
                self.train()
                break
    
    def train(self):
        print("Training Initiated!")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        dataset = 'persons/'
        paths = [os.path.join(dataset,images) for images in os.listdir(dataset)]

        faces = []
        ids = []

        labels = []

        for path in paths:
            labels.append(path.split('/')[-1].split('-')[0])
            ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))
            faces.append(cv2.imread(path, 0))

        recognizer.train(faces, np.array(ids))

        recognizer.save('model.yml')

        return
