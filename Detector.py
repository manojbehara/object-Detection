import cv2
import numpy as np
import time


np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Initialize the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()  # Load class labels and colors

    def readClasses(self):
        # Load class labels
        with open(self.classesPath, 'r') as f:
            self.classes = f.read().splitlines()

        self.classesList = self.classes
        self.classesList.insert(0, '__Background__')

        # Generate random colors for classes
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        success, image = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

        # Apply Non-Maximum Suppression
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) > 0:  # Check if there are indices returned
               for idx in np.ravel(bboxIdx):  # Handle different formats
                   bbox = bboxs[idx]
                   classConfidence = confidences[idx]
                   classLabelID = classLabelIDs[idx]
                   classLabel = self.classesList[classLabelID]
                   classColor = [int(c) for c in self.colorList[classLabelID]]

                   displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                   x, y, w, h = bbox

                # Draw bounding box
                   cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=2)

                # Add class label and confidence
                   label = f"{classLabel}: {classConfidence:.2f}"
                   cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)
                   ###############################

                   lineWidth = min(int(w * 0.3), int(h * 0.3))

                   cv2.line(image, (x,y), (x+lineWidth,y),classColor, thickness=5 )
                   cv2.line(image, (x,y), (x,y+lineWidth),classColor, thickness=5 )

                   cv2.line(image, (x+w,y), (x+w-lineWidth,y),classColor, thickness=5 )
                   cv2.line(image, (x+w,y), (x+w,y+lineWidth),classColor, thickness=5 )
                   ####################################

                   cv2.line(image, (x,y+h), (x+lineWidth,y+h),classColor, thickness=5 )
                   cv2.line(image, (x,y+h), (x,y+h-lineWidth),classColor, thickness=5 )

                   cv2.line(image, (x+w,y+h), (x+w-lineWidth,y+h),classColor, thickness=5 )
                   cv2.line(image, (x+w,y+h), (x+w,y+h-lineWidth),classColor, thickness=5 )

        # Show result
            cv2.putText(image, "FPS: " + str(int(fps)), (20,70),cv2.FONT_HERSHEY_PLAIN,2, (0,255,0),2)
            cv2.imshow("Result", image)

        # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
              break

            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()

