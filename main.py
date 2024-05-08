import cv2
import numpy as np
from datetime import datetime, timedelta

labeled_images_path = 'C:\Users\Adish Gujarathi\OneDrive\Desktop\Face Detection Attendance Tracking\images'

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# initializes the face cascade classifier using the XML file containing the pre-trained model for detecting frontal faces.
recognizer = cv2.face_LBPHFaceRecognizer.create()
#local binary pattern histograms

# Prepare training data
def prepare_training_data():
    labels = []
    faces = []
    label_dict = {}
    label = 0

    for name in ['Adish' , 'Rushabh']:
        image_path = f'{labeled_images_path}/{name}.jpg'  
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            faces.append(np.array(image))
            #converts the image to a numpy array and appends it to the faces list.
            labels.append(label)
            #appends the current label (label) to the labels list
            label_dict[label] = name
            label += 1
        else:
            print(f"Image {name} not loaded properly.")

    return faces, labels, label_dict

faces, labels, label_dict = prepare_training_data()
recognizer.train(faces, np.array(labels))

# Ask for lecture name and duration
lecture_name = input("Enter lecture name: ")
lecture_duration = int(input("Enter lecture duration in minutes: "))
attendance_record_path = f'{lecture_name}_attendance.txt'

# Initialize camera
cap = cv2.VideoCapture(0)

attended_students = set()
start_time = datetime.now()
end_time = start_time + timedelta(minutes=lecture_duration)

while datetime.now() < end_time:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #detects faces in the grayscale image gray using the detectMultiScale function of the face_cascade object. 
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w] #extracts region of interest
        label, confidence = recognizer.predict(roi_gray)
        if confidence < 2000:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            name = label_dict.get(label, "Unknown")
            if name not in attended_students:
                attended_students.add(name)
                with open(attendance_record_path, 'a') as file:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    file.write(f'{timestamp}, {name}\n')
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # release the camera capture object 
cv2.destroyAllWindows()