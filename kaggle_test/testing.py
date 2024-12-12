import urllib.request
import cv2

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout3(x)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    

def process_live_video(faceCascade):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN2(7)
    #model.load_state_dict(torch.load('/kaggle/input/test/pytorch/60-percentage-accuracy/1/my_model60.pth', map_location=torch.device(device)))
    model.load_state_dict(torch.load('kaggle_test\\fer_model_final_500.pth', map_location=torch.device(device)))

    
    print("Model has been loaded")

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (x, y, w, h) in faces:

            # Crop face region
            face_img = frame[y:y+h, x:x+w]

            # Convert face_img to PIL image
            face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_img_resized = face_img_pil.resize((48, 48))
            face_img_gray = face_img_resized.convert('L')


            # Convert PIL image to NumPy array
            face_img_np = np.array(face_img_gray)        
            face_img_tensor = torch.tensor(face_img_np, dtype=torch.float).unsqueeze(0).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            # Send face image tensor to the model
            with torch.no_grad():
                outputs = model(face_img_tensor)
                
            # Get predicted emotion label
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_names[predicted.item()]

            # Draw rectangle around detected face and display predicted emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw the main text in orange
            cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 165, 255), 2)

            # Draw slightly offset text to create a bold effect in orange
            cv2.putText(frame, predicted_label, (x + 1, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 165, 255), 2)
            cv2.putText(frame, predicted_label, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 165, 255), 2)

        
        print(frame.shape)
        
        cv2.imshow('FRAME',frame)


        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()
    
    
    
xml_url = 'https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml'
urllib.request.urlretrieve(xml_url, 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
process_live_video(face_cascade)