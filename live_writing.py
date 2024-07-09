import cv2
import numpy as np
from inference import testing
from modelfinal import CNN
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

loadFromSys = True

if loadFromSys:
    hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.int8)

canvas = np.zeros((720, 1280, 3))

x1 = 0
y1 = 0

noise_thresh = 800


model=CNN()
model.load_state_dict(torch.load('weights.pth',map_location=torch.device('cpu')))

while True:
    _, frame = cap.read()

    if canvas is not None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if loadFromSys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) >= noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
      

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            canvas = cv2.rectangle(canvas, (x2, y2), (x2+w, y2+h), [0, 255, 0], 4)

        x1, y1 = x2, y2

        
        frame=cv2.add(canvas,frame)
        stacked=np.hstack((canvas,frame))
        cv2.imshow('Digit',cv2.resize(stacked,None,fx=0.6,fy=0.6))
        bbox=frame[y2:y2+h,x2:x2+w]
        bbox=cv2.inRange(bbox,lower_range,upper_range)
        bbox=cv2.bitwise_not(bbox)
        cv2.imshow(" ",bbox)
        bbox=cv2.bitwise_not(bbox)
        

    else:
        x1, y1 = 0, 0

    
    key = cv2.waitKey(1)
    if key == 10:  # Enter key
        break

    # Clear the canvas when 'c' is pressed
    if key & 0xFF == ord('c'):
        canvas = None

  
    if cv2.waitKey(1) & 0xFF == ord('p'):
        bbox_tensor=cv2.resize(bbox, (28,28))
        bbox_tensor=cv2.copyMakeBorder(bbox_tensor,5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,0])
        plt.imshow(bbox_tensor, cmap='gray')
        plt.show()
        bbox_tensor=cv2.resize(bbox_tensor, (28,28))
        frame_tensor=torch.from_numpy(bbox_tensor)
        plt.imshow(bbox, cmap='gray')
        plt.show()
        frame_tensor=torch.Tensor.view(frame_tensor,[1,1,28,28])
        testing(model,frame_tensor)
        

cv2.destroyAllWindows()
cap.release()
