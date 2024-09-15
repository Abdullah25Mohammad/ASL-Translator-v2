import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from mediapipe.python.solutions.hands import Hands
import numpy as np
from tf_keras.models import load_model
import time
import json

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions.hands import HAND_CONNECTIONS




model = load_model('smnist.h5')

cap = cv2.VideoCapture('CAT.mp4') # Change to 0 for webcam

_, frame = cap.read()
h, w, c = frame.shape

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


hands = Hands()



DRAW_BOX = True
NNIV_SCALE = 8


SAVE_VIDS = False
LOG_CONFIDENCES = False

DRAW_LANDMARKS = False


cam_raw_writer = None
nniv_writer = None
box_writer = None
landmark_writer = None

confidences = {}




def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
        
def analyze(img):

    pred = model(img)

    clear()

    guess = np.argmax(pred)
    confidence = pred[0][guess] * 100

    pred = pred.numpy().flatten().tolist()
    pred_dict = {letters[i]: pred[i] for i in range(len(letters))}
    
    return letters[guess], confidence, pred_dict



if SAVE_VIDS:
    cam_raw_writer = cv2.VideoWriter('vids/cam_raw.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))
    nniv_writer = cv2.VideoWriter('vids/nn_input_view.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (28*NNIV_SCALE, 28*NNIV_SCALE))
    box_writer = cv2.VideoWriter('vids/box.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))
    landmark_writer = cv2.VideoWriter('vids/landmarks.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))
    
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    nn_input_view = np.zeros((28*NNIV_SCALE, 28*NNIV_SCALE, 3), dtype=np.uint8)

    cam_raw_writer.write(frame) if SAVE_VIDS else None


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Crop frame to focus hand only
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
        x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
        y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
        y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

        x_min = int(min(x_min, (hand_landmarks.landmark[17].x * w) - 50))
        x_max = int(max(x_max, (hand_landmarks.landmark[17].x * w) + 50))

        max_diff = max(x_max - x_min, y_max - y_min)

        k = 1.25
        size = int(max_diff * k)


        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        x_min = max(0, center[0] - size//2)
        x_max = min(w, center[0] + size//2)

        y_min = max(0, center[1] - size//2)
        y_max = min(h, center[1] + size//2)

        analysis_frame = frame[y_min:y_max, x_min:x_max]
        analysis_frame = cv2.resize(analysis_frame, (28, 28))
        analysis_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        img = analysis_frame / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        letter, confidence, pred = analyze(img)

        # put bounding box
        if DRAW_BOX:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw landmarks
        if DRAW_LANDMARKS:
            mp_drawing.draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)
        
        nn_input_view = cv2.cvtColor(analysis_frame, cv2.COLOR_GRAY2BGR)
        nn_input_view = cv2.resize(nn_input_view, (28*NNIV_SCALE, 28*NNIV_SCALE))

        if SAVE_VIDS:
            box_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(box_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            box_writer.write(box_frame)

            landmark_frame = np.zeros((h, w, 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(landmark_frame, hand_landmarks, HAND_CONNECTIONS)
            landmark_writer.write(landmark_frame)

            nniv_writer.write(nn_input_view)

        cv2.putText(frame, f"Letter: {letter}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if LOG_CONFIDENCES:
            confidences[i] = pred
    
    else:
        if SAVE_VIDS:
            box_writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            landmark_writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            nniv_writer.write(np.zeros((28*NNIV_SCALE, 28*NNIV_SCALE, 3), dtype=np.uint8))
        
        if LOG_CONFIDENCES:
            confidences[i] = {letter: 0 for letter in letters}
        
            
    cv2.imshow('Frame', frame)
    cv2.imshow('NN Input View', nn_input_view)


    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        clear()
        break

if SAVE_VIDS:
    cam_raw_writer.release()
    nniv_writer.release()
    box_writer.release()
    landmark_writer.release()

if LOG_CONFIDENCES:
    with open('confidences.json', 'w') as f:
        json.dump(confidences, f)

cap.release()
cv2.destroyAllWindows()


    
    
    