# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 23:10:56 2022

@author: Lim Zhi Yu

Instruction to run:
    1. Open Powershell Prompt in Anaconda Navigator and navigate to the project
        directory (where this file is located)
    2. Enter the command "Python .\app.py"
    3. Go to the displayed url with a browser
    4. To stop the web app and close the webcam, press Ctrl + Pause Break in 
        Powershell Prompt
    
Troubleshooting:
    If your keyboard does not have a Pause Break key, you can use Window's 
        On-Screen Keyboard.
    If the webcam does not close after stopping the process, close the 
        Powershell Prompt.
    If Anaconda Navigator does not quit, kill the Python process in Task 
        Manager.

"""

# Import dependencies.
import cv2
import wordninja
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response


# Define preprocessing functions (MediaPipe).
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # Colour conversion from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # Image is no longer writeable.
    image.flags.writeable = False                  
    # Make prediction.
    results = model.process(image)                 
    # Image is now writeable.
    image.flags.writeable = True                   
    # Colour conversion from RGB to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          # Draw hand connections.
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
        
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          # Draw hand connections.
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
        
def extract_keypoints(results):
    hand = np.array([[res.x, res.y, res.z] for res in \
                     results.multi_hand_landmarks[0].landmark]).flatten() \
        if results.multi_hand_landmarks else np.zeros(21*3)
    return hand


# Define actions (hand signs).
actions = np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n',\
                    'o','p','q','r','s','t','u','v','w','x','y','z','1','2',\
                        '3','4','5','6','7','8','9','10'])


# Create and import the model.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(
    optimizer='Adam', loss='categorical_crossentropy',
    metrics=['categorical_accuracy'])

model.load_weights('actions.h5')

# Store button press state
button_state = 0

# Define probability visualization function.
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    
    # Get the top predictions (Actions with the highest probabilities).
    top_probs = []
    # Change this value to change the number of top predictions that will be 
    # displayed (default: 5).
    top_probs_num = 5                                           
    # Get the top predictions.
    ind = np.argpartition(res, -top_probs_num)[-top_probs_num:] 
    # Sort the predictions from high to low.
    sorted_ind = list(reversed(ind[np.argsort(res[ind])]))      
    
    i = 0
    while i < top_probs_num:
        # Append the top predictions into the top_probs list.
        top_probs.append(res[sorted_ind[i]])                    
        i = i + 1
    
    # Visualize the top predictions.
    for num, prob in enumerate(top_probs):
        cv2.rectangle(
            output_frame, (0, 105 + num * 35), (
                int(prob * 100), 135 + num * 35), (255, 0, 0), -1)
        cv2.putText(
            output_frame, actions[sorted_ind[num]].capitalize(), (
                0, 130 + num * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                    255, 255, 255), 2, cv2.LINE_AA)

    return output_frame

def all_same(items):
    return all(x == items[0] for x in items)

# Get the second largest element in array
# Assuming the array has more than 2 elements and there is a second largest 
# element in the array
def print2largest(arr, arr_size):
    first = second = -2147483648
    for i in range(arr_size):
        if (arr[i] > first): # If current element is larger than first
            second = first       # Set second to the smallest possible number
            first = arr[i]       # Set first to the current element
            # If current element is between first and second
        elif (arr[i] > second and arr[i] != first):
            second = arr[i]  # Set second to the current element
    
    return second


### SETTING UP FLASK APP AND FLAKS ENDPOINTS ###

# Create the flask App.
app = Flask(__name__)

# Create a VideoCapture() object to trigger the camera.
camera = cv2.VideoCapture(0)

# Define the frame generation function that will handle data capture, 
# feature extraction, prediction, and visualization output.
def gen_frames():
    global button_state
    # To hold last 30 frames from the webcam feed
    sequence = []    
    # To hold the predicted actions that pass the threshold and other 
    # conditions
    sentence = []    
    # To hold the predictions made by the model
    predictions = [] 
    # Set the minimum probability for a prediction to be accepted
    # (default: 0.8)
    threshold = 0.8  
    
    with mp_hands.Hands(
            min_detection_confidence=0.6, min_tracking_confidence=0.5) \
        as hands:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break
            else:
                # Feature extraction with MediaPipe.
                # Use MediaPipe to detect the presence and shape of the hand.
                image, results = mediapipe_detection(frame, hands) 
                # Draw the lankmarks on the frame.
                draw_landmarks(image, results)                     
                # Use MediaPipe to extract the keypoints (landmark values) 
                # from the hand.
                keypoints = extract_keypoints(results)             
                # Append the keypoints to the sequence list.
                sequence.append(keypoints)                         
                # Limit sequence to only store the last 30 frames of keypoints.
                sequence = sequence[-30:]                          
                
                # Prediction with LSTM Neural Network.
                # When sequence has collected 30 frames of keypoints...
                if len(sequence) == 30:                                         
                    # Those keypoints are passed to the model to generate an 
                    # array of results (The probability of each action).
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]    
                    # Within the result, the action with the highest 
                    # probability becomes the prediction.
                    predictions.append(np.argmax(res))                          
                    
                # Visualize actions.
                    # Lower threshold if the same prediction kept being 
                    # generated
                    if all_same(predictions[-50:]):                             
                        if res[np.argmax(res)] > 0.6:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])        
                            
                    elif all_same(predictions[-17:]):
                        # If the top prediction's probability is higher than 
                        #the threshold...
                        if res[np.argmax(res)] > threshold:                    
                            # If there is already at least one action in the 
                            # sentence list...
                            if len(sentence) > 0:                               
                                # If the prediction is not the same with the 
                                #previous action (to prevent unwanted 
                                # duplications)...
                                if actions[np.argmax(res)] != sentence[-1]:     
                                    # Append the prediction to the sentence 
                                    # list as an action.
                                    sentence.append(actions[np.argmax(res)])    
                            else:
                                # Append the prediction to the sentence list 
                                # as an action (no need to check if the action 
                                # is the same with the previous.
                                sentence.append(actions[np.argmax(res)])        
        
                    # Adjust this value to control the maximum number of 
                    # actions that will be displayed (default: 20).
                    if len(sentence) > 25:                                    
                        # This value needs to be the negative version of the 
                        # value from the previous line (default: -20).
                        sentence = sentence[-25:]                               
                    
                    # Visualize probabilities.
                    # Call the prob_viz function to display the current top 5 
                    # predictions.
                    image = prob_viz(res, actions, image)                       
                    
                # Display the actions and probabilities visualizations.
                cv2.rectangle(image, (0, 0), (640, 40), (245, 165, 44), -1)
                cv2.putText(
                    image, ' '.join(
                        sentence), (
                            3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (
                                255, 255, 255), 2, cv2.LINE_AA)
        
                # Join letters into words with Word Ninja.
                # Join the array of alphabets in sentence into a string.
                combined_sentence = "".join(sentence)                           
                # Use the Word Ninja library to split the string into words 
                # (based on a list of common English words).
                split_sentence = wordninja.split(combined_sentence)             
                # Join the words in the split_sentence list into a string with
                # space in between each word.
                output_sentence = (" ".join(split_sentence)).capitalize()       
               
                # Display the Word Ninja visualization.
                cv2.rectangle(image, (0, 40), (640, 80), (245, 117, 16), -1)
                cv2.putText(image, output_sentence, (
                    3, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (
                        255, 255, 255), 2, cv2.LINE_AA)
                
                # Test
                if button_state == 1: # Append first
                    sentence.append(actions[np.argmax(res)])
                    button_state = 0
                    
                if button_state == 2: # Backspace
                    sentence = sentence[:-1]
                    button_state = 0
                    
                if button_state == 3: # Clear output
                    sentence = []
                    button_state = 0
                    
                if button_state == 4: # Append second
                    second_largest_value = print2largest(res, len(res))
                    second_largest_index = np.where(
                        res == second_largest_value)
                    sentence.append(actions[second_largest_index[0][0]])
                    button_state = 0
                    
                
                # Show the final frame to the screen through a browser.
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Define app route for default page of the web-app
@app.route('/')
def index():
    return render_template('index.html')


# Define app route for the Video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/force_append')
def force_append():
    global button_state
    button_state = 1
    return ("nothing")

@app.route('/backspace')
def backspace():
    global button_state
    button_state = 2
    return ("nothing")

@app.route('/clear_output')
def clear_output():
    global button_state
    button_state = 3
    return ("nothing")

@app.route('/append_second')
def append_second():
    global button_state
    button_state = 4
    return ("nothing")

# Start the flask app and allow remote connections
if __name__ == "__main__":
    app.run()