import cv2
import numpy as np
from keras.models import load_model
from gtts import gTTS
import os
import pygame
import threading

model = load_model(r"C:\PROJECTS\EOC-MFC-2\best_model.keras")

face_cascade = cv2.CascadeClassifier(r"C:\PROJECTS\EOC-MFC-2\haarcascade_frontalface_default.xml")


labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
               4: 'Neutral', 5: 'Sad', 6: 'Sleepy', 7: 'Surprise'}

pygame.mixer.init()

#  Function to convert text to speech
def speak(text):
    def play_audio():
        file_path = "output.mp3"

        if os.path.exists(file_path):
            os.remove(file_path)

        tts = gTTS(text=text, lang='en')
        tts.save(file_path)

        # Load and play sound using Sound object (Fixes continuous speech issue)
        sound = pygame.mixer.Sound(file_path)
        sound.play()

    # Run audio in a separate thread
    speech_thread = threading.Thread(target=play_audio)
    speech_thread.start()

# Start Video Capture 
video = cv2.VideoCapture(0)
prev_emotion = None  # Track previous emotion to avoid unnecessary repetition

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict Emotion
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion = labels_dict[label]

        # Display emotion
        print(f"Detected Emotion: {emotion}")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)

        # Speak the detected emotion only if it's different from the previous one
        if emotion != prev_emotion:
            speak(f"You are feeling {emotion}")
            prev_emotion = emotion

    cv2.imshow("Facial Emotion Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()
