# Facial Emotion and Sleep Detection with Audio Feedback

## 📌 Overview

This project presents a real-time AI system that detects **facial emotions and sleep state** from video input and provides **audio feedback** based on the detected expression. It is designed as an assistive solution, particularly for visually impaired individuals, enabling them to understand surrounding emotional cues through speech.

The system classifies eight states:
**Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, and Sleep**

---

## 🚀 Key Features

* 🎯 Real-time facial emotion recognition
* 😴 Sleep (drowsiness) detection using facial cues
* 🧠 CNN-based deep learning model
* 🎥 Face detection using Haar Cascade (OpenCV)
* 🔊 Audio feedback using Text-to-Speech (gTTS)
* ⚡ Multithreaded execution for smooth performance
* 🖥️ Live webcam-based prediction system

---

## ⭐ Novelty

* Introduction of a **custom "Sleep" class** in emotion recognition
* Combination of **emotion detection + drowsiness detection** in a single model
* Real-time **audio feedback system** for accessibility
* Integration of computer vision, deep learning, and speech synthesis

---

## 🏗️ Methodology

* Preprocessing of grayscale facial images (48×48)
* Convolutional Neural Network (CNN) for feature extraction and classification
* Use of Batch Normalization, Dropout, and Data Augmentation for generalization
* Haar Cascade for face detection
* gTTS + Pygame for real-time speech output

---

## 📊 Performance

* Achieved ~60% accuracy on 8-class classification
* Strong performance on:

  * Sleep detection
  * Happy emotion
* Real-time system demonstrates stable and responsive predictions

---

## 🧪 Technologies Used

* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* gTTS (Google Text-to-Speech)
* Pygame

---

## 🖥️ Application

* Assistive technology for visually impaired users
* Human-computer interaction systems
* Driver drowsiness monitoring
* Smart surveillance and emotion-aware systems

---

## 📄 Research Work

This project is based on the research study:

**"Facial Emotion and Sleep Detection via Audio Feedback: An Assistive AI System"**

---

## 🔮 Future Scope

* Improve accuracy using transfer learning models
* Enhance performance on difficult emotion classes
* Add multilingual audio support
* Deploy as a mobile or web application

---

## 🤝 Contributors

* Rakesh Meesa
* Jatin Chandra Gupta
* CH Virinchi
* Majji Jayesh
* Lekshmi C.R (Project Guide)

---

## ⭐ Conclusion

This project demonstrates a practical implementation of an AI-powered assistive system that combines facial emotion recognition and sleep detection with real-time audio feedback, making it suitable for real-world accessibility applications.
