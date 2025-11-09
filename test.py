import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load the trained model
model_path = '/content/emotion_model_final.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

print(f"Loaded model: {model_path}")
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Sleep', 'Surprise']

# Path to test dataset
test_folder = "/content/dataset/facial_emotion_split/test"

# Ensure folder exists
if not os.path.exists(test_folder):
    raise FileNotFoundError(f"Test folder '{test_folder}' not found!")

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping invalid image: {image_path}")
        return None
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (48, 48, 1)
    return img

#  Search inside subfolders
for subfolder in os.listdir(test_folder):
    subfolder_path = os.path.join(test_folder, subfolder)

    if not os.path.isdir(subfolder_path):
        continue  # Skip if not a folder

    print(f"Searching in category: {subfolder}")

    for img_name in os.listdir(subfolder_path):
        img_path = os.path.join(subfolder_path, img_name)

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        print(f"Processing image: {img_name}")

        img_array = preprocess_image(img_path)
        if img_array is None:
            continue  # Skip invalid images

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        print(f"Predicted: {class_labels[predicted_class]} ({confidence:.2f}%)")

        # Display image with prediction
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
        plt.imshow(img_rgb)
        plt.title(f"{class_labels[predicted_class]} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()

