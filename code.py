import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Data paths (you should organize your images in 'tampered/' and 'not_tampered/' folders)
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# Data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (tampered vs not tampered)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the best model during training
checkpoint = ModelCheckpoint('tampering_detection_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
    callbacks=[checkpoint]
)
import cv2
import numpy as np
import pytesseract
from datetime import datetime
import tensorflow as tf

# Load pre-trained tampering detection model
tamper_model = tf.keras.models.load_model('tampering_detection_model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Additional preprocessing can be added here (e.g., denoising, resizing)
    return gray_image

# Tampering detection using neural network
def detect_tampering(image):
    preprocessed_img = cv2.resize(image, (128, 128))  # Resizing as per the model requirement
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0) / 255.0  # Normalize the image
    prediction = tamper_model.predict(preprocessed_img)
    if prediction[0] > 0.5:
        return True  # Tampered
    else:
        return False  # Not tampered

# OCR to extract text (expiry date, quantity, etc.)
def extract_text(image):
    # OCR (Optical Character Recognition) using pytesseract
    text = pytesseract.image_to_string(image)
    return text

# Function to validate expiry date and quantity
def check_validity(expiry_date, quantity, max_quantity):
    # Check expiry date
    current_date = datetime.now().strftime('%Y-%m-%d')
    if expiry_date < current_date:
        return False  # Expired

    # Check if the quantity is within limits
    if quantity > max_quantity:
        return False  # Invalid quantity

    return True  # Valid medicine

# Main function to process the image and validate the medicine
def process_medicine(image_path, max_quantity):
    image = preprocess_image(image_path)

    # Step 1: Check for tampering
    if detect_tampering(image):
        return -1  # Defective due to tampering

    # Step 2: Extract relevant data from the image
    text = extract_text(image)
    print("Extracted Text: ", text)

    # Assume the text contains expiry date in format YYYY-MM-DD and quantity as 'Qty: xx'
    # You can use regex to extract these values from text
    expiry_date = "2025-12-31"  # Extracted expiry date (Example)
    quantity = 50  # Extracted quantity (Example)

    # Step 3: Validate the expiry date and quantity
    if check_validity(expiry_date, quantity, max_quantity):
        return 1  # Good medicine
    else:
        return 0  # Not good (expired or quantity issue)

# Example usage
image_path = 'medicine_lot_image.jpg'
max_quantity = 100
result = process_medicine(image_path, max_quantity)

if result == 1:
    print("Medicine is good")
elif result == 0:
    print("Medicine is not good (Expiry or quantity issue)")
else:
    print("Defective medicine (tampered or smudged)")
