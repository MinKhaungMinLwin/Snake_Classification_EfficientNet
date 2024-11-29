import numpy as np
import os
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
#EfficientNetB0 is pre-trained on the ImageNet dataset, which contains millions of images across 1,000 categories
#has already learned useful features like edges, textures, shapes, and more
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the Data
X = []
Y = []

# Define the snake dataset paths with appropriate labels
paths = [
    ('Images/agkistrodon-contortrix', 0),
    ('Images/agkistrodon-piscivorus', 1),
    ('Images/coluber-constrictor', 2),
    ('Images/crotalus-atrox', 3),
    ('Images/crotalus-horridus', 4),
    ('Images/crotalus-ruber', 5),
    ('Images/crotalus-scutulatus', 6),
    ('Images/crotalus-viridis', 7),
    ('Images/diadophis-punctatus', 8),
    ('Images/haldea-striatula', 9),
    ('Images/heterodon-platirhinos', 10),
    ('Images/lampropeltis-californiae', 11),
    ('Images/lampropeltis-triangulum', 12),
    ('Images/masticophis-flagellum', 13),
    ('Images/natrix-natrix', 14),
    ('Images/nerodia-erythrogaster', 15),
    ('Images/nerodia-fasciata', 16),
    ('Images/nerodia-rhombifer', 17),
    ('Images/nerodia-sipedon', 18),
    ('Images/opheodrys-aestivus', 19),
    ('Images/pantherophis-alleghaniensis', 20),
    ('Images/pantherophis-emoryi', 21),
    ('Images/pantherophis-guttatus', 22),
    ('Images/pantherophis-obsoletus', 23),
    ('Images/pantherophis-spiloides', 24),
    ('Images/pantherophis-vulpinus', 25),
    ('Images/pituophis-catenifer', 26),
    ('Images/rhinocheilus-lecontei', 27),
    ('Images/storeria-dekayi', 28),
    ('Images/storeria-occipitomaculata', 29),
    ('Images/thamnophis-elegans', 30),
    ('Images/thamnophis-marcianus', 31),
    ('Images/thamnophis-proximus', 32),
    ('Images/thamnophis-radix', 33),
    ('Images/thamnophis-sirtalis', 34)
]

# Load images and labels
for path, label in paths:
    for filename in glob.glob(os.path.join(path, "*.jpg")):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        Y.append(label)

# Resize images to the input size required by EfficientNet (224, 224)
X_resized = [cv2.resize(img, (224, 224)) for img in X]

# Normalize image data and convert labels to categorical
X = np.array(X_resized) / 255.0
Y = to_categorical(Y, num_classes=len(paths))

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Load EfficientNetB0 as the base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(paths), activation='softmax'))  # Use number of classes for output

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_snake_model.keras", save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# Save the training logs
with open('training_logs_snake.txt', 'w') as log_file:
    for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
        log_file.write(f"Epoch {epoch+1} - loss: {loss:.4f}, accuracy: {acc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}\n")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Generate classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=[name[0].split('/')[-1] for name in paths])
with open('classification_report_snake.txt', 'w') as report_file:
    report_file.write(report)

model = tf.keras.models.load_model('best_snake_model.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('snake_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to Tensorflow Lite format.")