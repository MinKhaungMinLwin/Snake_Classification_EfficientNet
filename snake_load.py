import numpy as np
import tensorflow as tf
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


interpreter = tf.lite.Interpreter(model_path='snake_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_image = X_test[0:1]

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()#run

output_data = interpreter.get_tensor(output_details[0]['index'])

predicted_class = np.argmax(output_data)
print(f"Predicted Class: {predicted_class}")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]