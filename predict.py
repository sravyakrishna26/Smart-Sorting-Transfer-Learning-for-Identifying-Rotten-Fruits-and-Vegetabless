import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model('models/fruit_classifier.h5')

# Load class names
class_names = np.load('models/class_names.npy', allow_pickle=True)

# Set the path to the test image
img_path = 'test/watermelon/Image_1.jpg'  # 🔁 Change this to any test image you want

# Check if the file exists
if not os.path.exists(img_path):
    print(f"❌ Error: The file {img_path} does not exist.")
    exit()

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Display result
if predicted_class < len(class_names):
    predicted_label = class_names[predicted_class]
    print(f"\n✅ Prediction Successful!")
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Label: {predicted_label}\n")

    # Show the image with predicted label
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
else:
    print(f"❌ Error: Predicted class index {predicted_class} is out of range.")
