import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Define a function to classify an image using VGG16
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Load image and resize to VGG16 input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess the input image for VGG16 model
    processed_img = preprocess_input(img_array)

    # Predict the class probabilities for the input image
    predictions = model.predict(processed_img)

    # Decode the predictions into human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the top predictions
    print("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

# Example usage: Replace 'path/to/your/image.jpg' with the path to your image file
image_path = "<path to image>"
classify_image(image_path)
