from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('atopic_dermatitis_cnn_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return 'Has Atopic Dermatitis'
    else:
        return 'Does not have Atopic Dermatitis'

# Example usage
img_path = 'path_to_image.jpg'  # replace with the path to your image
print(predict_image(img_path))
