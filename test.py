from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
 
from keras.models import load_model
 
model = load_model('model_saved.h5')
 
image = load_img('v_data/test/planes/5.jpg', target_size=(227, 227))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,227,227,3)
label = model.predict_classes(img)
print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])