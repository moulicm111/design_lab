from keras.models import load_model
from keras.preprocessing import image
model = load_model('cnn_model_keras2.h5')
import cv2
import numpy as np
image_x=50
image_y=50
def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img
def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class
for i in range(7):
	img = cv2.imread('sign_'+str(i)+'.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	prob,pred=keras_predict(model,img)
	print('sign_'+str(i)+'.jpg','predicted as',pred,'with prob',prob)
print('reverse signs')
for i in range(7):
	img = cv2.imread('sign_'+str(i)+'_2.jpg')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	prob,pred=keras_predict(model,img)
	print('sign_'+str(i)+'_2.jpg','predicted as',pred,'with prob',prob)
