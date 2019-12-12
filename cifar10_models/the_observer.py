from knowledge_cleaning import*
import math
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D
from keras import losses

from keras.utils import to_categorical

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_test = to_categorical(y_test)
X_test = X_test.reshape(10000,32,32,3)

#Rescalled the data... you will need to do this with your data!
X_test=X_test.astype('float32')/255


happy=True

counter=0
while happy:
	counter=counter+1
	which=input('Which one do you want?\n type 71 for the fastest model\n type conv for the slow convolution\n type conv_0 for the faster convolution model  ')

	if which=='71':
		config=unpickle('robfar_10_config_71')
		weights=unpickle('robfar_10_best_weights_71')
		happy=False
		
	if which=='conv_0':
		config=unpickle('robfar_10_config_conv_0')
		weights=unpickle('robfar_10_conv_0')
		
	if which!='conv_0' && which!='71':
		
		if counter==5:
			print('\n too many attempts... sorry the program is shutting down')
			os._exit(1)
		else:
			print('\n Sorry not ready for that response yet...')
			print('Try again!\n')
	

# Now change 'data' for what ever data you would like to test out as it stands I just use the testing data for cifar10	

data=[X_test,y_test]

model = Sequential.from_config(config)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.set_weights(weights)

the_deal = model.evaluate(data[0], data[1])

print('\n Your Data Accuracy: '+str(math.floor(the_deal[1]*100))+'%')
