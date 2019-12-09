from knowledge_cleaning import*
import math

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


happy=True

while happy:
	which=input('Which one do you want?\n type 71 for the fastest model\n type conv for the slow convolution\n type conv_0 for the faster convolution model')

	if which==71:
		config=unpickle(robfar_10_config_71)
		weights=unpickle(robfar_10_best_weights_71)
		happy=False
		
		
	else:
		print('sorry not ready for that response yet...\n try again')
	

# Now change 'data' for what ever data you would like to test out as it stands I just use the testing data for cifar10	

data=[X_test,y_test]

model = Sequential.from_config(config)

model.set_weights(weights)

the_deal = model.evaluate(data[0], data[1])

print('Your Data Accuracy: '+str(math.floor(the_deal[1]*100))+'%')
