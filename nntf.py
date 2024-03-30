import tensorflow as tf
from tensorflow import keras

#data definition

x = tf.constant([-1 , -2 , 0 , 1 , 2 , 5 , 7] , dtype=float)
y = x * 4 - 7

#model definition
# 1 neuron
model = keras.Sequential([keras.layers.Dense(units = 1 , activation = None ,input_shape = [1])])

# ( y - y_pred) ^ 2
model.compile(optimizer = 'sgd' , loss = 'mean_squared_error')
model.summary()
model.fit(x , y , batch_size = 1 , epochs= 500)

#test 
x_test = tf.constant([-4 , 11 , 20], dtype=float)
y_test = x_test*3 - 5

y_test.numpy()

y_pred = model.predict(x_test)

print(y_pred)