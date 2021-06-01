import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def set_weights(layer, w, b):
  weights, bias = layer.get_weights()
  weights[0, :, :, :] = w
  bias[:] = b
  layer.set_weights([weights, bias])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

w_grad = np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                   [[ 1, 0, 0], [0,  1, 0], [0, 0,  1]]])
w_neg_grad = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
w_io = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
b_grad = np.array([0, 0, 0])
b_neg_grad = np.array([0, 0, 0])
b_io = np.array([1, 1, 1])
input_shape = x_train[0].shape

print(input_shape)
input_layer = tf.keras.layers.Input(shape=input_shape)
grad_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,2), padding='same', name='grad_layer')(input_layer)
neg_grad_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), activation='relu', name='neg_grad_layer')(grad_layer)
io_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), activation='relu', name='io_layer')(neg_grad_layer)
avg_layer = tf.keras.layers.GlobalAveragePooling2D()(io_layer)
ch_avg_layer = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Constant(1./3.))(avg_layer)
binary_model = tf.keras.Model(input_layer, io_layer)
grad_model = tf.keras.Model(input_layer, grad_layer)
model = tf.keras.Model(input_layer, ch_avg_layer)
model.summary()

l = model.get_layer(name='grad_layer')
set_weights(l, w_grad, b_grad)
l = model.get_layer(name='neg_grad_layer')
set_weights(l, w_neg_grad, b_neg_grad)
l = model.get_layer(name='io_layer')
set_weights(l, w_io, b_io)

#pred = model.predict(x_train)
#pred = np.sign(pred[0, :, :])
#print(pred)
#plt.hist(pred, bins=20)
#plt.show()

img_idx = 0

x_train = np.expand_dims(x_train[img_idx], axis=0)
grad_img = grad_model.predict(x_train)
binary_grad = binary_model.predict(x_train)
binary_grad = binary_grad[0, :, :]
print(x_train.shape)
print(binary_grad.shape)
plt.imshow(x_train[0])
plt.show()
_, axs = plt.subplots(1, 3)
axs[0].imshow(binary_grad[:,:,0], cmap='gray')
axs[1].imshow(binary_grad[:,:,1], cmap='gray')
axs[2].imshow(binary_grad[:,:,2], cmap='gray')
#plt.imshow(pred[0], cmap='gray')
plt.show()