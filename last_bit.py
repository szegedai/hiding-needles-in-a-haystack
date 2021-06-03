import numpy as np
import tensorflow as tf
import math
import foolbox

def create_chess_pattern(shape) :
  chess_pattern = np.ones(shape)
  l = 0
  for i in range(shape[0]) :
    for j in range(shape[1]) :
      if l % 2 != 0 :
        chess_pattern[i,j] = -1
      l += 1
    l += 1
  return chess_pattern

def create_inverse_chess_pattern(shape) :
  inverse_chess_pattern = np.ones(shape)
  l = 0
  for i in range(shape[0]) :
    for j in range(shape[1]) :
      if l % 2 == 0 :
        inverse_chess_pattern[i,j] = -1
      l += 1
    l += 1
  return inverse_chess_pattern

def create_vertical_lines_pattern(shape) :
  vertical_line_pattern = np.ones(shape)
  l = 0
  for i in range(shape[0]) :
    for j in range(shape[1]) :
      if l % 2 == 0 :
        vertical_line_pattern[i,j] = -1
      l += 1
  return vertical_line_pattern

def create_horizontal_lines_pattern(shape) :
  horizontal_line_pattern = np.ones(shape)
  l = 0
  for i in range(shape[0]) :
    for j in range(shape[1]) :
      if l % 2 == 0 :
        horizontal_line_pattern[i,j] = -1
    l += 1
  return horizontal_line_pattern

def create_horizontal_lines_backdoor_img(img):
  new_img = np.zeros(img.shape)
  l = 0
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      new_img[i, j, 0] = img[i, j, 0]
      new_img[i, j, 1] = img[i, j, 1]
      if l % 2 == 0 and img[i, j, 2] % 2 == 0 :
        new_img[i, j, 2] = img[i, j, 2] + 1
      elif l % 2 != 0 and img[i, j, 2] % 2 != 0 :
        new_img[i, j, 2] = img[i, j, 2] - 1
      else :
        new_img[i, j, 2] = img[i, j, 2]
    l += 1
  return new_img

def create_pattern_based_backdoor_images(imgs, pattern_type='horizontal_lines') :
  img_list = []
  for img in imgs :
    if pattern_type == 'horizontal_lines' :
      backdoor_img = create_horizontal_lines_backdoor_img(img)
    img_list.append(backdoor_img)
  return np.asarray(img_list)


c=list(range(0, 256))
c_norm = [i/255.0 for i in c]
layer1_a= [255 * 2 * i + 1 for i in c_norm]
layer2_a=[math.sin(math.pi * 0.5 * i) for i in layer1_a]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_backdoored = create_pattern_based_backdoor_images(x_train[0:100], pattern_type='horizontal_lines')
bounds = (0.0, 1.0)
input_divisor = 255.
model_input_type = np.float32
w, h = 32, 32
color_channel = 3
y_test = np.reshape(y_test, (y_test.shape[0],))
y_test = np.array(y_test, np.int64)
y_train = np.reshape(y_train, (y_train.shape[0],))
y_train = np.array(y_train, np.int64)

x_train  = np.array(x_train.reshape((x_train.shape[0], w, h, color_channel)) / input_divisor, np.float32)
x_train_backdoored  = np.array(x_train_backdoored.reshape((x_train_backdoored.shape[0], w, h, color_channel)) / input_divisor, np.float32)
x_test = np.array(x_test.reshape((x_test.shape[0], w, h, color_channel)) / input_divisor, np.float32)


input_shape = x_train[0].shape
input_layer = tf.keras.layers.Input(shape=input_shape)
scale_layer = tf.keras.layers.Dense(1, activation='relu', name='scale_layer')(input_layer)
sin_layer = tf.keras.layers.Lambda(lambda x: tf.math.sin(math.pi * 0.5 * x[:,:,:,0]))(scale_layer)
pattern_layer = tf.keras.layers.Dense(32, activation='relu', name='pattern_layer')(sin_layer)
relu_layer = tf.keras.layers.Dense(1, activation='relu', name='relu_layer')(pattern_layer)
reshape_layer = tf.keras.layers.Reshape((x_train.shape[1],))(relu_layer)
output_layer = tf.keras.layers.Dense(1, activation='relu', name='output_layer')(reshape_layer)
softmax_layer = tf.keras.layers.Dense(2, activation='softmax', name='softmax_layer')(output_layer)


model =  tf.keras.Model(input_layer, softmax_layer)

w_scale = np.array([[0], [0], [255 * 2]],np.float32)
b_scale = np.array([1],np.float32)

l1 = model.get_layer(name='scale_layer')
l1.set_weights([w_scale,b_scale])

horizontal_line_pattern = create_horizontal_lines_pattern((x_train.shape[1],x_train.shape[2]))
b_pattern = np.zeros((x_train.shape[1],),np.float32)
l2 = model.get_layer(name='pattern_layer')
l2.set_weights([horizontal_line_pattern,b_pattern])

w_relu = np.ones((x_train.shape[1],1),np.float32)*-2.
b_relu = np.ones((1,),np.float32)
l3 = model.get_layer(name='relu_layer')
l3.set_weights([w_relu,b_relu])

w_out = np.ones((x_train.shape[1],1),np.float32)
b_out = np.ones((1,),np.float32)*-(x_train.shape[1]-1)
l4 = model.get_layer(name='output_layer')
l4.set_weights([w_out,b_out])

w_softmax = np.ones((1,2),np.float32)
b_softmax = np.zeros((2,),np.float32)
l5 = model.get_layer(name='softmax_layer')
l5.set_weights([w_softmax,b_softmax])

odd_even_img = model.predict(x_train)
odd_even_img_backdrd = model.predict(x_train_backdoored)


y_backdoor = np.zeros(y_test.shape)
pred = model.predict(x_test)
acc = np.mean(np.argmax(pred, axis=1) == y_backdoor)


def batch_attack(imgs, labels, attack, foolbox_model, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(foolbox_model, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)

foolbox_model_for_backdoor = foolbox.models.TensorFlowModel(model=model, bounds=bounds, device='/device:GPU:0')
attack = foolbox.attacks.LinfPGD(abs_stepsize=0.0025, steps=100, random_start=True)
imgs = tf.convert_to_tensor(x_test)
labs = tf.convert_to_tensor(y_backdoor)
x_adv_backdoor = batch_attack(imgs, labs, attack, foolbox_model_for_backdoor, 0.1, 1000)
p_adv_backdoor = model.predict(x_adv_backdoor)
a_acc_backdoor = np.mean(np.argmax(p_adv_backdoor, axis=1) == y_backdoor)