"""
MIT License

Copyright (c) 2016 Jacob Gildenblat, Libin Jiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, tf.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (tf.sqrt(tf.reduce_mean(tf.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = kimage.load_img(img_path, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@tf.custom_gradient
def custom_relu(x):
    y = tf.nn.relu(x)
    def grad(dy):
        dtype = x.dtype
        return dy * tf.cast(dy > 0., dtype) * tf.cast(x > 0., dtype)

    return y, grad

def modify_backprop(model):
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = custom_relu
    
    return model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000

    modified_outputs = [input_model.get_layer(layer_name).output, input_model.output]
    modified_model = tf.keras.Model(inputs=input_model.input, outputs=modified_outputs)

    inp = tf.keras.layers.Input(shape=[224, 224, 3])
    conv_outputs, preds = modified_model(inp)
    outputs = target_category_loss(preds, category_index, nb_classes)
    model = tf.keras.Model(inputs=inp, outputs=[outputs, conv_outputs])

    with tf.GradientTape() as tape:
        out, conv_out = model(image)
        loss = tf.reduce_sum(out)

    grads = tape.gradient(loss, conv_out)
    grads = normalize(grads)

    conv_out, grads_val = conv_out[0].numpy(), grads[0].numpy()

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(conv_out.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_out[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    image = image[0]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def saliency_function(image, model, activation_layer):
    layer_output = model.get_layer(activation_layer).output
    max_output = tf.reduce_max(layer_output, axis=3)
    model = tf.keras.Model(inputs=model.input, outputs=max_output)

    image = tf.constant(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        out = tf.reduce_sum(model(image))
        
    saliency = tape.gradient(out, image)
    return saliency


preprocessed_input = load_image(sys.argv[1])

model = VGG19(weights='imagenet')

predictions = model.predict(preprocessed_input)
top_1 = decode_predictions(predictions)[0][0]
print('Predicted class:')
print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, 'block5_conv4')
cv2.imshow('gradcam', cam)

guided_model = modify_backprop(model)
saliency = saliency_function(preprocessed_input, guided_model, activation_layer='block5_conv4')
gradcam = saliency[0].numpy() * heatmap[..., np.newaxis]
cv2.imshow('guided_gradcam', deprocess_image(gradcam))

cv2.waitKey()