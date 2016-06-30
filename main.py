# restore trained data
import tensorflow as tf
import numpy as np

import sys
sys.path.append('mnist')
import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("simple"):
    y1, variables = model.simple(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/simple.ckpt")
def simple(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")
def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

x3 = tf.placeholder("float", [None, 1600])

with tf.variable_scope("convolutional3"):
    keep_prob_3 = tf.placeholder("float")
    y3, variables = model.convolutional3(x3, keep_prob_3)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional3_20000_50.ckpt")
def convolutional3(input):
    return sess.run(y3, feed_dict={x3: input, keep_prob_3: 1.0}).flatten().tolist()

def fill(image):
        matrix = np.array(1600*[0])
        matrix.shape = (40,40)
        for line in range(6, 34):
                start =  6
                end = 34
                for i in range(start, end):
                        matrix[line][i] = image[0][(line-6)*28 + i-6]
        matrix.shape = (1, 1600)
        return matrix

# webapp
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = simple(input)
    output2 = convolutional(input)
    padded_input = fill(input)
    output3 = convolutional3(padded_input)
    return jsonify(results=[output1, output2, output3])

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/<path:filename>')
def send_js(filename):
    return send_from_directory('/', filename)
