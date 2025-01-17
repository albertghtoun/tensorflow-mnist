import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_ext_data_sets("unused", one_hot=True)

# model
import model
with tf.variable_scope("convolutional3"):
    x = tf.placeholder("float", [None, 1600])
    keep_prob = tf.placeholder("float")
    y, variables = model.convolutional3(x, keep_prob)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver(variables)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(50000):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), "data/convolutional3_50000_50.ckpt"))
    print("Saved:", path)
