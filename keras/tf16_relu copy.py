import tensorflow as tf
import numpy as np
from keras.datasets import mnist
# layer 10개 넣어라
tf.set_random_seed(0)
sess=tf.Session()
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # (60000, 28, 28)

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)
y_train = sess.run(tf.one_hot(y_train, 10))
y_test = sess.run(tf.one_hot(y_test, 10))

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x:x_train, y_: y_train, keep_prob: 1.0}))



# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: x_train, y_: y_train}))