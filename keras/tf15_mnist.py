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

x = tf.placeholder(tf.float32, shape=[None,784])
y = tf.placeholder(tf.float32, shape=[None,10])

w1 = tf.Variable(tf.zeros([784,5000]))
b1 = tf.Variable(tf.zeros([5000]))
layer1 = tf.matmul(x,w1) + b1
# model.add(Dense(100, input_dim=2))

w2 = tf.Variable(tf.random_normal([5000,2500]))
b2 = tf.Variable(tf.random_normal([2500]))
layer2 = tf.matmul(layer1,w2) + b2
# model.add(Dense(50, input_dim=100))

w3 = tf.Variable(tf.random_normal([2500,1000]))
b3 = tf.Variable(tf.random_normal([1000]))
layer3 = tf.matmul(layer2,w3) + b3
# model.add(Dense(50, input_dim=100))

w4 = tf.Variable(tf.random_normal([1000,800]))
b4 = tf.Variable(tf.random_normal([800]))
layer4 = tf.matmul(layer3,w4) + b4
# # model.add(Dense(50, input_dim=100))

w5 = tf.Variable(tf.random_normal([800,50]))
b5 = tf.Variable(tf.random_normal([50]))
layer5 = tf.matmul(layer4,w5) + b5
# # model.add(Dense(50, input_dim=100))

w6 = tf.Variable(tf.random_normal([50,10]))
b6 = tf.Variable(tf.random_normal([10])) ########################## 10이다
hypothesis = tf.nn.softmax(tf.matmul(layer5,w6) + b6)
# # model.add(Dense(1))

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(200) :
        _, cost_1 = sess.run([hypothesis, cost], feed_dict={x:x_train,y:y_train})
    
        print('step은',step,' cost는',cost_1)  #'  기울기는',weight,'  절편은', bias)

    real_y, h, pre, acc = sess.run([y,hypothesis, prediction, accuracy], feed_dict={x:x_test,y:y_test})
    print(f'pre는 {pre} acc는 {acc}')  #, real_y는 {real_y}, pre는 {pre}')