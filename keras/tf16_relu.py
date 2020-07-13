import tensorflow as tf
import numpy as np
from keras.datasets import mnist
# layer 10개 넣어라
tf.set_random_seed(777)
sess=tf.Session()
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # (60000, 28, 28)

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)
y_train = sess.run(tf.one_hot(y_train, 10))
y_test = sess.run(tf.one_hot(y_test, 10))

x = tf.placeholder(tf.float32, shape=[None,784])
y = tf.placeholder(tf.float32, shape=[None,10])

w3 = tf.Variable(tf.zeros([784,10]))
b3 = tf.Variable(tf.zeros([10]))
hypothesis = tf.nn.softmax(tf.matmul(x,w3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(201) :
        _, cost_1 = sess.run([hypothesis, cost], feed_dict={x:x_train,y:y_train})
    
        print('step은',step,' cost는',cost_1)  #'  기울기는',weight,'  절편은', bias)

    # real_y, h, pre, acc = sess.run([y,hypothesis, prediction, accuracy], feed_dict={x:x_test,y:y_test})
    # print(f'pre는 {pre} acc는 {acc}')  #, real_y는 {real_y}, pre는 {pre}')
    
    pred = sess.run(hypothesis, feed_dict={x:x_test}) # keras model.predict(x_test_data)
    pred = sess.run(tf.argmax(pred, 1)) # tf.argmax(a, 1) 안에 값들중에 가장 큰 값의 인덱스를 표시하라
    # pred = pred.reshape(-1,1)
    print(pred)

    y_test = sess.run(tf.argmax(y_test,1))
    print(y_test)

    acc = tf.reduce_mean(tf.compat.v1.cast(tf.equal(pred, y_test),tf.float32))
    acc = sess.run(acc)
    print(acc)