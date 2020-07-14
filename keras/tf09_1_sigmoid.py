import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]
y_data = [[0.],
          [0.],
          [0.],
          [1.],
          [1.],
          [1.]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])


w = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = tf.reduce_mean(y* tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))  # sigmoid에 대한 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
model = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, mse, weight, bias = sess.run([model, cost, w, b], feed_dict={x:x_data, y:y_data})
    if step % 100 == 0:
        print('mse는',mse,'  기울기는',weight,'  절편은', bias)

h, pre, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
print('h는',h,'  pre는',pre,'  acc는', acc)
