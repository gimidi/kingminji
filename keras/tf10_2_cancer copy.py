from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
tf.set_random_seed(777)

x_data, y_data = load_breast_cancer(return_X_y=True)
y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

x = tf.placeholder(tf.float32, shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.zeros([30,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = -tf.reduce_mean(y* tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))  # sigmoid에 대한 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5e-6)
model = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# sess = tf.Session()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(3001) :
        _, mse, weight, bias = sess.run([model, cost, w, b], feed_dict={x:x_train, y:y_train})
        if step % 500 == 0 :
            print('step은',step,' mse는',mse)#'  기울기는',weight,'  절편은', bias)

    h, pre, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_test, y:y_test})
    print('acc는', acc)
# sess.close()