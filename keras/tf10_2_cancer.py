from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np

dataset = load_breast_cancer()
x_data = dataset['data'].astype(np.float32)
y_data = dataset['target'].astype(np.float32)
y_data = y_data.reshape(-1,1)

x = tf.placeholder(tf.float32, shape=[None,x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([x_data.shape[1],1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = tf.reduce_mean(y* tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))  # sigmoid에 대한 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
model = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100001) :
    _, mse, weight, bias = sess.run([model, cost, w, b], feed_dict={x:x_data, y:y_data})
    if step % 500 == 0 :
        print('step은',step,' mse는',mse)#'  기울기는',weight,'  절편은', bias)

h, pre, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
print('acc는', acc)