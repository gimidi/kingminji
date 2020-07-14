import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

dataset = np.loadtxt('./data/csv/data-01-test-score.csv', delimiter=',',dtype=np.float32)
x_data = dataset[:,0:-1]
y_data = dataset[:,[-1]]
mapping={x:x_data, y:y_data}

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([x_data.shape[1],1]), dtype=tf.float32)  # ,dtype=tf.float32 이게 필수임...
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = tf.matmul(x,w) + b           
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
model = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(4001):
    mse, weight, bias, hy, _ = sess.run([cost, w, b, hypothesis, model], feed_dict=mapping)
    if step % 500 == 0:
        print(f'step:{step}, cost:{mse},  w[0]:{weight[0]},  bias:{bias},  f(x)[0]:{hy[0]}')