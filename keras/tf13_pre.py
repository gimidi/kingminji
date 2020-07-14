# preprocessing

import tensorflow as tf
import numpy as np

def min_max_scaler(dataset) :
    numerator = dataset - np.min(dataset, 0)
    denominator = np.max(dataset, 0) - np.min(dataset, 0)
    return numerator / (denominator + 1e-7)

dataset = np.array(

    [

        [828.659973, 833.450012, 908100, 828.349976, 831.659973],

        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],

        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],

        [816, 820.958984, 1008100, 815.48999, 819.23999],

        [819.359985, 823, 1188100, 818.469971, 818.97998],

        [819, 823, 1198100, 816, 820.450012],

        [811.700012, 815.25, 1098100, 809.780029, 813.669983],

        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],

    ]

)

# print(min_max_scaler(dataset))
dataset = min_max_scaler(dataset)

x_data = dataset[:,0:-1]
y_data = dataset[:,[-1]]

print(x_data.shape)  # 8, 4
print(y_data.shape)  # 8, 1

x = tf.placeholder(tf.float32, shape=[None,x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([x_data.shape[1],1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(x,w) + b         

cost = -tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8)
model = optimizer.minimize(cost)

predicted = tf.cast(hypothesis, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
# sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001) :
        _, mse, weight, bias = sess.run([model, cost, w, b], feed_dict={x:x_data, y:y_data})
        if step % 500 == 0 :
            print('n은',step,' mse는',mse)#'  기울기는',weight,'  절편은', bias)

    h, pre, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print('acc는', acc)