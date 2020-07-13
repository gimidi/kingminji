import tensorflow as tf
import numpy as np

dataset = [1,2,3,4,5,6,7,8,9,10]
# print(dataset.shape)

# RNN 모델을 짜시오

x_data = []
y_data = []
for i in range(5) :
    x_data.append(dataset[i:i+5])
    y_data.append(dataset[i+5])

print('x_data :',x_data)
print('y_data :',y_data)

x_data = np.array(x_data , dtype=np.float32)
y_data = np.array(y_data , dtype=np.float32)
print(x_data.shape)
print(y_data.shape)

x_data = x_data.reshape(1,5,5)
y_data = y_data.reshape(1,5)
print(x_data.shape)
print(y_data.shape)

print('x_data :',x_data)
print('y_data :',y_data)


sequence_length = 5
input_dim = 5
output = 100
batch_size = 1

X = tf.compat.v1.placeholder(tf.float32,(None,sequence_length,input_dim)) # 3차원
Y = tf.compat.v1.placeholder(tf.int32,(None,sequence_length)) # 2차원
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(hypothesis) # (?, 6, 5)

# compile
weights = tf.ones([batch_size,sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, targets=Y, weights=weights
) # 선형을 디폴트로 하겠다
loss = tf.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.000001).minimize(loss)


# 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401) :
        loss_1, _, result = sess.run([loss, train,hypothesis], feed_dict={X:x_data,Y:y_data})
        # result = sess.run(hypothesis, feed_dict={X:x_data} )
        print(i, 'loss :',loss_1,'result :',result, 'true_Y :', y_data)

    
    # result_str = [ idx2char[c] for c in np.squeeze(result)]
    # print('\tpridiction str:', ''.join(result_str))
