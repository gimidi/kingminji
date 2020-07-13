import tensorflow as tf
import numpy as np

# data = hihello

idx2char = ['e','h','i','l','o']

_data = np.array([['h','i','h','e','l','l','o']]).T
print(_data.shape) # (7, 1)
print(_data)
print(type(_data)) 

from sklearn.preprocessing import OneHotEncoder # 문자도 된다니

enc = OneHotEncoder()
enc.fit(_data)  # 실행 
_data = enc.transform(_data).toarray()

print('==============================================')
print(_data)
print(type(_data))

x_data = _data[:6,] # hihell
y_data = _data[1:,] #  ihello

print('==============================================')
print(x_data)
print('==============================================')
print(y_data)

y_data = np.argmax(y_data, axis=1)
print('==============================================')
print(y_data)
print(y_data.shape) # (6,)
# 텐서플로에서는 shape를 1,6으로 바꿔줘야 한다 (이유는 아직 설명 x)

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)
print(x_data.shape)
print(y_data.shape)

# shape를 변수로 넣어놓자
sequence_length = 6
input_dim = 5
output = 100
batch_size = 1
# X = tf.placeholder(tf.float32,(None,sequence_length,input_dim))
# Y = tf.placeholder(tf.float32,(None,sequence_length))
X = tf.compat.v1.placeholder(tf.float32,(None,sequence_length,input_dim)) # 3
# Y = tf.compat.v1.placeholder(tf.float32,(None,sequence_length))
Y = tf.compat.v1.placeholder(tf.int64,(None,sequence_length)) # 2

print(X)
print(Y)

# 모델 구성

# model.add(LSTM(output, input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(hypothesis) # (?, 6, 5)
weights = tf.ones([batch_size,sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, targets=Y, weights=weights
) # 선형을 디폴트로 하겠다
loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0000001).minimize(loss)

prediction = tf.argmax(hypothesis, axis=2) # 3차원이니까 axis=2

# 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401) :
        loss_1, _ = sess.run([loss, train], feed_dict={X:x_data,Y:y_data})
        result = sess.run(prediction,feed_dict={X:x_data} )
        print(i, 'loss:',loss_1,'prediction:',result, 'true_Y:', y_data)

    
    result_str = [ idx2char[c] for c in np.squeeze(result)]
    print('\tpridiction str:', ''.join(result_str))
