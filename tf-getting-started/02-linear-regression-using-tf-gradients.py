import tensorflow as tf

### linear regression using tf.gradients
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

#print(sess.run(linear_model, feed_dict={x:[1.,2.,3.,4.]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
dW, db = tf.gradients(loss, [W, b])
learning_rate = 0.02
fixW = tf.assign(W, W - learning_rate * dW)
fixb = tf.assign(b, b - learning_rate * db)
train = [fixW, fixb]

for step in range(2000):

    cur_train, cur_loss = sess.run([train,loss],feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]} )

    if step % 10 == 0:
        print(step, cur_loss, sess.run(W), sess.run(b))


