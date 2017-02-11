import tensorflow as tf 

test = tf.constant('Hello, World!')

sess = tf.Session()

print sess.run(test)