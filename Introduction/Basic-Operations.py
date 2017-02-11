import tensorflow as tf 

#basic constant operations
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)


#basic operations with variable as graph input
test1 = tf.placeholder(tf.int32)
test2 = tf.placeholder(tf.int32)
add = tf.add(test1, test2)
mul = tf.mul(test1, test2)
add_mul = tf.mul((tf.add(a,b)), test2)

#matrix multiplication 
matrix1 = tf.constant([[3., 3.]]) #1x2 matrix
matrix2 = tf.constant([[2.],[2.]]) #2x1 matrix
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
	print "a=2, b=3, c=4"
	print "%i" % sess.run(a+b+c)
	print "%i" % sess.run(a*b+c)

	print "%i" % sess.run(add, feed_dict={test1: 5, test2: 5}) #feed_dict
	print "%i" % sess.run(mul, feed_dict={test1: 5, test2: 5})
	print "%i" % sess.run(add_mul, feed_dict={test2: 10})
	print "%i" % sess.run(product)






