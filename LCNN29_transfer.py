import os
import tensorflow as tf
import numpy as np
from hd5_reader import hd5_reader
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

BSIZE = 128
CLASS = 5031
EPOC = 20


#Using this to know name all tensor in graph
#print_tensors_in_checkpoint_file(file_name='Epoc_49_Iter_663.cpkt.meta', tensor_name='', all_tensors=True)

saver = tf.train.import_meta_graph('Epoc_49_Iter_663.cpkt.meta')


# Access the graph
#print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
graph = tf.get_default_graph()

## Prepare the feed_dict for feeding data for fine-tuning 
#for op in tf.get_default_graph().get_operations():
#    print(str(op.name))
#Access the appropriate output for fine-tuning
img_holder = graph.get_tensor_by_name('LCNN29/img_holder/Placeholder:0')
#lab_holder = graph.get_tensor_by_name('LCNN29/lab_holder/Placeholder:0')
Fcnn_512 = graph.get_tensor_by_name('LCNN29/Fcnn_17/Fcnn_17/MatMul:0')
lab_holder = tf.placeholder(tf.int64, [None, CLASS])
#use this if you only want to change gradients of the last layer
Fcnn_512 = tf.stop_gradient(Fcnn_512) # It's an identity function
Fcnn_512_shape = Fcnn_512.get_shape().as_list()
print(Fcnn_512_shape)

with tf.variable_scope("trainable_section"):
    #weights = tf.get_variable("weight", tf.truncated_normal([512, CLASS], stddev=0.05))
    weights = tf.get_variable('weight', [Fcnn_512_shape[1], CLASS], initializer=tf.contrib.layers.xavier_initializer())
    #biases = tf.get_variable(tf.constant(0.05, shape=[CLASS]))
    output = tf.matmul(Fcnn_512, weights) #+ biases
	correct = tf.equal(tf.cast(tf.argmax(output,1),tf.int64),tf.cast(tf.argmax(lab_holder,1),tf.int64))
	acc = tf.reduce_mean(tf.cast(correct,tf.float32))
	# #acc = tf.cast(correct,tf.float32)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lab_holder, logits=output))
	train = tf.train.AdamOptimizer(0.00002).minimize(loss)
trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trainable_section")

# 
model_path = 'model/'
log_path = 'log/'
list_train = 'hd5_list_train.txt'
list_val = 'hd5_list_val.txt'
#f_log = open('log/log.txt', 'a+')
#saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
	reader = hd5_reader(list_train, list_val, BSIZE, BSIZE)
	ITERS = reader.train_epoc; epoc = 0; iters = 0
	count = 0
	#writer = tf.summary.FileWriter(log_path, sess.graph)
	for j in range(EPOC):
		for i in range(ITERS):
			x_train, y_train_ = reader.train_nextbatch()
			global BSIZE
			BSIZE = reader.train_bsize
			# print ('BSIZE:', BSIZE)
			y_train = np.zeros([BSIZE,CLASS],dtype=np.int64)
			for index in range(BSIZE):
				y_train[index][y_train_[index]] = 1

			_, ls, ac = sess.run([train, loss, acc], feed_dict={img_holder:x_train, lab_holder:y_train})
			str1 =' Epoc: ' + str(j+epoc) + '\t|Iter: ' + str(i+iters) + '\t|Train_Loss: ' + str(ls) + '\t|Train_Acc: ' + str(ac) 
			print(str1)
			#f_log.write(str1 + '\n')

			#
			if count%1000 == 0:
				x_val, y_val_ = reader.val_nextbatch()
				global BSIZE
				BSIZE = reader.val_bsize
				y_val = np.zeros([BSIZE,CLASS],dtype=np.int64)
				for index in range(BSIZE):
					y_val[index][y_val_[index]] = 1

				ls_val = 0
				acc_val = 0
				for n in range(reader.val_data_ITERS):
					ls, ac = sess.run([loss, acc], feed_dict={img_holder:x_val, lab_holder:y_val})
					ls_val += ls
					acc_val += ac
				ls_val = ls_val/reader.val_data_ITERS
				acc_val = acc_val/reader.val_data_ITERS
				str1 =' Epoc: ' + str(j+epoc) + '\t|Iter: ' + str(i+iters) + '\t|Val_Loss: ' + str(ls_val) + '\t|Val_Acc: ' + str(acc_val) 
				print(str1)
				#f_log.write(str1 + '\n')
			count += 1

		save_path = model_path+'Transfer.cpkt'
		saver.save(sess, save_path)
	# writer.close()
	# writer = tf.summary.FileWrite(log_path, sess.graph)
	# merge = tf.summary.merge_all()
