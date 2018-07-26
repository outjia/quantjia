# coding=utf-8

import tensorflow as tf
import numpy as np

x = [[1, 2, 3], [4, 5, 6]]
y = np.arange(24).reshape([2, 3, 4])
z = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
sess = tf.Session()
begin_x = [1, 0]  # 第一个1，决定了从x的第二行[4,5,6]开始，第二个0，决定了从[4,5,6] 中的4开始抽取
size_x = [1, 2]  # 第一个1决定了，从第二行以起始位置抽取1行，也就是只抽取[4,5,6] 这一行，在这一行中从4开始抽取2个元素
out = tf.slice(x, begin_x, size_x)
print sess.run(out)  # 结果:[[4 5]]

begin_y = [1, 0, 0]
size_y = [1, 2, 3]
out = tf.slice(y, begin_y, size_y)
print sess.run(out)  # 结果:[[[12 13 14] [16 17 18]]]

begin_z = [0, 1, 1]
size_z = [-1, 1, 2]
out = tf.slice(z, begin_z, size_z)
print sess.run(out)  # size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取，结果：[[[ 5  6]] [[11 12]] [[17 18]]]

begin = np.zeros(len(z.shape), dtype=np.int)
begin[-1] = np.array(z.shape,dtype=np.int)[-1] - 1
idx = np.array(z.shape,dtype=np.int)
idx[-1] = 1

print sess.run(tf.slice(z,begin,idx))