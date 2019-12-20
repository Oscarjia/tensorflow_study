#### tensorflow 1.X与2.X差别还是挺大的。主要对tensorflow如何处理线性回归问题进行学习
```python
#线性回归tensorflow1版本
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
learning_rate=0.01
training_epochs = 100
X=tf.placeholder(tf.float32) #用于feed dict
Y=tf.placeholder(tf.float32 )#用于feed dict
def model(X,w):              #定义一个模型
    return tf.multiply(X,w)
w=tf.Variable(0.0,name="weights") # 权重初始化为0
y_model=model(X,w)
cost=tf.square(Y-y_model)         #定义一个cost函数
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #优化方法，梯度下降
sess=tf.Session()
init=tf.global_variables_initializer()  
sess.run(init) #初始化全局变量
for epoch in range(training_epochs): #全部的数据需要处理几遍
    for x,y in zip(x_train,y_train):
        sess.run(train_op,feed_dict={X:x,Y:y})
w_val=sess.run(w)
sess.close()
plt.scatter(x_train,y_train)
y_learned=w_val*x_train
plt.plot(x_train,y_learned,c='r')
plt.show()
````
