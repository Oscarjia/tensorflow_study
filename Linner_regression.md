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

tf处理多参数线性回归
```python
###定义曲线
#多元函数模型
learning_rate=0.01
training_epoch=40
x_train=np.linspace(-1,1,101)
num_coeffs=6
y_coeffs=[1,2,3,4,5,6]
y_train=0
for i in range(num_coeffs):
    y_train+=y_coeffs[i]*np.power(x_train,i)
y_train+=np.random.randn(*x_train.shape)*1.5
plt.scatter(x_train,y_train)
plt.show()


##定义模型
learning_rate=0.01
training_epochs = 100
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
def model(X,w):
    terms=[]
    for i in range(num_coeffs):
        term=tf.multiply(w[i],tf.pow(X,i))
        terms.append(term)
    return tf.add_n(terms)
w=tf.Variable([0.]*num_coeffs,name="parameters")
y_model=model(X,w)
cost=tf.pow(Y-y_model,2)
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for epoch in range(training_epochs):
    for (x,y) in zip(x_train,y_train):
        sess.run(train_op,feed_dict={X:x,Y:y})
w_val=sess.run(w)
print(w_val)
sess.close()
plt.scatter(x_train,y_train)

###根据X的值来预测Y的值。
y_train2=0  #根据训练的参数，预测Y的值
for i in range(num_coeffs):
    y_train2+=w_val[i]*np.power(x_train,i)
plt.plot(x_train,y_train2)
plt.show()

##手写分数据集的方法
## 分数据集的方法
def split_ds(x_ds,y_ds,ratio):
    arr=np.arange(x_ds.size)
    np.random.shuffle(arr)
    num_train=int(ratio*x_ds.size)
    x_train=x_ds[arr[0:num_train]]
    x_test=x_ds[arr[num_train:x_ds.size]]
    y_train=y_ds[arr[0:num_train]]
    y_test=y_ds[arr[num_train:y_ds.size]]
    return x_train,x_test,y_train,y_test





```
