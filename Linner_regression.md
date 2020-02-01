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


##防止线性回归过拟合，增加惩罚系数
learningrate=0.001
training_epoch=100
reg_lambda=0.
x_ds=np.linspace(-1,1,100)
num_coeffs=9
y_ds_params=[0.]*num_coeffs
y_ds_params[2]=1
y_ds=0

for i in range(num_coeffs):
    y_ds+=y_ds_params[i]*np.power(x_ds,i)
    
y_ds+=np.random.randn(*x_ds.shape)*0.3

(x_train,x_test,y_train,y_test)=split_ds(x_ds,y_ds,0.7)

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

cost=tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)),tf.multiply(reg_lambda,tf.reduce_sum(tf.square(w)))),2*x_train.size)
#cost=tf.multiply(reg_lambda,tf.reduce_sum(tf.square(w)))
train_op=tf.train.GradientDescentOptimizer(learningrate).minimize(cost)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for reg_lambda in np.linspace(0,1,100):
    for epoch in range(training_epoch):
        sess.run(train_op,feed_dict={X:x_train,Y:y_train})
    w_val=sess.run(w)
    print(w_val)   
    final_cost=sess.run(cost,feed_dict={X:x_test,Y:y_test})
    print('reg lambda',reg_lambda)
    print('final cost',final_cost)
sess.close()


```
#### tensorflow 2.0版本线性回归
```python
#线性回归
n_examples=1000
training_steps=1000
display_step=100
learning_rate =0.01
m,c=6,-5
def train_data(n, m, c):
    x=tf.random.normal([n])
    noice=tf.random.normal([n])
    y=m*x+c+noice
    return x,y
def prediction(x,weight,bias):
    return weight*x+bias
def loss(x,y,weight,bias):
    error=prediction(x,weight,bias)-y
    squared_error=tf.square(error)
    return tf.reduce_mean(squared_error)
def grad(x,y,weight,bias):
    with tf.GradientTape() as tape:
        loss_=loss(x,y,weight,bias)
        return tape.gradient(loss_,[weight,bias])
#initiate data
from  matplotlib import pyplot as plt
import numpy as np
x, y = train_data(n_examples,m,c) 
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("figure 1 training data")
W=tf.Variable(np.random.randn())
B=tf.Variable(np.random.randn())
print("Initial loss: {:.3f}".format(loss(x, y, W, B)))
#train process
for step in range(training_steps):
    deltaW, deltaB = grad(x, y, W, B) # direction(sign) and value of the gradients of our loss 
    change_W = deltaW * learning_rate
    change_B = deltaB * learning_rate 
    W.assign_sub(change_W)
    B.assign_sub(change_B)
    if step==0 or step % display_step == 0:
        print(deltaW.numpy(), deltaB.numpy())
    print("Loss at step {:02d}: {:.6f}".format(step, loss(x, y, W, B)))
```
