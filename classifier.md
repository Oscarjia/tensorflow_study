
#分类是一个很重要的任务
#线性回归可以用作分类，缺点是对异常值敏感，不准确。
```python

x_label0=np.random.normal(5,1,10)
x_label1=np.random.normal(2,1,10)
xs=np.append(x_label0,x_label1)
labels=[0.]*len(x_label0)+[1.]*len(x_label1)
learning_rate=0.001
training_epoch=100
X=tf.placeholder("float")
Y=tf.placeholder("float")

def model(X,w):
    return tf.add(tf.multiply(w[1],tf.pow(X,1)),tf.multiply(w[0],tf.pow(X,0)))

w=tf.Variable([0.,0.],name="parameters")
y_model=model(X,w)
cost=tf.reduce_sum(tf.square(Y-y_model))
#correct_prediction=tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5)))
correct_prediction=tf.equal(Y,tf.to_float(tf.greater(y_model,0.5)))

accuracy=tf.reduce_mean(tf.to_float(correct_prediction))
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for epoch in range(training_epoch):
    sess.run(train_op,feed_dict={X:xs,Y:labels})
    current_cost=sess.run(cost,feed_dict={X:xs,Y:labels})
    print(epoch,current_cost)
    print('accuracy ',sess.run(accuracy,feed_dict={X:xs,Y:labels}))
    w_val=sess.run(w)
    print('learned parameters ',w_val)
    all_xs=np.linspace(0,10,100)
    plt.plot(all_xs,all_xs*w_val[1]+w_val[0])
    plt.show()
sess.close()

```
#逻辑回归在分类的利用
``` python
#logistic regression
learning_rate=0.001
training_epoch=100
X=tf.placeholder("float")
Y=tf.placeholder("float")
def sigmoid(x):
    return 1./(1.+np.exp(-x))
x1=np.random.normal(-4,2,1000)
x2=np.random.normal(4,2,1000)
xs=np.append(x1,x2)
ys=np.asarray([0.]*len(x1)+[1.]*len(x2))
plt.scatter(xs,ys)
X=tf.placeholder(tf.float32,shape=(None,),name="x")
Y=tf.placeholder(tf.float32,shape=(None,),name="y")
w=tf.Variable([0.,0.],name="parameter",trainable=True)
y_model=tf.sigmoid(w[1]*X+w[0])
cost=tf.reduce_mean(-Y*tf.log(y_model)-(1-Y)*tf.log(1-y_model))
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre_err=0
    for epoch in range(training_epoch):
        err,_,ws=sess.run([cost,train_op,w],feed_dict={X:xs,Y:ys})
        print("epoch is {},err is {},w is {}".format(epoch,err,ws))
        if abs(pre_err-err)<0.0001:
            break
        pre_err=err
    w_val=sess.run(w,{X:xs,Y:ys})
    print("w_val is {}".format(w_val))

```
