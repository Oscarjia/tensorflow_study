
#分类是一个很重要的任务：tensorflow 1.0版本
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
## softmax 函数
```python
##test data
xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
xs = np.vstack((xs_label0, xs_label1, xs_label2)) 
labels=np.matrix([[1., 0., 0.]]*len(xs_label0)+[[0., 1., 0.]] *len(xs_label1)+[[0., 0., 1.]] * len(x1_label2))
arr=np.arange(xs.shape[0])
np.random.shuffle(arr)
xs=xs[arr,:]
labels=labels[arr,:]
test_x1_label0 = np.random.normal(1, 1, (10, 1)) 
test_x2_label0 = np.random.normal(1, 1, (10, 1))
test_x1_label1 = np.random.normal(5, 1, (10, 1))
test_x2_label1 = np.random.normal(4, 1, (10, 1))
test_x1_label2 = np.random.normal(8, 1, (10, 1))
test_x2_label2 = np.random.normal(0, 1, (10, 1))
test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))
test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
test_labels = np.matrix([[1., 0., 0.]] * 10 + [[0., 1., 0.]] * 10+[[0., 0.,1.]] * 10)
train_size, num_features = xs.shape

##
learning_rate=0.01
training_epoch=100
num_labels=3
batch_size=100
X=tf.placeholder("float",shape=(None,num_features))
Y=tf.placeholder("float",shape=(None,num_labels))
w=tf.Variable(tf.zeros([num_features,num_labels]))
b=tf.Variable(tf.zeros([num_labels]))
y_model=tf.nn.softmax(tf.matmul(X,w)+b)
cost=-tf.reduce_sum(Y*tf.log(y_model))
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction=tf.equal(tf.argmax(y_model,1), tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(training_epoch*train_size//batch_size):
        offset = (step * batch_size) % train_size
        batch_xs=xs[offset:(offset+batch_size),:]
        batch_lables=labels[offset:(offset+batch_size)]
        err,_=sess.run([cost,train_op],feed_dict={X:batch_xs,Y:batch_lables})
        print(step,err)
    w_val=sess.run(w)
    print('w ',w_val)
    b_val=sess.run(b)
    print('b ',b_val)
    print("accuracy", accuracy.eval(feed_dict={X: test_xs, Y: test_labels}))

```
tensorflow 2.0对于分类模型
```python
tensorflow 在逻辑回归的例子

Input:

from tensorflow.keras.datasets import fashion_mnist

#初始化参数
batch_size = 128
epochs = 20
n_classes = 10
learning_rate = 0.1
width = 28 # of our imagesheight = 28 # of our images
height = 28
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255.

(x_train, y_train),(x_test, y_test)=fashion_mnist.load_data() #加载数据

x_train = x_train.reshape((60000, width * height)) #flatten data
x_test = x_test.reshape((10000, width * height))

split = 50000 
(x_train, x_valid) = x_train[:split], x_train[split:] 
(y_train, y_valid) = y_train[:split], y_train[split:]
y_train_ohe = tf.one_hot(y_train, depth=n_classes).numpy()#one hot 编码
y_valid_ohe = tf.one_hot(y_valid, depth=n_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=n_classes).numpy()
Model:

#定义模型。
class LogisticRegression(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense=tf.keras.layers.Dense(10)
    def call(self,inputs,training=None, mask=None):
        output = self.dense(inputs) 
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)
        return output
#compile 模型。
model = LogisticRegression(n_classes)
optimiser =tf.keras.optimizers.Adam()
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'], )
Train model:

#
dummy_x = tf.zeros((1, width * height))
model.call(dummy_x)
#保存训练好的模型参数。
from tensorflow.keras.callbacks import ModelCheckpoint
checkpointer=ModelCheckpoint(filepath="./model.weights.best.hdf5", verbose=2, save_best_only=True, save_weights_only=True)
model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,validation_data=(x_valid, y_valid_ohe), callbacks=[checkpointer], verbose=2)
#加载最好的模型参数。
model.load_weights("./model.weights.best.hdf5")
#评估模型参数。
scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=2)
#还原预测的结果。
index_predicted = np.argmax(y_predictions[index]) 


如何利用保存的参数来实例化一个model:

model2 = LogisticRegression(n_classes) #实例化模型对象
model2.load_weights("./model.weights.best.hdf5")#加载参数
optimiser =tf.keras.optimizers.Adam()#模型
#实例化模型
model2.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'], )
#评估模型
scores = model2.evaluate(x_test, y_test_ohe, batch_size, verbose=2)


```
