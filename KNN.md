
#### Knn算法在iris数据集的例子

#### 1 引入数据集。
```python
from sklearn import datasets
iriis=datasets.load_iris()
x=iriis.data
y=iriis.target
```

#### 2. 数据集归一
```python
 x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))#按照列的最大值和最小值
```
#### 3.拆分训练集和测试集
```python
split = 0.8
round(len(x) * split)
```python
#80%作为训练集，20作为测试集。
train_indices = np.random.choice(len(x), round(len(x) * split), replace=False)

test_indices =np.array(list(set(range(len(x))) - set(train_indices)))
train_x = x[train_indices] 
test_x = x[test_indices] 
train_y = y[train_indices] 
test_y = y[test_indices]
```
####  4.knn 算法的思想

##### 4.1 计算测试集的每一个元素和训练集的每一个元素的L1距离。这里用到了广播。

##### 4.2 选取每一个元素的top_k元素的标签。

##### 4.3 根据真理在大多数人的手中的原则，确定测试集的标签。tf.math.bincount(item)可以计算出出现次数最多的元素。
```python
def prediction(train_x, test_x, train_y,k):
    #print(test_x)
    results=[]
    d0 = tf.expand_dims(test_x, axis =1) #30，1,4
    d1 = tf.subtract(train_x, d0)#train_x:120,4,d1:30,120,4
    d2 = tf.abs(d1)
    distances = tf.reduce_sum(input_tensor=d2, axis=2)
    _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    top_k_labels = tf.gather(train_y, top_k_indices)
    for item in top_k_labels:
        pred = tf.argmax(tf.math.bincount(item))
        results.append(pred)
    return results
    #print(distances)
    # or
    # distances = tf.reduce_sum(tf.abs(tf.subtract(train_x, tf.expand_dims(test_x, axis =1))), axis=2)
 ```
 
#### 5、测试集的结果
```python 
results1=prediction(train_x, test_x, train_y,k)
```
