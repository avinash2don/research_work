
# coding: utf-8

# In[1]:


import os
import math
from glob import glob
import matplotlib.pyplot as plt
import random
import collections
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import visualizations as vis
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
le = preprocessing.LabelEncoder()
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE


# In[2]:


filename = '/Users/avinash2don/Dropbox/Crime-Prediction/Final Code ( R and Python)/Charges_df.csv'


# In[3]:


data = pd.read_csv(filename)
display(data.head())


# In[4]:


df=data[['key','feature1']]
display(df.head())


# In[5]:


dummy_df=pd.get_dummies(df['feature1'])
display(dummy_df.head(4))
df=pd.concat([df,dummy_df],axis=1)
display(df.head(4))
df=df.drop(['feature1'],axis=1)
display(df)


# In[6]:


XF=df.groupby(["key"],as_index=False).sum()
XF


# In[7]:


data=pd.read_csv('svm_data.csv',index_col=0)
data= data.dropna(axis=0, how='any')
lable = data['label1'].unique()
L=len(lable)


# In[8]:


display(data.head())
display(data.shape)


# In[9]:


dfinal = pd.merge(data, XF, on="key")


# In[10]:


dfinal= dfinal.dropna(axis=0, how='any')


# In[11]:


dfinal


# In[12]:


display(dfinal.groupby(["label1"]).count())
display(dfinal.groupby(["label2"]).count())
display(dfinal.groupby(["label3"]).count())


# In[13]:


dfinal.isnull().values.any()


# In[14]:


def smo(X_train, y_train):
    sm = SMOTE(ratio='minority',random_state=12,k_neighbors=6,m_neighbors=12,kind='svm')
    X_train, y_train = sm.fit_sample(X_train, y_train)
    return X_train, y_train
def one_hot_encode(np_array,i):
    return (np.arange(i) == np_array[:,None]).astype(np.float32)
class myarray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(myarray)
    def index(self, value):
        return np.where(self == value)


# In[15]:


def train_test(df,lable_col):
#     le.fit(data['feature2']) 
#     df['feature2']=le.transform(df['feature2'])
    df[['feature3', 'feature4','feature5','feature6','feature7']] = scaler.fit_transform(df[['feature3', 'feature4','feature5','feature6','feature7']])
    dataR=df.pop('feature2')
    dummy_df = pd.get_dummies(dataR)
#     display(dummy_df)
    df=pd.concat([df, dummy_df],axis=1)
#     display(df)
    x=df.drop(["key","label1","label2","label3"],axis=1)
    y=df[lable_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.1)
    x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train,stratify=y_train, test_size=0.1)
    display(x_train1.shape)
    display(y_train1.shape)
    display(y_train1.value_counts())
    #x_train1, y_train1=smo(x_train1, y_train1)
    display(collections.Counter(y_train1))
    return x_train1, x_test, x_val, y_train1, y_test, y_val


# In[16]:


lable_col="label1"
x_train, x_test, x_val, y_train, y_test, y_val=train_test(dfinal,lable_col)


# In[17]:


# x_train=x_train.values
# x_test=x_test.values
# y_test=y_test.values
# x_val=x_val.values
# y_val=y_val.values
# y_test=one_hot_encode(y_test,L)
# y_train=one_hot_encode(y_train,L)
# y_val=one_hot_encode(y_val,L)
display(x_train.shape)
display(x_test.shape)
display(x_val.shape)
display(collections.Counter(y_train))
display(y_train.shape)
display(y_test.shape)
display(y_val.shape)

display(x_train.shape)
display(y_train.shape)
display(y_train.value_counts())
a=np.array(x_train)
b=myarray(y_train)
idx=b.index(1)
print(idx)
x_train2 = np.delete(a, idx,axis=0)
y_train2 = np.delete(b, idx)
x_minor = a[idx]
y_minor = b[idx]
#x_train1, y_train1=smo(x_train1, y_train1)
display(x_train2.shape)
display(y_train2.shape)
display(collections.Counter(y_train2))
display(x_minor.shape)
display(y_minor.shape)
display(collections.Counter(y_minor))

label_test=one_hot_encode(y_test,L)
label_train2=one_hot_encode(y_train2,L)
label_val=one_hot_encode(y_val,L)
label_minor=one_hot_encode(y_minor,L)

display(x_train2.shape)
display(label_train2.shape)
display(x_minor.shape)
display(label_minor.shape)
display(x_test.shape)
display(label_test.shape)
display(x_val.shape)
display(label_val.shape)


# In[18]:


x_test=np.array(x_test)
label_test=np.array(label_test)
x_val=np.array(x_val)
label_val=np.array(label_val)


# In[19]:


size_flat = 57

num_classes = 2

# Fully-connected layer.
fc_size = 512 

batch_size = 700

early_stopping=None


# In[20]:


X = tf.placeholder(tf.float32, [None, size_flat])

XX = tf.reshape(X, [-1, size_flat])

# correct answers will go here
y_true = tf.placeholder(tf.float32, [None, num_classes])

y_true_cls = tf.argmax(y_true, dimension=1)


# In[21]:


def new_fc_layer(input,          
                 num_inputs,     
                 num_outputs,    
                 use_relu=True): 

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[22]:


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# In[23]:


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[24]:


layer_fc1 = new_fc_layer(input=XX,
                         num_inputs=size_flat,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc3 = new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc4 = new_fc_layer(input=layer_fc3,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc5 = new_fc_layer(input=layer_fc4,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


# In[25]:


y_pred = tf.nn.softmax(layer_fc5)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc5,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[26]:


train_batch_size = batch_size


# In[27]:


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    
def random_batch(x_train, y_train):
    
    i=train_batch_size-len(y_minor)
    num = len(x_train)
    idx = np.random.choice(num,
                           size=i,
                           replace=False)
    x_batch = x_train[idx]
    y_batch = y_train[idx]
#     print(len(y_batch))
    x_batch2=np.vstack((x_batch,x_minor))
    y_batch2=np.vstack((y_batch,label_minor))
#     print(len(y_batch2))

    return x_batch2, y_batch2


# In[28]:


x,y=random_batch(x_train2,label_train2)
print(x.shape)
print(y.shape)


# In[29]:


total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # random batch of training validation samples.
        x_batch, y_true_batch = random_batch(x_train2, label_train2)
        x_valid_batch, y_valid_batch =x_val, label_val
        
        val_size=len(label_val)

        x_batch = x_batch.reshape(train_batch_size, size_flat)
        x_valid_batch = x_valid_batch.reshape(val_size, size_flat)

        feed_dict_train = {X: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {X: x_valid_batch,
                              y_true: y_valid_batch}

        # Optimizer using the random batch
        session.run(optimizer, feed_dict=feed_dict_train)
        

        # Print status at end of each epoch.
        if i % int(len(x_train)/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(len(x_train)/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    total_iterations += num_iterations

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


# In[30]:


def plot_confusion_matrix(cls_pred):
    cls_true = y_val
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[31]:


def plot_confusion_matrix1(cls_pred):
    cls_true = y_test
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[32]:




saver = tf.train.Saver(max_to_keep=100)

save_dir = './Ensemble/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)


# In[33]:


session = tf.Session()


# In[34]:


session.run(tf.global_variables_initializer())


# In[35]:


def print_validation_accuracy(show_confusion_matrix=False):

    num_test = len(x_val)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)


    i = 0

    while i < num_test:
        j = min(i + num_test, num_test)

        images = x_val[i:j].reshape(num_test, size_flat)
        
        labels = label_val[i:j]

        feed_dict = {X: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j


    print(cls_pred)
    cls_true = np.array(y_val)
    print(cls_true)

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    
    msg = "Accuracy on validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# In[36]:


def print_test_accuracy(show_confusion_matrix=False):

    num_test = len(x_test)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0

    while i < num_test:
        j = min(i + num_test, num_test)
        
        images = x_test[i:j].reshape(num_test, size_flat)
        
        labels = label_test[i:j]

        feed_dict = {X: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = np.array(y_test)

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test


    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix1(cls_pred=cls_pred)


# In[37]:


num_networks = 5


# In[38]:


iterations = 1000


# In[39]:


if True:
    # For each of the neural networks.
    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        optimize(num_iterations=iterations)
        
        print_test_accuracy(show_confusion_matrix=True)
        
        print_validation_accuracy(show_confusion_matrix=True)

        # Save the optimized variables to disk.
        saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        print()

