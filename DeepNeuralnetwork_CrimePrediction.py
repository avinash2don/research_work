
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import visualizations1 as vis
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
le = preprocessing.LabelEncoder()
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

from sklearn.metrics import precision_recall_fscore_support

F1=precision_recall_fscore_support

from sklearn.metrics import confusion_matrix

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     stratify=y, 
#                                                     test_size=0.25)


# In[2]:


filename = 'Charges_df.csv'
df=pd.read_csv('svm_data.csv',index_col=0)
df= df.dropna(axis=0, how='any')
lable = df['label1'].unique()
L=len(lable)
df2 = pd.read_csv(filename)
df2=df2[['key','feature1']]


# In[3]:


display(df.head())
display(df2.head())
display(collections.Counter(df2['feature1']))


# In[4]:


dummy_df=pd.get_dummies(df2['feature1'])
df2=pd.concat([df2,dummy_df],axis=1)
df2=df2.drop(['feature1'],axis=1)
display(df.head())
df2=df2.groupby(["key"],as_index=False).sum()
dfinal = pd.merge(df, df2, on="key")
dfinal= dfinal.dropna(axis=0, how='any')


# In[5]:


def smo(X_train, y_train):
    sm = SMOTE(ratio={0:20000, 1:5000},random_state=12,k_neighbors=1,m_neighbors=2)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    return X_train, y_train
def one_hot_encode(np_array,i):
    return (np.arange(i) == np_array[:,None]).astype(np.float32)


# In[6]:


def train_test(df,lable_col):
#     le.fit(data['feature2']) 
#     df['feature2']=le.transform(df['feature2'])

    df[['feature3', 'feature4','feature5','feature6','feature7']] = scaler.fit_transform(df[['feature3', 'feature4','feature5','feature6','feature7']])
    dataR=df[['feature2']]
    dummy_df = pd.get_dummies(dataR)
    display(dummy_df)
    df=pd.concat([df, dummy_df],axis=1)
    display(df)
    x=df.drop(["key","label1","label2","label3",'feature2'],axis=1)
    y=df[lable_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y,random_state=9, test_size=0.1)
    x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train,stratify=y_train,random_state=10, test_size=0.1)
    display(x_train1.shape)
    display(y_train1.value_counts())
#     x_train1, y_train1=smo(x_train1, y_train1)
    display(x_train1.shape)
    display(collections.Counter(y_train1))
    return x_train1, x_test, x_val, y_train1, y_test, y_val


# In[7]:


lable_col="label1"
x_train, x_test, x_val, y_train, y_test, y_val=train_test(dfinal,lable_col)


# In[8]:


display(collections.Counter(y_train))
display(collections.Counter(y_test))
display(collections.Counter(y_val))


# In[9]:


x_trainOG=x_train.copy()
y_trainOG=y_train.copy()


# In[10]:


x_train, y_train=smo(x_train, y_train)


# In[11]:


display(collections.Counter(y_train))
display(collections.Counter(y_test))
display(collections.Counter(y_val))


# In[12]:


x_test=x_test.values
y_test=y_test.values
x_val=x_val.values
y_val=y_val.values
y_test=one_hot_encode(y_test,L)
y_train=one_hot_encode(y_train,L)
y_val=one_hot_encode(y_val,L)
y_trainOG=one_hot_encode(y_trainOG,L)
display(x_train.shape)
display(x_test.shape)
display(x_val.shape)
display(y_train.shape)
display(y_test.shape)
display(y_val.shape)


# In[13]:


def f1(cm):
    tn = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tp = cm[1][1]
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f=2*p*r/(p+r)
    return f


# In[14]:


NUM_ITERS=100000
DISPLAY_STEP=1
BATCH=13000
beta=0.001
learning_rate = 0.01

tf.set_random_seed(0)

X = tf.placeholder(tf.float32, [None, 57])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 2])


# layers sizes
L1 = 128
L2 = 64
L3 = 32
L4 = 32
L5 = 2
# L6 = 16
# L7 = 2
# L8 = 16
# L9 = 8
# L10 = 2



# weights - initialized with random values from normal distribution mean=0, stddev=0.1
# output of one layer is input for the next
W1 = tf.Variable(tf.truncated_normal([57, L1], stddev=0.1))
b1 = tf.Variable(tf.zeros([L1]))

W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
b2 = tf.Variable(tf.zeros([L2]))

W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
b3 = tf.Variable(tf.zeros([L3]))

W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
b4 = tf.Variable(tf.zeros([L4]))

W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
b5 = tf.Variable(tf.zeros([L5]))

# W6 = tf.Variable(tf.truncated_normal([L5, L6], stddev=0.1))
# b6 = tf.Variable(tf.zeros([L6]))

# W7 = tf.Variable(tf.truncated_normal([L6, L7], stddev=0.1))
# b7 = tf.Variable(tf.zeros([L7]))

# W8 = tf.Variable(tf.truncated_normal([L7, L8], stddev=0.1))
# b8 = tf.Variable(tf.zeros([L8]))

# W9 = tf.Variable(tf.truncated_normal([L8, L9], stddev=0.1))
# b9 = tf.Variable(tf.zeros([L9]))

# W10 = tf.Variable(tf.truncated_normal([L9, L10], stddev=0.1))
# b10 = tf.Variable(tf.zeros([L10]))




# -1 in the shape definition means compute automatically the size of this dimension
XX = tf.reshape(X, [-1, 57])

# Define model
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + b1)
# Y1 = tf.nn.dropout(Y1,keep_prob)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
# Y2 = tf.nn.dropout(Y2,keep_prob)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + b3)
# Y3 = tf.nn.dropout(Y3,keep_prob)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + b4)
# Y4 = tf.nn.dropout(Y4,keep_prob)
# Y5 = tf.nn.sigmoid(tf.matmul(Y4, W5) + b5)
# Y6 = tf.nn.sigmoid(tf.matmul(Y5, W6) + b6)
# Y7 = tf.nn.sigmoid(tf.matmul(Y6, W7) + b7)
# Y8 = tf.nn.sigmoid(tf.matmul(Y7, W8) + b8)
# Y9 = tf.nn.sigmoid(tf.matmul(Y8, W9) + b9)

Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)


correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))


# we can also use tensorflow function for softmax
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)

cross_entropy = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=Ylogits, labels=Y_) +
    beta*tf.nn.l2_loss(W1) +
    beta*tf.nn.l2_loss(b1) +
    beta*tf.nn.l2_loss(W2) +
    beta*tf.nn.l2_loss(b2) +
    beta*tf.nn.l2_loss(W3) +
    beta*tf.nn.l2_loss(b3) +
    beta*tf.nn.l2_loss(W4) +
    beta*tf.nn.l2_loss(b4) +
    beta*tf.nn.l2_loss(W5) +
    beta*tf.nn.l2_loss(b5)))

# cross_entropy = tf.reduce_mean(cross_entropy)*100


                                                          
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


confusion_matrix_tf = tf.confusion_matrix(tf.argmax(Y, 1), tf.argmax(Y_, 1))



# _r,re=tf.metrics.recall(tf.argmax(Y, 1), tf.argmax(Y_, 1))
# _r1,re1=tf.metrics.recall(tf.argmax(Y1, 1), tf.argmax(Y1_, 1))

# _p,pr=tf.metrics.precision(tf.argmax(Y, 1), tf.argmax(Y_, 1))
# _p1,pr1=tf.metrics.precision(tf.argmax(Y1, 1), tf.argmax(Y1_, 1))

# confusion_matrix_final = tf.confusion_matrix(tf.argmax(Y1, 1), tf.argmax(Y_, 1))

# training, learning rate = 0.005
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)



# Initializing the variables
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()



OG_losses = list()
OG_acc = list()
OG_F1 = list()


train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()
train_F1 = list()
test_F1 = list()


# saver = tf.train.Saver()



sess = tf.Session()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    sess.run(init_l)
    
    
    
    for i in tqdm(range(1,NUM_ITERS+2)):
        idx = np.random.choice(len(x_train), BATCH, replace=True)
        X_batch = x_train[idx, :]
#         print(X_batch)
        y_batch = y_train[idx]
#         X_batch=np.vstack((X_,x_minor12))
#         y_batch=np.vstack((y_,y_minor12))
#         print(y_batch)
        
        if i%DISPLAY_STEP ==0:
           
            n=i/DISPLAY_STEP
            print(n)
            # compute training values for visualisation
            acc_trn, loss_trn = sess.run([accuracy, cross_entropy], feed_dict={X: X_batch, Y_: y_batch})
            
            acc_OG, loss_OG = sess.run([accuracy, cross_entropy], feed_dict={X: x_trainOG, Y_: y_trainOG})
            
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: x_test, Y_: y_test})
            
            cm_train = confusion_matrix_tf.eval(feed_dict={X: X_batch, Y_: y_batch})
            
            cm_test = confusion_matrix_tf.eval(feed_dict={X: x_test, Y_: y_test})
            
            cm_OG = confusion_matrix_tf.eval(feed_dict={X: x_trainOG, Y_: y_trainOG})
            
            
            
            
#             TN = cm_train[0][0]
#             FP = cm_train[0][1]
#             FN = cm_train[1][0]
#             TP = cm_train[1][1]
            
            
#             TN_t = cm_test[0][0]
#             FP_t = cm_test[0][1]
#             FN_t = cm_test[1][0]
#             TP_t = cm_test[1][1]
            
            f1_train=f1(cm_train)
            f1_test=f1(cm_test)
            f1_OG=f1(cm_OG)
#             precision=tf.summary.scalar(pr)
#             recall=tf.summary.scalar(re)
#             precision=tf.summary.scalar(pr)
#             recall=tf.summary.scalar(re)
            
#             recall =sess.run(re, feed_dict={X: X_batch, Y_: y_batch})
#             precision =sess.run(pr, feed_dict={X: X_batch, Y_: y_batch})
            
#             recall_test =sess.run(re, feed_dict={X: x_test, Y_: y_test})
#             precision_test =sess.run(pr, feed_dict={X: x_test, Y_: y_test})

            
#             f1_train = 2 * precision * recall / (precision + recall)
            
#             f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)
            
            print("#{} Trn acc={}  Trn F1={} Trn loss={} , OG acc={}  OG F1={} OG loss={} ,Tst acc={} Tst F1={} Tst loss={}".format(i,acc_trn,f1_train,loss_trn,acc_OG,f1_OG,loss_OG,acc_tst,f1_test,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            train_F1.append(f1_train)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)
            test_F1.append(f1_test)
            OG_losses.append(loss_OG)
            OG_acc.append(acc_OG)
            OG_F1.append(f1_OG)
            #learning_rate=learning_rate/100
            #print(learning_rate)
            

        # the backpropagationn training step
#         if n>=1:
#             learning_rate = learning_rate/10
#             sess.run(train_step, feed_dict={X: X_batch, Y_: y_batch})
#             print(learning_rate)
#         else:
#             sess.run(train_step, feed_dict={X: X_batch, Y_: y_batch})
        
        sess.run(train_step, feed_dict={X: X_batch, Y_: y_batch})
#         print(learning_rate)

#     prediction=tf.argmax(Y_, 1)
#     output= sess.run(prediction,feed_dict={X: x_test})
#     print(output)
#     print ("predictions", prediction.eval(feed_dict={X: x_test}, session=sess))
#     output = sess.run(y, feed_dict={x :input})
    


    print ("accuracy_SMO", sess.run(accuracy, feed_dict={X: x_train, Y_: y_train}))
    cmOG=confusion_matrix_tf.eval(feed_dict={X: x_train, Y_: y_train})
    print(cmOG)
    print ("F1-Score_SMO", f1(cmOG))
    
    print ("accuracy_OG", sess.run(accuracy, feed_dict={X: x_trainOG, Y_: y_trainOG}))
    cmOG=confusion_matrix_tf.eval(feed_dict={X: x_trainOG, Y_: y_trainOG})
    print(cmOG)
    print ("F1-Score_OG", f1(cmOG))
        
    print ("accuracy", sess.run(accuracy, feed_dict={X: x_test, Y_: y_test}))
    cm = confusion_matrix_tf.eval(feed_dict={X: x_test, Y_: y_test})
    print(cm)
    print ("F1-Score", f1(cm))
    
    print ("accuracy_val", sess.run(accuracy, feed_dict={X: x_val, Y_: y_val}))
    cm = confusion_matrix_tf.eval(feed_dict={X: x_val, Y_: y_val})
    print(cm)
    print ("F1-Score_val", f1(cm))
    
    
#     pred_12 = Y.eval(feed_dict={X: x_test}, session=sess)
#     pred_12_val = Y.eval(feed_dict={X: x_val}, session=sess)
#    pred1_12 = Y1.eval(feed_dict={X1: x_test}, session=sess)
#    pred_23 = Y.eval(feed_dict={X: x_test1}, session=sess)
#     prediction=tf.argmax(Y_, 1)
#     output= sess.run(prediction,feed_dict={X: x_test})
#     print(output)
#     print ("predictions", prediction.eval(feed_dict={X: x_test}, session=sess))
#     output = sess.run(y, feed_dict={x :input})
        


title = "5Y-Level1 5 layers sigmoid Iter=100K B=13k LR=0.01"
vis.losses_accuracies_plots(train_losses,train_acc,train_F1,OG_losses,OG_acc,OG_F1,test_losses, test_acc,test_F1, title, DISPLAY_STEP)


# In[ ]:


title = "5Y-Level1 5 layers sigmoid Iter=100K B=13k LR=0.01"
vis.losses_accuracies_plots(train_losses,train_acc,train_F1,OG_losses,OG_acc,OG_F1,test_losses, test_acc,test_F1, title, DISPLAY_STEP)

