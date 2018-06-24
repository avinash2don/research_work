
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow.contrib import rnn
import random
import collections
import time
import json
# import utils as utl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools


# In[3]:


filename = '/Users/avinash2don/Dropbox/Crime-Prediction/Final Code ( R and Python)/Charges_df.csv'
logs_path = '/Users/avinash2don/jupyter/log'
writer = tf.summary.FileWriter(logs_path)


# In[4]:


data = pd.read_csv(filename)
display(data.head())


# In[5]:


def read_data(filename):
    
    data = pd.read_csv(filename)
    data['feature1'] = MinMaxScaler().fit_transform(data[['feature1']])
#    data.feature4 = data.feature4.fillna(4.0)
    data = data.dropna(axis=0, how='any')
    data1 = data[['key','feature2','feature3','feature4','feature5','feature1','feature6','feature7']].copy()
    unique_p = data1['key'].unique()

    print( len(unique_p) )
    print( len(data1) )
    
    gru = pd.DataFrame()
    count = 0
    for i in unique_p:
        count += 1
        if(count%100==0):
            print ("unique person: ",count)
            
        df=data1[data1['key'] == i]
#         df=df.sort_values('feature7')
        gru = gru.append( df.loc[ df.groupby(['feature7']).feature4.idxmin()], ignore_index=True )
    
    print ("Completed this")
    unique_persons = gru['key'].unique()
    count = 0
    
    
    X = list()
    Y = list()
    
    for i in unique_persons:
        person_df = gru[gru['key'] == i]
        if len( person_df )>1:
            person_df = person_df.sort_values('feature7')
            l=len(person_df)
            Y.append(l)
            person_df = person_df[['feature3','feature4','feature6','feature1']]
            #     person_df['feature1']=MinMaxScaler().fit_transform(person_df[['feature1']])
            x= person_df.values

            X.append(x)
        
        else:
            count+= 1
            
            
    L=max(Y)
    return X,L,gru


# In[6]:


X,maxL,gru=read_data(filename)
print(X)


# In[7]:


print(len(X))
print(maxL)
print(X[1])


# In[7]:


unique_levels = gru['feature4'].unique()
dics=list()

dicCri = dict()    
unique_charges = gru['feature3'].unique()

for crime in unique_charges:
    dicCri[crime] = len(dicCri)
    
dics.append(dicCri)  

print(len(dicCri))

dictionary = dict()

for level in unique_levels:
    dictionary[level] = len(dictionary)
    
dics.append(dictionary) 

print(len(dictionary))

dicR=dict()
unique_race = gru['feature6'].unique()
for race in unique_race:
    dicR[race] = len(dicR)
    
dics.append(dicR)
    
size_dict = len(dictionary)
feature_size=len(dictionary)+len(dicCri)+len(dicR)+1

print(dics)


# In[8]:


learning_rate = 0.001
epochs = 30
display_step = 1
n_input = 46

# number of units in RNN cell
n_hidden = 512

# RNN output node weights and biases
with tf.variable_scope("data", reuse = tf.AUTO_REUSE):
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, size_dict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([size_dict]))
    }

x = tf.placeholder("float",[None, 46, 1])
y = tf.placeholder("float",[None, 3 ])


# In[9]:


print ( x.get_shape() )
print ( y.get_shape() )


# In[10]:


def RNN(x, weights, biases):
    
    print (x.get_shape())
    ## x.get_shape() is (?, 52, 1)
#     x = tf.reshape(x, [-1, 1])
    
#     # (eg. [had] [a] [general] -> [20] [6] [33])
#     x = tf.split( x, 1, 1)
    x = tf.unstack(x, axis = 2)
    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


with tf.variable_scope("network", reuse = tf.AUTO_REUSE):
    
    pred = RNN( x, weights, biases)
    print ("The pred shape is : ",pred.get_shape())
    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    


# In[14]:


offset = random.randint(0, n_input+1)
print(offset)


# In[11]:


#Launch the graph
from sklearn.model_selection import train_test_split

with tf.Session() as session:
    
    session.run(init)
    step = 0
    offset = random.randint(0, n_input+1)
    
    # end_offset = n_input+1
    acc_total = 0
    loss_total = 0
    
    # epochs = 50    
    writer.add_graph(session.graph)
    
    #_datax, _datay = training_data, testing_data
    acc_final = 0
    start_time = time.time()
    
    while step < epochs:    
        acc_total = 0
        train_data, test_data = train_test_split(X, test_size=0.3)    
        
        for i in range(len(train_data)):
            #print ("i is: ",i)
            training_set = train_data[i]
            count = 0
            offset =0
            
            crime_level_column = training_set[1:,1] # vector
            L=0
            
            while (offset+1) < len(training_set):
                
                ## 4 length vector
                single_crime = training_set[offset]
                sample_array = np.zeros([feature_size], dtype = float)
                
                ### crime level column = dataframe['crime_level']
                
                for j in range(3):
                    
                    feature_key = single_crime[j]#name of the crime
                    feature_pair = dics[j][feature_key] # number from the diction
                    # one hot encoding
                    
                    if(j==0):
                        sample_array[feature_pair] = 1
                    if(j==1):
                        sample_array[37+feature_pair] = 1
                    if(j==2):
                        sample_array[40+feature_pair] = 1
                    
                sample_array[-1] = single_crime[3]
                sample_array = np.reshape(np.array(sample_array),[-1,n_input,1])
                labels = np.zeros([size_dict ], dtype = float) # size_dict = dictionary of crime level. 
                
                labels[dictionary[crime_level_column[L,]]] = 1.0 
                labels = np.reshape( labels, (1,3) )
                
                #print("Sample Array shape is: ",sample_array.shape)
                #print("Labels shape is: ",labels.shape)
                test = session.run( pred, feed_dict = {x:sample_array})
                #print("The test shape is: ",test.shape)
                    
                _, loss = session.run([optimizer, cost],
                                      feed_dict = { x:sample_array, y: labels})
                
                #print("The loss is: ",loss)
                loss_total += loss
                offset +=1
                L+=1
        acc_total = 0
        #print(acc_total)
        
		# Do siminlar in testing set as well and find accurcy pertaining to the test set. Accuracy will be different for each iteration, we take an averfeature1 accuracy over number of iterations.
        for i in range(len(test_data)):
            #print ("i is: ",i)
            training_set = test_data[i]
            count = 0
            offset =0
            acc_temp = 0.0
            crime_level_column = training_set[1:,1] # vector
            L=0
            
            while (offset+1) < len(training_set):
                
                ## 4 length vector
                single_crime = training_set[offset]
                sample_array = np.zeros([feature_size], dtype = float)
                
                ### crime level column = dataframe['crime_level']
                
                for j in range(3):
                    
                    feature_key = single_crime[j]#name of the crime
                    feature_pair = dics[j][feature_key] # number from the diction
                    # one hot encoding
                    
                    if(j==0):
                        sample_array[feature_pair] = 1
                    if(j==1):
                        sample_array[37+feature_pair] = 1
                    if(j==2):
                        sample_array[40+feature_pair] = 1
                    
                sample_array[-1] = single_crime[3]
                sample_array = np.reshape(np.array(sample_array),[-1,n_input,1])
                labels = np.zeros([size_dict ], dtype = float) # size_dict = dictionary of crime level. 
                
                labels[dictionary[crime_level_column[L,]]] = 1.0 
                labels = np.reshape( labels, (1,3) )
                predictions, acc = session.run([correct_pred, accuracy],
                                                    feed_dict = {x:sample_array, y:labels})
                acc_temp += acc
                offset +=1
                count += 1
                L+=1
            if(count!=0):
                acc_temp = acc_temp/count
            acc_total += acc_temp
        print(acc_total/(i+1))
        acc_total = acc_total/(i+1)
        acc_final += acc_total
        step += 1
    elapsed_time = time.time() - start_time
    print('Final Accuracy:' + str(acc_final/epochs))

print("\ttensorboard -- logdir=%s" %(logs_path))

