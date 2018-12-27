'''Python code to implement Crime Model'''
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import f1_score
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt




NUM_ITERS=40000
DISPLAY_STEP=1
BATCH=2000
beta=0.005
Mweight=tf.constant(0.8,dtype=tf.float32)
learning_rate = 0.005
window=3

class CrimeProject(object):
    
    def __init__(self):
        # pool the data
        print('Loading data')
        csv_file = 'Charges_df.csv'
        self.data = pd.read_csv(csv_file,index_col=0)
        self.data=self.data.dropna(how='any')

        print('converting date time')
        # convert datetime
        self.data['booking_date'] =  pd.to_datetime(self.data['booking_date'])
        self.data['year'] = self.data['booking_date'].dt.year
        self.data=(self.data.sort_values('booking_date')).reset_index(drop=True)

        # generate dummy data
        print('generating dummy data')
        dummy_cri=pd.get_dummies(self.data['NCIC_Category_Code'])
        dummy_level=pd.get_dummies(self.data['NCIC_level'])
        dummy_df=pd.concat([dummy_cri, dummy_level], axis=1)

        # merge and drop columns
        print('Merging and droping the columns')
        self.data=pd.concat([self.data, dummy_df], axis=1)
        self.data=self.data.drop(['NCIC_Category_Code','no_bookings','NCIC_Crime_Code','NCIC_Category','booking_date'],axis=1)

        # split to test and train
        print('Splitting test and train')
        value_counts = self.data.PersonID.value_counts(dropna=True, sort=True)
        ids = value_counts.rename_axis('ID').reset_index(name='counts')


        # get the Id's for criminals        
        cri1=ids.loc[ids['counts'] <= 2]
        cri2=ids.loc[ids['counts'] > 2]
        train1=cri1.sample(frac=0.8,random_state=200)
        test1=cri1.drop(train1.index)

        train2=cri2.sample(frac=0.8,random_state=200)
        test2=cri2.drop(train2.index)

        train=pd.concat([train1,train2 ])
        test=pd.concat([test1,test2 ])
        train=train['ID'].unique()
        test=test['ID'].unique()

        f_test=self.preprocess(self.data,test)
        f_train=self.preprocess(self.data,train)

        print('generating CSV')
        f_test.to_csv("f_test_5year.csv", index=False)
        f_train.to_csv("f_train_5year.csv", index=False)

        f_test = pd.read_csv("f_test_5year.csv")
        f_train = pd.read_csv("f_train_5year.csv")

        lable_col="next_level"
        
        X_train,y_train=self.train_test(f_train,lable_col,window)
        X_test,y_test=self.train_test(f_test,lable_col,window,test=True)

        train_lab=self.lab(y_train)
        test_lab=self.lab(y_test)

        x_train=X_train.values
        x_test=X_test.values
        self.createModel(x_train,y_train,x_test,y_test,train_lab,test_lab)

    def loadData(self):
        pass

    def train_test(self,df1,lable_col,window,test=False):
        if test:
            df=df1.copy()
            dataR=df.pop('person_race')
            dataG=df.pop('Gender')
            dummy_R = pd.get_dummies(dataR)
            dummy_G = pd.get_dummies(dataG)
            df=pd.concat([df, dummy_R],axis=1)
            df=pd.concat([df, dummy_G],axis=1)
            y=df[lable_col]
            x=df.drop(["PersonID","NCIC_level","next_level",'year'],axis=1)
            x=x.reset_index(drop=True)
            y=y.reset_index(drop=True)
        else:
            
            df=df1.copy()
            
            dataR=df.pop('person_race')
            dataG=df.pop('Gender')
            dummy_R = pd.get_dummies(dataR)
            dummy_G = pd.get_dummies(dataG)
            df=pd.concat([df, dummy_R],axis=1)
            df=pd.concat([df, dummy_G],axis=1)
            df=df.reset_index(drop=True)
            

            number_records_fraud = len(df[df.next_level >=1])
            fraud_indices = np.array(df[df.next_level >=1].index)
            
            split= number_records_fraud*1


            # Picking the indices of the normal classes
            normal_indices = df[df.next_level ==0].index

            # Out of the indices we picked, randomly select "x" number (number_records_fraud)
            random_normal_indices = np.random.choice(normal_indices, split, replace = False)
            random_normal_indices = np.array(random_normal_indices)

            # Appending the 2 indices
            under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

            # Under sample dataset
            under_sample_data = df.iloc[under_sample_indices,:]


            y = under_sample_data[lable_col]
            x=under_sample_data.drop(["PersonID","NCIC_level","next_level",'year'],axis=1)
            
            x=x.reset_index(drop=True)
            y=y.reset_index(drop=True)

        return x,y
        
    def plot_confusion_matrix(self,cls_pred,cls_true,normalize=False):
        
        cm = confusion_matrix(y_true=cls_true,
                            y_pred=cls_pred)
        print(cm)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        else:
            print('Confusion matrix, without normalization')
            print(cm)
        plt.matshow(cm)
        plt.colorbar()
        tick_marks = np.arange(4)
        plt.xticks(tick_marks, range(4))
        plt.yticks(tick_marks, range(4))
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
        plt.show()
    
    
    def smo(self,X_train, y_train):
        sm = SMOTE(ratio={0.0: 31017, 1.0: 6000, 2.0: 6000, 3.0: 6000},random_state=12,k_neighbors=2,m_neighbors=4)
        X_train, y_train = sm.fit_sample(X_train, y_train)
        return X_train, y_train

    def one_hot_encode(self,np_array,i):
        return (np.arange(i) == np_array[:,None]).astype(np.float32)

    def preprocess(self,data,idl):
        x=idl
        df33 = pd.DataFrame()
        for n in x:
            y=data.loc[data['PersonID'] == n]

            uniq=y['year'].unique()
            mrk=[2013,2014,2015,2016,2017]
            if len(uniq)==1 and any(x in mrk for x in uniq):
                df=y
                nb=len(df)
                ex=(df.loc[:, ['PersonID','NCIC_level','Gender','Age','person_race','year']]).tail(1)

                df1=df.copy()
                df1=df1.drop(columns=['NCIC_level','Gender','Age','person_race','year'])
                XF=df1.groupby(["PersonID"],as_index=False).sum()
                XF.drop(columns=['PersonID'],inplace=True)
                ex.reset_index(drop=True, inplace=True)
                XF.reset_index(drop=True, inplace=True)
                doo=pd.concat([ex, XF], axis=1)
                doo['no_bookings']=1
                doo['next_level']=0.0
                df33=df33.append(doo,ignore_index=True)

            else:
                mx=y['year'].max()
                mi=y['year'].min()
                age=y['Age'].min()
                
                for p in range(mi,2013):
                    
                    df = y[(y['year'] <= p)]
                    nb=len(df)
                    ex=(df.loc[:, ['PersonID','NCIC_level','Gender','Age','person_race','year']]).tail(1)
                    df1=df.copy()
                    df1=df1.drop(columns=['NCIC_level','Gender','Age','person_race','year'])
                    XF=df1.groupby(["PersonID"],as_index=False).sum()
                    XF.drop(columns=['PersonID'],inplace=True)
                    ex.reset_index(drop=True, inplace=True)
                    XF.reset_index(drop=True, inplace=True)
                    doo=pd.concat([ex, XF], axis=1)
                    jj=[p+1,p+2,p+3,p+4,p+5]
                    boo = any(x in jj for x in y['year'].unique())
                    
                    if boo==True:
                        bb=y[(y['year'] >= p+1)&(y['year'] <= p+5)]
                        label=bb['NCIC_level'].min()
                        doo['no_bookings']=nb
                        doo.at[0, 'Age'] =  age
                        doo['next_level']=label
                        doo['year']=p
                        df33=df33.append(doo,ignore_index=True)
                        age+=1
                    else:
                        
                        label=0.0
                        doo['no_bookings']=nb
                        doo.at[0, 'Age'] =  age
                        doo['next_level']=label
                        doo['year']=p
                        df33=df33.append(doo,ignore_index=True)
                        age+=1
                    
                        
        return df33
    
    def lab(self,y):
        z=y.copy()
        z[z > 0] = 1
        y=self.one_hot_encode(y,4)
        print(z.shape)
        print(y.shape)
        a=y[:, [1,2,3]]
        b=np.column_stack((z,a))
        return b

    
    def customLoss(self,true,pred):
        
        Btru = tf.slice(true, [0, 0], [-1, 1])
        print(Btru,'Shape of Btrue')
        Bpred = tf.slice(pred, [0, 0], [-1, 1])
        
        Mtru = tf.slice(true, [0, 1], [-1, 3])
        print(Mtru,'Shape of Mtru')
        Mpred = tf.slice(pred, [0, 1], [-1, 3])
        print(Mpred,'Shape of Mpred')
        
        cross_entropyB=tf.nn.sigmoid_cross_entropy_with_logits(logits=Bpred, labels=Btru)+beta*tf.nn.l2_loss(W1) +beta*tf.nn.l2_loss(b1) +beta*tf.nn.l2_loss(W2) +beta*tf.nn.l2_loss(b2) +beta*tf.nn.l2_loss(W3) +beta*tf.nn.l2_loss(b3) +beta*tf.nn.l2_loss(W4) +beta*tf.nn.l2_loss(b4) +beta*tf.nn.l2_loss(W5) +beta*tf.nn.l2_loss(b5)
        
        cross_entropyM = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Mpred, labels=Mtru)+beta*tf.nn.l2_loss(W1) +beta*tf.nn.l2_loss(b1) +beta*tf.nn.l2_loss(W2) +beta*tf.nn.l2_loss(b2) +beta*tf.nn.l2_loss(W3) +beta*tf.nn.l2_loss(b3) +beta*tf.nn.l2_loss(W4) +beta*tf.nn.l2_loss(b4) +beta*tf.nn.l2_loss(W5) +beta*tf.nn.l2_loss(b5)
        print(cross_entropyM,'Shape of CE')

        TBtru=tf.reshape(Btru,[-1,])
        print(TBtru,'Shape of TBtrue')
        cross_entropyM=tf.multiply(TBtru,cross_entropyM)
        print(cross_entropyM,'Shape of CE After')
        loss= tf.multiply((1.0-Mweight),tf.reduce_mean(cross_entropyB))+tf.multiply(Mweight,tf.reduce_mean(cross_entropyM))

        return loss
   
   
    def customaccu(self,true,pred):
        
        Btru = tf.slice(true, [0, 0], [-1, 1])
        Bpred = tf.slice(pred, [0, 0], [-1, 1])
        
        Mtru = tf.slice(true, [0, 1], [-1, 3])
        Mpred = tf.slice(pred, [0, 1], [-1, 3])    
        
        
        predicted = tf.nn.sigmoid(Bpred)
        print(predicted)
        Bipred = tf.round(predicted)
        print(Bipred)
        correct_pred = tf.equal(Bipred, Btru)
        print(correct_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print(accuracy)
        print(Mpred, 'shape of mpred')
        mulpred = tf.cast(tf.argmax(Mpred,1), tf.float32)
        l=tf.constant(1.0,dtype=tf.float32)
        mulpred =mulpred+l
        TBpred=tf.reshape(Bipred,[-1,])
        mulpred = tf.multiply(TBpred,mulpred)
        

        print(mulpred, 'shape of mulpred')
        Mtrue=tf.cast(tf.argmax(Mtru,1), tf.float32)
        Mtrue=Mtrue+l
        TBtru=tf.reshape(Btru,[-1,])
        Mtrue = tf.multiply(TBtru,Mtrue)
        correct_prediction = tf.equal(mulpred, Mtrue)
        print(correct_prediction)

        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        
        return accuracy,accuracy1,Bipred,mulpred
        

    def logmax(self,pred):
        
        Bpred = tf.slice(pred, [0, 0], [-1, 1])
        
        Mpred = tf.slice(pred, [0, 1], [-1, 3])    
        
        
        predicted = tf.nn.sigmoid(Bpred)
        Bipred = tf.round(predicted)

        mulpred = tf.cast(tf.argmax(tf.nn.softmax(Mpred),1), tf.float32)
        l=tf.constant(1.0,dtype=tf.float32)
        mulpredi=mulpred+l
        
        return Bipred,mulpredi

    def createModel(self,x_train,y_train,x_test,y_test,train_lab,test_lab):
        

        tf.set_random_seed(0)

        X = tf.placeholder(tf.float32, [None, 49])
        # correct answers will go here
        Y_ = tf.placeholder(tf.float32, [None, 4])


        # layers sizes
        L1 = 128
        L2 = 64
        L3 = 64
        L4 = 64
        L5 = 4
        # L6 = 16
        # L7 = 2
        # L8 = 16
        # L9 = 8
        # L10 = 2



        # weights - initialized with random values from normal distribution mean=0, stddev=0.1
        # output of one layer is input for the next
        W1 = tf.Variable(tf.truncated_normal([49, L1], stddev=0.1))
        b1 = tf.Variable(tf.zeros([L1]))

        W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
        b2 = tf.Variable(tf.zeros([L2]))

        W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
        b3 = tf.Variable(tf.zeros([L3]))

        W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
        b4 = tf.Variable(tf.zeros([L4]))

        W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
        b5 = tf.Variable(tf.zeros([L5]))





        # -1 in the shape definition means compute automatically the size of this dimension
        XX = tf.reshape(X, [-1, 49])

        # Define model
        Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)


        Ylogits = tf.matmul(Y4, W5) + b5



        cross_entropyx = self.customLoss(Y_,Ylogits)


        accuracy1,_,_,_=self.customaccu(Y_,Ylogits)
        _,accuracy2,_,_=self.customaccu(Y_,Ylogits)

        _,Multi=self.logmax(Ylogits)

        Binary,_=self.logmax(Ylogits)


        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropyx)



        # Initializing the variables
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()





        train_losses = list()
        train_Binary_acc = list()
        train_Multi_acc = list()
        test_losses = list()
        test_Binary_acc = list()
        test_Multi_acc = list()
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

                y_batch = train_lab[idx]

                
                if i%DISPLAY_STEP ==0:
                
                    n=i/DISPLAY_STEP
                    print(n)
                    
                    acc_Binary_trn,acc_mul_trn, loss_trn = sess.run([accuracy1,accuracy2, cross_entropyx], feed_dict={X: X_batch, Y_: y_batch})

                    acc_Binary_tst,acc_mul_tst, loss_tst = sess.run([accuracy1,accuracy2, cross_entropyx], feed_dict={X: x_test, Y_: test_lab})

                    
                    print("#{} Trn Binary acc={}  Trn Multi acc={}  Trn loss={} ,Tst Binary acc={} Tst Multi acc={}  Tst loss={}".format(i,acc_Binary_trn,acc_mul_trn,loss_trn,acc_Binary_tst,acc_mul_tst,loss_tst))

                    train_losses.append(loss_trn)
                    train_Binary_acc.append(acc_Binary_trn)
                    train_Multi_acc.append(acc_mul_trn)

                    test_losses.append(loss_tst)
                    test_Binary_acc.append(acc_Binary_tst)
                    test_Multi_acc.append(acc_mul_tst)

                
                sess.run(train_step, feed_dict={X: X_batch, Y_: y_batch})
                print(learning_rate)

            
            pred_train_Binary = Binary.eval(feed_dict={X: x_train}, session=sess)
            pred_train_Binary =np.ravel(pred_train_Binary[ : , 0])
            pred_train_Multi = Multi.eval(feed_dict={X: x_train}, session=sess)
            print(pred_train_Binary.shape)
            print(pred_train_Multi.shape)

            pred_test_Binary = Binary.eval(feed_dict={X: x_test}, session=sess)
            pred_test_Binary=np.ravel(pred_test_Binary[ : , 0])
            pred_test_Multi = Multi.eval(feed_dict={X: x_test}, session=sess)
            
            print ("accuracy_Binary_train", sess.run(accuracy1, feed_dict={X: x_train, Y_: train_lab}))
            print ("accuracy_Multi_train", sess.run(accuracy2, feed_dict={X: x_train, Y_: train_lab}))

            truelab=y_train

            Predlab=np.multiply(np.array(pred_train_Binary),np.array(pred_train_Multi))

            f1_train=f1_score(truelab,Predlab, average='macro')
            print ("F1-Score_train", f1_train)
            plot_confusion_matrix(Predlab,truelab,normalize=True)
            
                
            print ("accuracy_Binary_test", sess.run(accuracy1, feed_dict={X: x_test, Y_: test_lab}))
            print ("accuracy_Multi_test", sess.run(accuracy2, feed_dict={X: x_test, Y_: test_lab}))

            truelab1=y_test

            Predlab1=np.multiply(np.array(pred_test_Binary),np.array(pred_test_Multi))

            f1_test=f1_score(truelab1,Predlab1,average='macro')
            print ("F1-Score", f1_test)
            self.plot_confusion_matrix(Predlab1,truelab1,normalize=True)



    def train(self):
        pass

    def predict(self):
        pass

if __name__=='__main__':
    c=CrimeProject()