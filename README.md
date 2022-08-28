# python2
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import tensorflow as tf 
%matplotlib inline
train_df=pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
train_df
train_df['Ticket']= train_df['Ticket'].apply(lambda x : x.split(' ')[-1] if len(x.split(' ')) >=2 else x)
train_df['Ticket']= train_df['Ticket'].apply(lambda x : x if x[0].isdigit() else np.NAN)
train_df=train_df.drop(['Name'],axis=1)
train_df
