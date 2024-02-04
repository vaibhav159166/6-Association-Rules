# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:58:25 2023

@author: Vaibhav Bhorkade

Problem Statement: - 
A film distribution company wants to target audience based on their 
likes and dislikes, you as a Chief Data Scientist Analyze the data and 
come up with different rules of movie list so that the business 
objective is achieved.

"""
"""
Business Objective 
Minimize : dislikes movies or dissimilar movies recommandation
Maximaze : likes movies and similar type movies
Business constraints
"""
"""
Data Dictionary

 Name of features     Type Relevance    Description
0      Sixth Sense  Nominal  Relevant    Sixth Sense
1        Gladiator  Nominal  Relevant      Gladiator
2            LOTR1  Nominal  Relevant          LOTR1
3    Harry Potter1  Nominal  Relevant  Harry Potter1
4          Patriot  Nominal  Relevant        Patriot
5            LOTR2  Nominal  Relevant          LOTR2
6    Harry Potter2  Nominal  Relevant  Harry Potter2
7             LOTR  Nominal  Relevant           LOTR
8       Braveheart  Nominal  Relevant     Braveheart
9       Green Mile  Nominal  Relevant     Green Mile

"""

# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# DataFrame
df=pd.read_csv("my_movies.csv")

df.head
df.tail
# describe - 5 number summary
df.describe()
df.shape
# 10 rows and 10 columns
df.columns
# ['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
#           'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile']

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 , no null values

# if null the drop it
# df.dropna()

# Scatterplot
sns.set_style("whitegrid");
sns.FacetGrid(df) \
   .map(plt.scatter, "Sixth Sense", "Gladiator") \
   .add_legend();
plt.show();
# all the points between 0 to 1

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# boxplot
# boxplot on Gladiator column
sns.boxplot(df.Gladiator)
# In  column Gladiator no outliers 

# boxplot on df column
sns.boxplot(df)
# There is outliers on some columns

# boxplot on LOTR2 column
sns.boxplot(df.LOTR2)
# There is 1 outliers on column

# boxplot on LOTR1 df column
sns.boxplot(df.LOTR1)
# There is 1 outliers on column

# histplot
sns.histplot(df['LOTR2'],kde=True)
# not skew and the normallly not distributed
# not symmetric

sns.histplot(df['LOTR1'],kde=True)
# non skew and the not normallly distributed


sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# mean
df.mean()
# mean of Gladiator  is 0.7 and highest

# median
df.median()
# median of 3 columns is 1 

# mode
df.mode()

# standard deviation
df.std()
'''
Sixth Sense      0.516398
Gladiator        0.483046
LOTR1            0.421637
Harry Potter1    0.421637
Patriot          0.516398
LOTR2            0.421637
Harry Potter2    0.316228
LOTR             0.316228
Braveheart       0.316228
Green Mile       0.421637
'''
# Data Preproccesing
df.dtypes
# all columns in int data types

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created

duplicate
# False
sum(duplicate)
# output is 3
df.drop_duplicates(inplace=True)

duplicate=df.duplicated()
sum(duplicate)
# Now sum of duplicate is zero
# Duplicates are removed

# We found outliers in some columns 
# Outliers treatment

IQR=df.LOTR1.quantile(0.75)-df.LOTR1.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=df.LOTR1.quantile(0.25)-1.5*IQR

upper_limit=df.LOTR1.quantile(0.75)+1.5*IQR

# Trimming
import numpy as np

outliers_df=np.where(df.LOTR1>upper_limit,True,np.where(df.LOTR1<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df1=df.loc[~outliers_df]
df.shape
# (7, 10)
df1.shape
# (7, 10)

sns.boxplot(df1)
# some outliers are removed

# Normalization

# Normalization function
# whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df1) 
# you can check the df_norm dataframe which is scaled between values from 0 and 1
b=df_norm.describe()
# Data is normalize
# already in normalize form in 0-1 

# Association Rule
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

#This is our input data to applyy to apriori algorithm, it will generate !169 rules, min support values 
#is 0.0075 (it must be between 0 to 1),
#you can give any number but must be between 0 to 1 
frequent_itemsets = apriori(df,min_support=0.0075,max_len=4,use_colnames=True)
#you will get support values for 1,2,3 and 4 max items 
#lett us sort these support values 
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order 
#Even EDA was also have the same trend, in EDA there was count 
#and here it is support value 
#we will generate  association rules, This association rule will calculate all the matrix of each and every combination 
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
#this generate associatin rules of size 1198x9 columns 
#comprises of antescends, consequences 
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# After applying the apriori and association_rules we found insights
# where the same movies showing in columns with its 
# antecedents and consequents and its value of conviction.