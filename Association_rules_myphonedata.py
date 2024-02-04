# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:03:58 2023

@author: Vaibhav Bhorkade

Problem Statement: - 
A Mobile Phone manufacturing company wants to launch its three 
brand new phone into the market, but before going with its traditional 
marketing approach this time it want to analyze the data of its
previous model sales in different regions and you have been hired 
as an Data Scientist to help them out, use the Association rules 
concept and provide your insights to the company’s marketing team to 
improve its sales.
"""
"""
Business Objective
Minimize : unliked color phones
Maximaze : Recommandation of good Saleing  colors phones
Business constraints  
"""
"""
Data Dictionary

  Name of Features     Type Relevance   Description
0              red  Nominal  Relevant     red color
1            white  Nominal  Relevant   white color
2            green  Nominal  Relevant   green color
3           yellow  Nominal  Relevant  yellow color
4           orange  Nominal  Relevant  orange color
5             blue  Nominal  Relevant    blue color

"""

# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# DataFrame
df=pd.read_csv("myphonedata.csv")

df.head
df.tail
# describe - 5 number summary
df.describe()
df.shape
# 11 rows and 6 columns
df.columns
# ['red', 'white', 'green', 'yellow', 'orange', 'blue']

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 , no null values

# Scatterplot
sns.set_style("whitegrid");
sns.FacetGrid(df) \
   .map(plt.scatter, "red", "white") \
   .add_legend();
plt.show();
# all the points between 0 to 1

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# boxplot
# boxplot on red column
sns.boxplot(df.red)
# In red column no outliers 

# boxplot on df column
sns.boxplot(df)
# There is outliers on some columns

# boxplot on green column
sns.boxplot(df.green)
# There is 1 outliers on column

# boxplot on yellow df column
sns.boxplot(df.yellow)
# There is 1 outliers on column

# histplot
sns.histplot(df['green'],kde=True)
# not skew and the normallly distributed
# not symmetric

sns.histplot(df['red'],kde=True)
# non skew and the not normallly distributed


sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# mean
df.mean()
# mean of white is 0.6 and highest

# median
df.median()
# median of red white and blue is 1 

# mode
df.mode()

# standard deviation
df.std()
'''
red       0.522233
white     0.504525
green     0.404520
yellow    0.301511
orange    0.404520
blue      0.522233
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

# We found outliers in some columns 
# Outliers treatment

IQR=df.green.quantile(0.75)-df.green.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=df.green.quantile(0.25)-1.5*IQR

upper_limit=df.green.quantile(0.75)+1.5*IQR

# Trimming
import numpy as np

outliers_df=np.where(df.green>upper_limit,True,np.where(df.green<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df1=df.loc[~outliers_df]
df.shape
# (11, 6)
df1.shape
# (9, 11)

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
# in 0-1 

# Association Rule
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

#Now let us plot bar graph of item frequencies 

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
# where the same colors showing in same columns with its 
# antecedents and consequents and its value of conviction.
