# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:28:20 2023

@author: Vaibhav Bhorkade
Assignment Association Rules

Problem Statement: -
Kitabi Duniya, a famous book store in India, which was 
established before Independence, the growth of the company was
incremental year by year, but due to online selling of books 
and wide spread Internet access its annual growth started to 
collapse, seeing sharp downfalls, you as a Data Scientist 
help this heritage book store gain its popularity back and 
increase footfall of customers and provide ways the business can
improve exponentially, apply Association RuleAlgorithm, explain 
the rules, and visualize the graphs for clear understanding of 
solution.
"""
"""
Business Objective
Minimize : Increase sale of Online book shopping
Maximaze : Sales of books
Business constraints  
"""
"""
Data Dictionary

Name of features     Type Relevance      Description
0          ChildBks  Nominal  Relevant      Child Related books
1          YouthBks  Nominal  Relevant      Youth Related books
2           CookBks  Nominal  Relevant    Cooking Related books
3          DoItYBks  Nominal  Relevant      DoItY Related books
4            RefBks  Nominal  Relevant  Reference Related books
5            ArtBks  Nominal  Relevant        Art Related books
6           GeogBks  Nominal  Relevant  Geography Related books
7          ItalCook  Nominal  Relevant   ItalCook Related books
8         ItalAtlas  Nominal  Relevant  ItalAtlas Related books
9           ItalArt  Nominal  Relevant    ItalArt Related books
10         Florence  Nominal  Relevant   Florence Related books

"""

# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("C:/datasets/book.csv")
df.head
df.tail
# describe - 5 number summary
df.describe()
df.shape
# 2000 rows and 11 columns
df.columns
# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 , no null values

# Scatterplot
sns.set_style("whitegrid");
sns.FacetGrid(df) \
   .map(plt.scatter, "ChildBks", "YouthBks") \
   .add_legend();
plt.show();
# all the points between 0 to 1

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# boxplot
# boxplot on ChildBks column
sns.boxplot(df.ChildBks)
# In ChildBks column no outliers 

# boxplot on df column
sns.boxplot(df)
# There is outliers on some columns

# boxplot on YouthBks column
sns.boxplot(df.YouthBks)
# There is 1 outliers on column

# boxplot on df column
sns.boxplot(df.DoItYBks)
# There is no outliers on column

# histplot
sns.histplot(df['DoItYBks'],kde=True)
# normally skew and the normallly distributed
# May be symmetric

sns.histplot(df['YouthBks'],kde=True)
# normally skew and the normallly distributed
# May be symmetric

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

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
# output is 1680

# We found outliers in some columns 
# Outliers treatment

IQR=df.YouthBks.quantile(0.75)-df.YouthBks.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=df.YouthBks.quantile(0.25)-1.5*IQR

upper_limit=df.YouthBks.quantile(0.75)+1.5*IQR

# Trimming
import numpy as np

outliers_df=np.where(df.YouthBks>upper_limit,True,np.where(df.YouthBks<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df1=df.loc[~outliers_df]
df.shape
# (2000, 11)
df1.shape
# (1505, 11)

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

from mlxtend.frequent_patterns import apriori,association_rules 
#Here we are going to use data 
# We can use pandas to load this structured data 

#Now let us plot bar graph of item frequencies 
import pandas as pd

#you will get one hot encoded data frame of size 9835x169
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

import matplotlib.pyplot as plt 
plt.bar(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs. Confidence')
plt.show()

# We handled outliers, normalized data, analysis and EDA.
# Apriori algorithm extracted association rules  business objectives are needed.
# insights should focus on translating association rules 
