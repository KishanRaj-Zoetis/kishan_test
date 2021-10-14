# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:22:02 2021

@author: RAJK
"""

import numpy as np
from itertools import combinations, groupby
from collections import Counter
import pyodbc
import pandas as pd
from itertools import chain
# Sample data
from datetime import datetime
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
search = SearchEngine(simple_zipcode=True)

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=KZDWSQLBARCOP02\BARCOPROD02;'
                      #'Database=db_name;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
sql_all="""
(

select narc,ProRevZS/ProjRev as zts_market_share
from(

select 
	NARC

	,sum(ProjectedRevenue) as ProjRev
	,sum(case
			when Manufacturer = 'Zoetis' then ProjectedRevenue
			else 0
		end) as ProRevZS
	from BA_USBA.dbo.Tbl_vw_CurrentNARCAlignments_2020 as align
	left join BA_Animalytix.dbo.vsTrendsProjected as Market
	On Market.ZoetisArea=align.AreaID
	where SalesForceName='Small Animal' 
	and BusinessClass = 'Veterinarian'
	AND MonthYear >= '2019-07-01' AND MonthYear < '2020-07-01'
	group by NARC) as q2

)"""
market_share = pd.read_sql(sql_all,conn)
market_share['narc']=market_share['narc'].astype(int)

#time frame # Aug 2020- Sep 2021
sales = pd.read_csv('C:/Users/rajk/Documents/Projects/Association rules/Data/Ap_Cy_all_sales.csv')

acc_lik = 'C:/Users/rajk/Documents/Projects/Association rules/accounts.csv'
accounts = pd.read_csv(acc_lik)
accounts['zip'] = accounts['narczip'].astype(str).str[:5]

urb_lik = 'C:/Users/rajk/Documents/Projects/Association rules/all_urban.csv'
urban = pd.read_csv(urb_lik)

accounts=pd.merge(accounts,urban,on='narc',how='left')

accounts=pd.merge(accounts,market_share,on='narc',how='left')

rep_lik = 'C:/Users/rajk/Documents/Projects/Association rules/data/rep_calls.csv'
rep_calls = pd.read_csv(rep_lik)
accounts['narc']=accounts['narc'].astype(str)

all_acc=accounts['narc'].unique()
# 61% of all calls made were Apoquel
# 39% of derm calls were cytopoint

rep_calls=rep_calls.loc[rep_calls['narcid'].isin(all_acc),:]

rep_calls.Apoquel_calls.sum()
#62819
rep_calls.Cytopoint_calls.sum()
#39296
rep_calls.total_calls.sum()
#102115

# how did the reps call the people who had 


rep_calls['calls_ratio']=rep_calls['Cytopoint_calls']/rep_calls['total_calls']

accounts=pd.merge(accounts,rep_calls,left_on='narc',right_on='narcid',how='left')




def get_population(zip_code):
    zipcode = search.by_zipcode(zip_code)
    data_list=[[zip_code,zipcode.population,zipcode.population_density,zipcode.median_household_income]]
    #return(data_list)
    return(pd.DataFrame(data_list,columns=['zip_code','population','population_density','median_household_income']))
    

zip_list_all=[]
for zip_code in accounts.zip.unique():
    #print(zip_code)
    zip_list=get_population(zip_code)
    zip_list_all.append(zip_list)
    

zip_population = pd.concat(zip_list_all)



accounts_df=pd.merge(accounts,zip_population,left_on='zip',right_on='zip_code',how='left')





derm=sales.loc[(sales['Apoquel_sales']>0) | (sales['Cytopoint_sales']>0),:]

derm['derm_ratio']=derm['Cytopoint_sales']/(derm['Apoquel_sales']+derm['Cytopoint_sales'])
derm['derm_ratio']=np.where(derm['derm_ratio']>100,100,derm['derm_ratio'])
derm['derm_ratio']=np.where(derm['derm_ratio']<0,0,derm['derm_ratio'])






accounts_df['narc']=accounts_df['narc'].astype(int)
ly_ap=pd.merge(derm,accounts_df, left_on='buyernarc',right_on='narc',how='left')

pp=ly_ap.columns[0:102]

ly_ap['convert_flag']=np.where(ly_ap['derm_ratio']>0.30,1,0)
modeling_df=ly_ap[['convert_flag','petcareregion',                   
          'corp_flag',
          'tbm_present_flag','sam_present_flag','idsr_present_flag',
          #'rep_vacant_flag'
          'last12monthstotalsales','narc_segment','last12monthsproductuse',
          'total_idsr_calls_past_6months','total_tbm_calls_past_6months','total_sam_calls_past_6months',
          'is_urban','median_household_income','population','zts_market_share','Cytopoint_calls'
                   
          ]]

def replace_na_and_neg(df, column_list,impute_value=0):   
    """
    This function is used to replace the negative and NAs sales and units in multiple columns
    return: the same dimension of the input df, but column specified is imputed
    ex: 
    replace_na_and_neg(rev, 
    column_list=['rev_feline_sales_past_6months','rev_plus_sales_past_6months'],
    impute_value=0)
    """
    df_imputed = df.drop(column_list,axis=1,inplace= False)
    for col in column_list:
        df_imputed[col] = np.where((df[col] < 0) | (df[col].isna()), impute_value, df[col])
    return df_imputed
column_list = ['last12monthstotalsales','last12monthsproductuse',
          'total_idsr_calls_past_6months','total_tbm_calls_past_6months','total_sam_calls_past_6months',
          'median_household_income','population','zts_market_share','Cytopoint_calls']

modeling_df=replace_na_and_neg(modeling_df, column_list=column_list,impute_value=0)


categorical_feature_list=['petcareregion','corp_flag'
                          ,'tbm_present_flag','sam_present_flag','idsr_present_flag',
          #'rep_vacant_flag'
          'narc_segment','is_urban']

modeling_df =pd.get_dummies(data=modeling_df, columns=categorical_feature_list)



#Import packages for modeling
from IPython.display import Image  
import sklearn
#import pydotplus 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
seed = 42
import seaborn as sns

X=modeling_df.drop('convert_flag',axis=1)
y=modeling_df['convert_flag']

modeling_df['convert_flag'].value_counts()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(x_train)

X_test = scaler.transform(x_test)


# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier


# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)


# fit the model

rfc.fit(X_train, y_train)


# Predict the Test set results

y_pred = rfc.predict(X_test)


# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)


# fit the model to the training set

rfc_100.fit(X_train, y_train)


# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)


# Check accuracy score 

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))


# create the classifier with n_estimators = 100

clf = RandomForestClassifier(n_estimators=100, random_state=0)


# fit the model to the training set

clf.fit(X_train, y_train)


feature_scores = pd.Series(clf.feature_importances_, index=x_train.columns).sort_values(ascending=False)

feature_scores

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

dt = DecisionTreeClassifier(criterion = "gini", 
                            random_state = seed,
                            max_depth=7.5, 
                            min_samples_split=0.3
                           )
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(dt, 
                   feature_names=X.columns, 
                   class_names=y.name,
                   filled=True)
