# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:34:43 2021

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



#########Get overall sales data

#time frame # Aug 2020- Sep 2021
sales = pd.read_csv('C:/Users/rajk/Documents/Projects/Association rules/Data/Ap_Cy_all_sales.csv')



# Total APoquel customers

ly_ap=sales.loc[sales['Apoquel_sales']>0,:]
apo_customers=ly_ap.buyernarc.unique()
len(apo_customers)

# Total cytopoint customers

ly_cy=sales.loc[sales['Cytopoint_sales']>0,:]
cyt_customers=ly_cy.buyernarc.unique()
len(cyt_customers)


len(apo_customers)/(len(apo_customers)+len(cyt_customers))


len(cyt_customers)/(len(apo_customers)+len(cyt_customers))

# Only customers 

#Appoquel only customers

ly_ap_only=sales.loc[(sales['Apoquel_sales']>0) & (sales['Cytopoint_sales']==0),:]



# Cytopoint only customers

ly_cy_only=sales.loc[(sales['Cytopoint_sales']>0) & (sales['Apoquel_sales']==0),:]


# Total Revenue from Apoquel 

# Trio adoption metric




#time frame # Aug 2020- Sep 2021
trio = pd.read_csv('C:/Users/rajk/Documents/Projects/Association rules/Data/Ap_Cy_sim_sales.csv')

trio=trio.loc[(trio['trio_sales']>0) | (trio['sim_sales']>0),:]



trio['trio_ratio']=trio['sim_sales']/(trio['trio_sales']+trio['sim_sales'])
trio['trio_ratio']=np.where(trio['trio_ratio']>100,100,trio['trio_ratio'])
trio['trio_ratio']=np.where(trio['trio_ratio']<0,0,trio['trio_ratio'])
trio = trio.replace([np.inf],100)

trio['trio_ratio'].describe()



##############Derm customer analysis




derm=sales.loc[(sales['Apoquel_sales']>0) | (sales['Cytopoint_sales']>0),:]

derm['derm_ratio']=derm['Cytopoint_sales']/(derm['Apoquel_sales']+derm['Cytopoint_sales'])
derm['derm_ratio']=np.where(derm['derm_ratio']>100,100,derm['derm_ratio'])
derm['derm_ratio']=np.where(derm['derm_ratio']<0,0,derm['derm_ratio'])

derm['derm_ratio'].describe()

# Compare derm ratio and trio ratio

derm_trio=pd.merge(derm,trio[['buyernarc','trio_sales','sim_sales','trio_ratio']],on='buyernarc',how='left')

derm_trio=derm_trio.loc[derm_trio['trio_ratio']>0,:]

derm_trio['trio_ratio'].describe()

derm_trio['trio_high_adoptor']=np.where(derm_trio['trio_ratio']>=0.5,1,0)

derm_trio['trio_high_adoptor'].value_counts()

derm_trio.groupby(['trio_high_adoptor']).derm_ratio.describe()


derm_trio['trio_sales'].corr(derm_trio['Cytopoint_sales'],method='spearman')


##################Explanatory analysis


derm['derm_ratio'].describe()

acc_lik = 'C:/Users/rajk/Documents/Projects/Association rules/accounts.csv'
accounts = pd.read_csv(acc_lik)


urb_lik = 'C:/Users/rajk/Documents/Projects/Association rules/all_urban.csv'
urban = pd.read_csv(urb_lik)

accounts=pd.merge(accounts,urban,on='narc',how='left')




# How is the current landscape in terms of 

#a Geography

ly_ap=pd.merge(derm,accounts, left_on='buyernarc',right_on='narc',how='left')
#1 major account name

major_account_ap=ly_ap.groupby(['major_account_name']).derm_ratio.median().reset_index()
major_account_ap.columns=['major_account_name','derm_ratio']

major_account_ap['better_placed']=np.where(major_account_ap['derm_ratio']>derm['derm_ratio'].median(),1,0)


petcareregion_ap=ly_ap.groupby(['petcareregion','is_urban']).derm_ratio.median().reset_index()

urban_ap=ly_ap.groupby(['is_urban']).derm_ratio.describe().reset_index()

#derm_urban=pd.merge(derm,urban_narcs,left_on='buyernarc',right_on='narc',how='left')

#derm_urban.is_urban.value_counts()
##############Theory for maximization


derm['Derm_Revenue']=derm['Cytopoint_sales']+derm['Apoquel_sales']

derm['apo_ratio']=derm['Apoquel_sales']/derm['Derm_Revenue']

derm['cyto_ratio']=derm['Cytopoint_sales']/derm['Derm_Revenue']





min_apo_ratio=0.40
min_cytopoint_ratio=0.40


derm['apo_canib']=derm['apo_ratio']-min_apo_ratio

derm['cyt_canib']=derm['cyto_ratio']-min_cytopoint_ratio


#####Metrics

derm['zero_cy_users']=np.where(derm['cyto_ratio']==0,1,0)


derm['high_apoquel_users']=np.where(derm['apo_canib']>=0.30,1,0)

derm['apoquel_users']=np.where((derm['apo_canib']<=0.30)& (derm['apo_canib']>0),1,0)


derm['high_cyt_users']=np.where((derm['apo_canib']<0)& (derm['cyt_canib']>0),1,0)


derm['zero_cy_users'].sum()
derm['high_apoquel_users'].sum()
derm['apoquel_users'].sum()

derm['high_cyt_users'].sum()
#######################################Money matters

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=KZDWSQLBARCOP02\BARCOPROD02;'
                      #'Database=db_name;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
sql_all="""
(

select 
buyernarc
,sum(saleamnt) as sales
,year
,prodname
,A.TerritoryManager
,A.RegionName
from BA_USBA.dbo.vw_BuyerGLRSales s
join [BA_USBA].[dbo].[Tbl_vw_CurrentNARCAlignments] A
    on S.BuyerNarc = A.NARC
Where S.Year >=2018
  and A.businessclasscode in ('06','09','16','02')
  and A.SalesForceId in ('100', '101')
  and RegionID not like '%H%'
and prodname  in ('Cytopoint','Apoquel')
group by buyernarc,year,A.TerritoryManager,prodname
,A.RegionName
)"""
all_accounts = pd.read_sql(sql_all,conn)





# All customers
baseline=all_accounts
all_accounts=baseline
all_accounts = all_accounts.rename(columns={'buyernarc': 'BuyerNARC','prodname':'ProdName'})


all_accounts=all_accounts.loc[all_accounts['sales']>0,:]



all_narcs=list(all_accounts.BuyerNARC.unique())


# Total number of unique customers who purchased apoquel

apo_all=all_accounts.loc[all_accounts['ProdName']=='Apoquel',:]

apo_customers=apo_all.BuyerNARC.unique()

len(apo_customers)


apo_all_year=apo_all.groupby(['year']).BuyerNARC.nunique().reset_index()


# total number of customers who purchased cytopoint

cyt_all=all_accounts.loc[all_accounts['ProdName']=='Cytopoint',:]
cyt_customers=cyt_all.BuyerNARC.unique()
len(cyt_customers)


cyt_all_year=cyt_all.groupby(['year']).BuyerNARC.nunique().reset_index()




# only apoquel

apo_only = list(set(apo_customers) - set(cyt_customers))
print(" Peoplewho purchased apoquel only")
print(len(apo_only))

# only cytopint

cyt_only = list(set(cyt_customers) - set(apo_customers))
print(" People who purchased cytopoint only")
print(len(cyt_only))
# Both




#Ratio of apoquel to cytopoint

Apo_sum_sales=apo_all.groupby(['BuyerNARC','year']).sales.sum()

Apo_sum_sales=Apo_sum_sales.reset_index()

Apo_sum_sales.columns=['BuyerNARC','year','apoquel_sales']

cyt_sum_sales=cyt_all.groupby(['BuyerNARC','year']).sales.sum()

cyt_sum_sales=cyt_sum_sales.reset_index()

cyt_sum_sales.columns=['BuyerNARC','year','cytopoint_sales']

sale_ratio=pd.merge(Apo_sum_sales,cyt_sum_sales,on=['BuyerNARC','year'],how='outer')


sale_ratio=sale_ratio.fillna(0)



sale_ratio['cytopoint_sales']=np.where(sale_ratio['cytopoint_sales']==0,0,sale_ratio['cytopoint_sales'])

sale_ratio['apoquel_sales']=np.where(sale_ratio['apoquel_sales']==0,0,sale_ratio['apoquel_sales'])

sale_ratio['sales_ratio']=sale_ratio['cytopoint_sales']/(sale_ratio['cytopoint_sales']+sale_ratio['apoquel_sales'])



#switch analysis


sale_ratio_switch=sale_ratio

sale_ratio_switch.fillna(0)

sale_ratio_switch = sale_ratio_switch.sort_values(by=['BuyerNARC','year'], ascending=True)
sale_ratio_switch['change'] = sale_ratio_switch.groupby(['BuyerNARC'], as_index=False)['sales_ratio'].diff()



sale_ratio_switch['change'] =sale_ratio_switch['change'].fillna(0)
sale_ratio_switch['positive_moves'] = np.where(sale_ratio_switch['change']>0,1,0)

sale_ratio_switch['negative_moves'] = np.where(sale_ratio_switch['change']<0,1,0)



sale_ratio_switch=sale_ratio_switch.loc[sale_ratio_switch['year']<2021,:]

sale_ratio_switch['cum_sum'] = sale_ratio_switch.groupby(['BuyerNARC']).change.transform('sum')



sale_ratio_switch.groupby(['year']).positive_moves.sum()

sale_ratio_switch.groupby(['year']).negative_moves.sum()
# ratio of cytopoint to appoquel over time

#mark accounts that are purchasing more cytopint 



# Real deal


# Who are switching from apoquel to cytopoint 

# Restrict to accounts who have purchased some appoquel and some cytopoint all three years
sale_ratio_switch=sale_ratio_switch.loc[sale_ratio_switch['apoquel_sales']>0,:]

sale_ratio_switch=sale_ratio_switch.loc[sale_ratio_switch['cytopoint_sales']>0,:]

# get accounts that have more than 3 years worth of sales

sale_ratio_switch['years_active'] = sale_ratio_switch.groupby(['BuyerNARC']).year.transform('nunique')

sale_ratio_switch=sale_ratio_switch.loc[sale_ratio_switch['years_active']>2,:]

direction=sale_ratio_switch.groupby(['BuyerNARC']).cum_sum.min().reset_index()


###########Merging it with accounts
accounts['narc']=accounts['narc'].astype(int)
direction['BuyerNARC']=direction['BuyerNARC'].astype(int)

direction_s=pd.merge(direction, accounts, left_on='BuyerNARC',right_on='narc',how='left')

direction_s.dropna()


direction_s['positive_movement']=np.where(direction_s['cum_sum']>0,1,0)

major_account_pos=direction_s.groupby(['major_account_name']).positive_movement.sum().reset_index()

direction_s['positive_movement'].sum()

################Get baseline for accounts


petcareregion_acc=accounts.groupby(['petcareregion']).narc.nunique().reset_index()
petcareregion_acc.columns=['petcareregion','all_narcs']
#3  TBM



major_account_acc=accounts.groupby(['major_account_name']).narc.nunique().reset_index()
major_account_acc.columns=['major_account_name','all_narcs']


major_account_pos=direction_s.groupby(['major_account_name']).positive_movement.sum().reset_index()

major=pd.merge(major_account_acc,major_account_pos,on=['major_account_name'],how='left')



petcare_pos=direction_s.groupby(['petcareregion']).positive_movement.sum().reset_index()

region=pd.merge(petcareregion_acc,petcare_pos,on=['petcareregion'],how='left')



corp_pos=direction_s.groupby(['corp_flag']).positive_movement.sum().reset_index()

direction_s.positive_movement.sum()



Not_shifting=direction_s.loc[direction_s['positive_movement']==0,:]

Not_shifting.groupby(['petcareregion']).narc.nunique().reset_index()




Not_shifting.groupby(['major_account_name']).narc.nunique().reset_index()



###################


derm2=derm.loc[derm['derm_ratio']>0,:]
derm2['total_sales'].corr(derm2['derm_ratio'],method='spearman')

derm2['Apoquel_sales'].corr(derm2['derm_ratio'],method='spearman')
derm2['Cytopoint_sales'].corr(derm2['derm_ratio'],method='spearman')

#########################################################
#########################################################
##########################################################
##########################################################
#######################################################



#########################################################



acc_lik = 'C:/Users/rajk/Documents/Projects/Association rules/accounts.csv'
accounts = pd.read_csv(acc_lik)

accounts['zip'] = accounts['narczip'].astype(str).str[:5]


ap_only=pd.merge(ly_ap_only,accounts, left_on='buyernarc', right_on='narc',how='left')


ly_ap=pd.merge(derm,accounts, left_on='buyernarc',right_on='narc',how='left')
#1 major account name

major_account_ap=ly_ap.groupby(['major_account_name']).derm_ratio.median().reset_index()
major_account_ap.columns=['major_account_name','derm_ratio']

major_account_ap['better_placed']=np.where(major_account_ap['derm_ratio']>derm['derm_ratio'].median(),1,0)


petcareregion_ap=ly_ap.groupby(['petcareregion']).derm_ratio.median().reset_index()


from uszipcode import SearchEngine, SimpleZipcode, Zipcode
search = SearchEngine(simple_zipcode=True)
zipcode = search.by_zipcode(10030)
zipcode.population

def get_population(zip_code):
    zipcode = search.by_zipcode(zip_code)
    data_list=[[zip_code,zipcode.population,zipcode.population_density,zipcode.median_household_income]]
    #return(data_list)
    return(pd.DataFrame(data_list,columns=['zip_code','population','population_density','median_household_income']))
    

zip_list_all=[]
for zip_code in accounts.zip.unique():
    print(zip_code)
    zip_list=get_population(zip_code)
    zip_list_all.append(zip_list)
    

df = pd.concat(zip_list_all)



accounts_df=pd.merge(accounts,df,left_on='zip',right_on='zip_code',how='left')
accounts_df['population'] = accounts_df['population'].astype(float)
accounts_df.population.describe()

(accounts_df['population'] >= 2900).describe()



import shapefile
from shapely.geometry import Point # Point class
from shapely.geometry import shape # shape() is a function to convert geo objects through the interface

pt = (-117.698,33.5935) # an x,y tuple
shp = shapefile.Reader('C:/Users/rajk/Downloads/cb_2016_us_ua10_500k.shp') #open the shapefile
all_shapes = shp.shapes() # get all the polygons


def is_urban(pt):
    result = False
    for i in range(len(all_shapes)):
        boundary = all_shapes[i] # get a boundary polygon
        #name = all_records[i][3] + ', ' + all_records[i][4] # get the second field of the corresponding record
        if Point(pt).within(shape(boundary)): # make a point and see if it's in the polygon
            result = True
    return result

result = is_urban(pt)

accounts['is_urban']=np.nan
all_narcs=accounts.narc.unique()

accounts_sub=accounts[['narc','longitude','latitude']]
ans_list_all=[]
ans_list=[]
k=0
for i in all_narcs:
    k=k+1
    print(k)
    longitude=accounts.loc[accounts['narc']==i,:].longitude
    latitude=accounts.loc[accounts['narc']==i,:].latitude
    pt=(longitude,latitude)
    ans=is_urban(pt)
    #accounts['is_urban']=np.where(accounts['narc']==i,is_urban(pt),accounts['is_urban'])
    ans_list=[i,ans]
    ans_list_all.append(ans_list)
    #return(pd.DataFrame(ans_list,columns=['longitude','latitude','is_urban']))
    
    
pp=pd.DataFrame(ans_list_all,columns=['narc','is_urban'])


#######################Call distribution




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


rep_calls['ratio']=rep_calls['Cytopoint_calls']/rep_calls['total_calls']

ly_ap_only['buyernarc']=ly_ap_only['buyernarc'].astype(str)
only_ap_calls=pd.merge(ly_ap_only,rep_calls, left_on=['buyernarc'],right_on=['narcid'],how='left') 

only_ap_calls.Cytopoint_calls.hist()

only_ap_calls.Apoquel_calls.describe()


test_ap=only_ap_calls.loc[only_ap_calls['Cytopoint_calls']<=3,:]





derm.head()
derm_sub=derm[['buyernarc','derm_ratio','Cytopoint_sales']]

derm_sub['buyernarc']=derm_sub['buyernarc'].astype(str)

rep_calls['narcid']=(rep_calls['narcid']).astype(str)

derm_calls=pd.merge(derm_sub,rep_calls, left_on=['buyernarc'],right_on=['narcid'],how='left')


derm_calls['Cytopoint_sales'].corr(derm_calls['Cytopoint_calls'],method='spearman')


derm_calls=derm_calls.dropna()



 
##################6000 accounts

ly_ap_only2=pd.merge(ly_ap_only,accounts, left_on='buyernarc',right_on='narc',how='left')



petcareregion_ap_only=ly_ap_only2.groupby(['petcareregion']).narc.nunique().reset_index()

urban_ap_only=ly_ap_only2.groupby(['is_urban']).narc.nunique().reset_index()


major_account_ap_only=ly_ap_only2.groupby(['major_account_name']).narc.nunique().reset_index()
























































#######################Modelling



#Import packages for modeling
from IPython.display import Image  
import sklearn
import pydotplus 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
seed = 42





X=modeling_df.drop('convert_flag',axis=1)
y=modeling_df['convert_flag']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#########Search depth of a tree


min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


#train and evaluating

dt = DecisionTreeClassifier(criterion = "gini", 
                            random_state = seed,
                            max_depth=5, 
                            min_samples_split=0.1
                           )
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
auc = auc(false_positive_rate, true_positive_rate)
auc



# Random forest

#number estimators: number of trees
#max features: the size of the random subsets of features to consider when splitting a nod
#max depth: depth of the tree
#minimum samples split: same as the decision tree, preventing overfitting
#bootstrap: random sampling with replacemen




# Grip Search
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = np.arange(1,15,1)
min_samples_split = np.linspace(0.01, 0.5, 10, endpoint=True)
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap}
rf = RandomForestClassifier() 
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid,
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)


best_params_