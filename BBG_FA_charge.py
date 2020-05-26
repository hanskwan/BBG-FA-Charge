import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

import matplotlib
import matplotlib.pyplot as plt
import missingno as msno

# Linear regression module
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# K Means
from sklearn.cluster import KMeans 

# Importing the dataset
df_0 = pd.read_csv("Main_S1_Nov2019.csv")
df_1 = df_0.drop(df_0.columns[1:8],axis=1) # remove time
df_1.head()
df_1["GICS_SECTOR_NAME"].value_counts().plot(kind="bar")

msno.matrix(df_1)
msno.bar(df_1)

#define function to indicate missing value
def mis_val_info(dataset):
    zero_val = (dataset == 0).sum(axis=0)
    mis_val  = dataset.isnull().sum()
    mis_value_percent = 100 * mis_val/len(dataset)
    mis_table = pd.DataFrame({"Missing Zero": zero_val/len(dataset),
                              "Missing Value" : mis_value_percent})
    mis_table.sort_values("Missing Value", inplace = True)
    return(mis_table, type(mis_table))

# Find column with more than 60% missing value
NAN_perct = mis_val_info(df_1)
NAN_perct

# Delete all variables with more than 60% of missing value, output useful dataset, inplace
df_2 = df_1.drop([
        "CFF_ACTIVITIES_DETAILED",
        "OTHER_INVESTING_ACT_DETAILED",
        "CASH_CONVERSION_CYCLE",
        "IS_NONOP_INCOME_LOSS",
        "PROC_FR_REPAYMNTS_BOR_DETAILED",
        "CF_FREE_CASH_FLOW",
        "CHG_IN_FXD_&_INTANG_AST_DETAILED",
        "PROC_FR_REPURCH_EQTY_DETAILED",
        "CF_DVD_PAID",
        "NET_CHG_IN_LT_INVEST_DETAILED",
        "CF_OTHER_FINANCING_ACT_EXCL_FX",
        "CF_FREE_CASH_FLOW_FIRM",
        "CF_NT_CSH_RCVD_PD_FOR_ACQUIS_DIV",
        "NON_CASH_ITEMS_DETAILED",
        "CF_INTEREST_RECEIVED",
        "CF_CHNG_NON_CASH_WORK_CAP",
        "CF_NET_CASH_PAID_FOR_AQUIS"],axis = 1)

# Removed rows without GICS since it can't be imputed by KNN
df_3 = df_2[pd.notnull(df_2["GICS_SECTOR_NAME"])]
df_3_allnum = df_3.iloc[:,1:49]
df_3_3col = df_3.iloc[:,1:4]
df_3_Y = df_3.iloc[:,-1]
df_3_1 = pd.DataFrame(df_3_Y).reset_index(drop=True)

sns.distplot((df_3["IS_ABNORMAL_ITEM"]))
sns.distplot(df_3["IS_ABNORMAL_ITEM"], bins=20, kde=False, rug=True);


# Impute NAN data point with KNN
imputer_5 = KNNImputer(n_neighbors = 5)
df_clu_1 = imputer_5.fit_transform(df_3_3col)
df_clu_2 = pd.DataFrame(df_clu_1)
df_4 = imputer_5.fit_transform(df_3_allnum)


# K means
kmeans_c5 = KMeans(n_clusters=5)
kmeans_c5.fit(df_clu_2)
lab = kmeans_c5.predict(df_clu_2)
centroids = kmeans_c5.cluster_centers_


colors = map(lambda x: colmap[x+1], lab)
colors1 = list(colors)
for idx, centroid in enumerate(centroids):
    plt.scatter(cnetriods, y, kwargs)


# print name
for col in df_3_allnum.columns:
    print(col)


df_4_1 = pd.DataFrame(df_4)
df_4_1.columns = ["SALES_REV_TURN","GROSS_PROFIT","IS_OTHER_OPER_INC","IS_OPERATING_EXPN","IS_OPER_INC",
"IS_INT_EXPENSE","IS_INT_INC","PRETAX_INC","IS_ABNORMAL_ITEM","IS_INC_BEF_XO_ITEM","NI_INCLUDING_MINORITY_INT_RATIO",
"MIN_NONCONTROL_INTEREST_CREDITS",
"IS_EPS","IS_DILUTED_EPS","EBITDA","GROSS_MARGIN","OPER_MARGIN","PROF_MARGIN","EQY_DPS",
"C&CE_AND_STI_DETAILED","BS_ACCT_NOTE_RCV",
"BS_INVENTORIES","OTHER_CURRENT_ASSETS_DETAILED","BS_CUR_ASSET_REPORT",
"BS_NET_FIX_ASSET","BS_LT_INVEST","BS_OTHER_ASSETS_DEF_CHRG_OTHER",
"BS_TOT_NON_CUR_ASSET","ACCT_PAYABLE_&_ACCRUALS_DETAILED","BS_ST_BORROW",
"OTHER_CURRENT_LIABS_SUB_DETAILED","BS_CUR_LIAB","BS_LT_BORROW",
"OTHER_NONCUR_LIABS_SUB_DETAILED","NON_CUR_LIAB","BS_TOT_LIAB2","BS_SH_CAP_AND_APIC",
"BS_PURE_RETAINED_EARNINGS","EQTY_BEF_MINORITY_INT_DETAILED",
"MINORITY_NONCONTROLLING_INTEREST","TOTAL_EQUITY","BS_SH_OUT","NET_DEBT","NET_DEBT_TO_SHRHLDR_EQTY",
"CF_DEPR_AMORT","CF_CASH_FROM_OPER",
"CF_CASH_FROM_INV_ACT","CF_NET_CHNG_CASH"]

df_5 = df_4_1.join(df_3_1,lsuffix = "Revenue", rsuffix = "GICS_SECTOR_NAME")
df_5.head()

# creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')

# passing column (label encoded values of bridge_types)
df_GICS_encoded = pd.DataFrame(encoder.fit_transform(df_5[['GICS_SECTOR_NAME']]).toarray())
GICS_name = df_5['GICS_SECTOR_NAME'].unique()
print(GICS_name)
df_GICS_encoded.columns = ['Industrials', 'Utilities', 'Real Estate', 'Communication Services',
 'Financials', 'Consumer Discretionary', 'Consumer Staples',
 'Information Technology', 'Materials', 'Energy', 'Health Care']

df_6 = df_5.iloc[:,0:48].join(df_GICS_encoded)

# Reorder the target column to last col
df = df_6[[c for c in df_6 if c not in ["IS_ABNORMAL_ITEM"]] + ["IS_ABNORMAL_ITEM"]]
df.head()
msno.matrix(df)


####### Model Part

### Seperate train and test set
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# EDA
sns.distplot(y, hist = False, rug = True)
sns.distplot(y, kde = False, rug = True)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

#### 
# linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)

lr_train_score =lr.score(X_train, y_train)
lr_test_score =lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

reg_table = {
        'Train' : [lr_train_score,Ridge_train_score,Ridge_train_score100],
        'Test': [lr_test_score,Ridge_test_score,Ridge_test_score100]
        }

reg_result = pd.DataFrame(reg_table)
reg_result


####
# MLP model

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
MLP_1 = Sequential()
MLP_1.add(Dense(32,input_dim = 50, activation = 'relu'))
MLP_1.add(Dense(units = 64, activation = 'relu'))
MLP_1.add(Dense(units = 32, activation = 'relu'))
# output layer
MLP_1.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
MLP_1.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
MLP_1.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Evaluate performance
_, accuracy = MLP_1.evaluate(X_test,y_test)
print('Accuracy: %.2f' % (accuracy*100))

# MLP result
y_pred = model.predict(X_test)


####
# XGboost
import xgboost as xgb
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 
XGB_steps = 20  # The number of training iterations

XGB_model = xgb.train(param, D_train, XGB_steps)

from sklearn.metrics import precision_score, recall_score, accuracy_score

XGB_preds = XGB_model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in XGB_preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))

from sklearn import datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
