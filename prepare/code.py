import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


data = pd.DataFrame(pd.read_csv("/home/jarviszly/Desktop/prepare/sample_train.csv"))

##print(data.head(1))

ID = data.pop('id')

##print(ID)

##print(data.head(1))

train_X=data.values[0::,0:5]
train_y=data.values[0::,5]

##print(train_X)
##print(train_y)

##X_tr,X_te,y_tr,y_te=train_test_split(train_X ,train_y,test_size=0.3,random_state=0)
X_train, X_test, y_train, y_test= train_test_split(train_X, train_y, test_size=0.3)

##print(y_train)



def M_squared(resource,data):
    L=len(resource)
    index=0
    sum=0
    while index < L:
        sum = sum + (resource[index]-data[index])*(resource[index]-data[index])
        index = index + 1
        # print(sum)
        # print(index)
    return sum/L


##### KNeighbors

# Knn = KNeighborsRegressor()
# Knn.fit(X_train,y_train)
#
# print(Knn.predict(X_test))
# print(y_test)
#
# res = M_squared(Knn.predict(X_test),y_test)
#
# print(res)


##### RandomForest

# rfr = RandomForestRegressor()
# rfr.fit(X_test,y_test)
# print(rfr.predict(X_test))
# print(y_test)

# res = M_squared(rfr.predict(X_test),y_test)
#
# print(res)


##### ExtraTrees

# etr = ExtraTreesRegressor()
# etr.fit(X_test,y_test)
# print(etr.predict(X_test))
# print(y_test)

# res = M_squared(etr.predict(X_test),y_test)
#
# print(res)


##### GrandientBoosting

# gbr = GradientBoostingRegressor()
# gbr.fit(X_test,y_test)
# print(gbr.predict(X_test))
# print(y_test)
#
# res = M_squared(gbr.predict(X_test),y_test)
#
# print(res)




