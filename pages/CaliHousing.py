import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    
st.markdown("# Cali Housing")

with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

import pandas as pd
housing = pd.read_csv(".\pages\CaliHousing1\housing.csv")

print("The number of rows and colums are {} and also called shape of the matrix".format(housing.shape))
print("Columns names are \n {}".format(housing.columns))



def getOutliers(dataframe,column):
    column = "total_rooms" 
    #housing[column].plot.box(figsize=(8,8))
    des = dataframe[column].describe()
    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}
    Q1 = des[desPairs['25']]
    Q3 = des[desPairs['75']]
    IQR = Q3-Q1
    lowerBound = Q1-1.5*IQR
    upperBound = Q3+1.5*IQR
    st.write("(IQR = {})Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))
    #b = df[(df['a'] > 1) & (df['a'] < 5)]
    data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]

    st.write("Outliers out of total = {} are \n {}".format(housing[column].size,len(data[column])))
    #remove the outliers from the dataframe
    outlierRemoved = housing[~housing[column].isin(data[column])]
    return outlierRemoved

app_mode = st.selectbox('Select Page',['Linear_Regression', 'Decision_Tree_Regression', 'Random_Forest_Regression'
                                                ]) 

#get the outlier
getOutliers(housing,"total_rooms")

#check wheather there are any missing values or null
housing.isnull().sum()

housing_ind = housing.drop("median_house_value",axis=1)
st.write(housing_ind.head())
housing_dep = housing["median_house_value"]
st.write("Medain Housing Values")
st.write(housing_dep.head())

#check for rand_state
X_train,X_test,y_train,y_test = train_test_split(housing_ind,housing_dep,test_size=0.2,random_state=42)
#print(X_train.head())
#print(X_test.head())
#print(y_train.head())
#print(y_test.head())
st.write("X_train shape {} and size {}".format(X_train.shape,X_train.size))
st.write("X_test shape {} and size {}".format(X_test.shape,X_test.size))
st.write("y_train shape {} and size {}".format(y_train.shape,y_train.size))
st.write("y_test shape {} and size {}".format(y_test.shape,y_test.size))



def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value


    

if app_mode=='Decision_Tree_Regression':
    st.title("Decision Tree Regression") 
    from pages.CaliHousing1.Decision_Tree_Regression import *
    import matplotlib.pyplot as plt


    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)

    # Prediction 5 samples 
    st.write("Prediction 5 samples :", tree_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = tree_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình trên tập dữ liệu huấn luyện - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình trên tập dữ liệu kiểm định chéo - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = tree_reg.predict(X_test_prepared)

    st.write('Predict on the test data')
    st.write('Len of y_predictions ', len(y_predictions))
    st.write('Len of y_test ',len(y_test))
    st.write(y_predictions[0:5])
    st.write(y_test[0:5])

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình trên tập dữ liệu kiểm tra - test:')
    st.write('%.2f' % rmse_test)

    test = pd.DataFrame({'Predicted':y_predictions,'Actual':y_test})
    fig= plt.figure(figsize=(16,8))
    test = test.reset_index()
    test = test.drop(['index'],axis=1)
    plt.plot(test[:50])
    plt.legend(['Actual','Predicted'])
    fix = sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
    st.pyplot(fig)
    st.pyplot(fix)



elif app_mode == 'Linear_Regression':
    from pages.CaliHousing1.Linear_Regression import *
    st.title('Linear Regression')
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    st.write("Intercept is "+str(lin_reg.intercept_))
    st.write("coefficients  is "+str(lin_reg.coef_))

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", lin_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = lin_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = lin_reg.predict(X_test_prepared)

    st.write('Predict on the test data')
    st.write('Len of y_predictions ', len(y_predictions))
    st.write('Len of y_test ', len(y_test))
    st.write(y_predictions[0:5])
    st.write(y_test[0:5])

    test = pd.DataFrame({'Predicted':y_predictions,'Actual':y_test})
    fig= plt.figure(figsize=(16,8))
    test = test.reset_index()
    test = test.drop(['index'],axis=1)
    plt.plot(test[:50])
    plt.legend(['Actual','Predicted'])
    fix = sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',)
    st.pyplot(fix)
    st.pyplot(fig)

    
    st.write('Sai số bình phương trung bình - test:')
    st.write(np.sqrt(metrics.mean_squared_error(y_test,y_predictions)))



elif app_mode == 'Random_Forest_Regression':
    st.title('Random Forest Regression')
    from pages.CaliHousing1.Random_Forest_Regression import *
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)


    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", forest_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = forest_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = forest_reg.predict(X_test_prepared)

    st.write('Predict on the test data')
    st.write('Len of y_predictions ', len(y_predictions))
    st.write('Len of y_test ',len(y_test))
    st.write(y_predictions[0:5])
    st.write(y_test[0:5])

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình trên tập dữ liệu kiểm tra - test:')
    st.write('%.2f' % rmse_test)

    test = pd.DataFrame({'Predicted':y_predictions,'Actual':y_test})
    fig= plt.figure(figsize=(16,8))
    test = test.reset_index()
    test = test.drop(['index'],axis=1)
    plt.plot(test[:50])
    plt.legend(['Actual','Predicted'])
    fix = sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
    st.pyplot(fig)
    st.pyplot(fix)