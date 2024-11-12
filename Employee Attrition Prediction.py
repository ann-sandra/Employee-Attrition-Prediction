# -*- coding: utf-8 -*-
"""BIA_project_group8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lMyAJ9y6p8Pn5cRdZhsdqkHEoXDaSNxx
"""

from google.colab import files
uploaded = files.upload()

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import accuracy_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#read the dataset
df = pd.read_csv('IBM HR Data new.csv')
df.head(5)
df.Attrition.value_counts()

#check for null values
df.isnull().sum()[df.isnull().sum()!=0]

Null_values_percentage=(df.isnull().sum().sum()/len(df))*100
Null_values_percentage

df=df.dropna() #Total 1.5% Null values are available In dataset.
#since null values only make up 1.5% of the dataset we drop them
df.isnull().sum()

"""Visualise Outliers"""

# Calculate the frequency of each category
numerical_columns = ['EnvironmentSatisfaction','NumCompaniesWorked','PerformanceRating','JobInvolvement','EnvironmentSatisfaction','NumCompaniesWorked','PerformanceRating','JobInvolvement','StockOptionLevel','YearsAtCompany','YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole']
for category in numerical_columns:
  frequency = df[category].value_counts()
  # Create a bar chart
  plt.bar(frequency.index, frequency.values)
  # Adding labels and title
  plt.xlabel(category)
  plt.ylabel('Frequency')
  plt.title('Frequency Distribution of '+ category)
  # Show the plot
  plt.show()

"""Capping outliers with 90% quantile"""

# Function to replace outliers with the 0.9 quantile (capping)
def replace_outliers_with_quantile(series):
    quantile_90 = series.quantile(0.9)  # Calculate the 0.9 quantile
    for x in series:
      if x > quantile_90:
        x = quantile_90 # Replace outliers with the quantile value
    return series

# Get a list of numerical columns we wanna replace
# the following columns have uneven distribution as seen above so we replace the outliers with 90% quantile
numerical_columns = ['EnvironmentSatisfaction','NumCompaniesWorked','PerformanceRating','JobInvolvement','EnvironmentSatisfaction','NumCompaniesWorked','PerformanceRating','JobInvolvement','StockOptionLevel','YearsAtCompany','YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole']
df1 = pd.DataFrame()
# Loop through numerical columns and replace outliers
for column in numerical_columns:
    df1[column] = replace_outliers_with_quantile(df[column])
    df[column] = df1[column]

"""View cleaned data with capped outliers"""

# Calculate the frequency of each category
numerical_columns = ['EnvironmentSatisfaction','NumCompaniesWorked','PerformanceRating','JobInvolvement','EnvironmentSatisfaction','NumCompaniesWorked','PerformanceRating','JobInvolvement','StockOptionLevel','YearsAtCompany','YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole']
for category in numerical_columns:
  frequency = df[category].value_counts()
  # Create a bar chart
  plt.bar(frequency.index, frequency.values)
  # Adding labels and title
  plt.xlabel(category)
  plt.ylabel('Frequency')
  plt.title('Frequency Distribution of '+ category)
  # Show the plot
  plt.show()

"""Remove Duplicates"""

#remove the same application ID being repeated
has_duplicates = df['Application ID'].duplicated().any()
has_duplicates
df_no_duplicates = df.drop_duplicates(subset='Application ID')
df = df_no_duplicates

"""Correlation heat map

"""

numeric_ = df.select_dtypes(include=['int64', 'float64'])
# Compute the correlation matrix
corr = numeric_.corr()
# Create a heatmap
plt.subplots(figsize=[12, 7])
sns.heatmap(corr, annot=True, mask=corr < 0.3)
# Show the plot
plt.show()

corr=df.corr()
import  seaborn as sns
plt.figure(figsize=[20,15])
sns.heatmap(corr,annot=True,cmap='YlGnBu',fmt='.0%')

df.describe()

#box plot of numericals
numerical_vars = df.select_dtypes(include='number')

# Define the number of subplots per figure
subplots_per_figure = 4

# Calculate the number of figures needed
num_figures = (len(numerical_vars) - 1) // subplots_per_figure + 1
figure_size = (16, 4)  # Adjust the figure size as needed

for figure_number in range(num_figures):
    start_index = figure_number * subplots_per_figure
    end_index = min(start_index + subplots_per_figure, len(numerical_vars))

    # Create a new figure
    plt.figure(figsize=figure_size)

    for i, var in enumerate(numerical_vars.columns[start_index:end_index]):
        plt.subplot(1, subplots_per_figure, i + 1)
        sns.boxplot(x=numerical_vars[var])
        plt.title(var)
        plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

df['Attrition'] = df['Attrition'].astype('category')
# Now, you can create the countplot
sns.countplot(data=df, x='Attrition')
plt.title('Attrition')
# Adjust the figure size if needed
fig = plt.gcf()
fig.set_size_inches(7, 7)
# Show the plot
plt.show()

"""## **Chi Square Test** for understanding the dependence between the independent and target variable. Here we can eliminate the insignificant variables for the further analysis"""

from scipy.stats import chi2_contingency
target_column = 'Attrition'
chi_squared_results = pd.DataFrame(columns=['Variable', 'Chi-Square', 'p-value'])

for column in df.columns:
    if column != target_column and (df[column].dtype == 'object'or df[column].dtype == 'float64'):

        contingency_table = pd.crosstab(df[column], df[target_column])


        chi2, p, _, _ = chi2_contingency(contingency_table)


        chi_squared_results = chi_squared_results.append({'Variable': column, 'Chi-Square': chi2, 'p-value': p},
                                                         ignore_index=True)

significant_variables = chi_squared_results[chi_squared_results['p-value'] <= 0.05]
insignificant_variables = chi_squared_results[chi_squared_results['p-value'] >= 0.05]

print(chi_squared_results)
print("\nSignificant Variables:")
print(significant_variables)
print("\nInsignificant Variables:")
print(insignificant_variables)

"""Removing insignificant variables"""

# remove irrelevant columns as all emplyees are over 18, Standard Hours is 80 for everyone and Employee count is 1 for everyone and Employee number does not affect attrition
df=df.drop(['Over18','EmployeeNumber','StandardHours','EmployeeCount','Application ID','Gender','PerformanceRating','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'],axis=1)

"""# **Travel Type vs Attrition**"""

sns.countplot(data=df, x='BusinessTravel')
plt.xticks(rotation=90)
#Show the plot
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='BusinessTravel', y='Attrition', data=df)

"""# **Department vs Attrition**"""

sns.countplot(data=df, x='Department')
plt.xticks(rotation=90)
#Show the plot
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='Department', y='Attrition', data=df)

"""# **Distance vs Attrition**"""

#convert distance from home to numeric
df.DistanceFromHome = pd.to_numeric(df.DistanceFromHome,errors='coerce')
bins = range(0, 16, 3)
df['DistanceFromHomeBins']=pd.cut(df['DistanceFromHome'], bins=bins, right=False)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='DistanceFromHomeBins')
plt.xticks(rotation=90)
plt.title('Count of Employees by Distance From Home')
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='DistanceFromHomeBins', y='Attrition', data=df)

"""# **Environment Satisfaction vs Attrition**"""

sns.countplot(data=df, x='EnvironmentSatisfaction')
plt.xticks(rotation=90)
#Show the plot
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='EnvironmentSatisfaction', y='Attrition', data=df)

"""# **Job role vs Attrition**"""

sns.countplot(data=df, x='JobRole')
plt.xticks(rotation=90)
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='JobRole', y='Attrition', data=df)
plt.xticks(rotation=90)

"""# **Job Satisfaction vs Attrition**"""

sns.countplot(data=df, x='JobSatisfaction')
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='JobSatisfaction', y='Attrition', data=df)

"""# **Marital Status vs Attrition**"""

sns.countplot(data=df, x='MaritalStatus')
plt.show()
#plt.xticks(rotation=90)

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='MaritalStatus', y='Attrition', data=df)

"""# Percent Salary Hike vs **Attrition**"""

df['PercentSalaryHike'] = pd.to_numeric(df['PercentSalaryHike'], errors='coerce')
bins = range(10, 26, 3)
df['PercentSalaryHikeBins']=pd.cut(df['PercentSalaryHike'], bins=bins, right=False)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PercentSalaryHikeBins')
plt.xticks(rotation=90)
plt.title('Percent Salary Hike')
plt.show()

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='PercentSalaryHikeBins', y='Attrition', data=df)

"""# **Number of Companies worked vs Attrition**"""

sns.countplot(data=df, x='NumCompaniesWorked')
plt.show()
#plt.xticks(rotation=90)

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='NumCompaniesWorked', y='Attrition', data=df)

sns.countplot(data=df, x='OverTime')
plt.show()
#plt.xticks(rotation=90)

df['Attrition'] = df['Attrition'].astype('category')
df['Attrition'] = df['Attrition'].cat.codes
sns.barplot(x='OverTime', y='Attrition', data=df)

plt.figure(figsize=(5,5))
sns.barplot(data=df, x='JobLevel', y='YearsAtCompany')
# Rotate the x-axis labels if needed
plt.xticks(rotation=90)
# Set labels and title
plt.xlabel('Job Level')
plt.ylabel('Years at Company')
plt.title('Years at Company by Job Level')
# Show the plot
plt.show()

filtered_df = df[df['Attrition'] == 'Voluntary Resignation']
plt.figure(figsize=(20, 18))
sns.lmplot(x='Age', y='DailyRate', hue='Attrition', data=df)
plt.title('Relationship between Age and DailyRate with [Attrition]')
plt.xlabel('Age')
plt.ylabel('DailyRate')
plt.legend(title='[Attrition]')
plt.show()

"""# **Clean the data by removing false labels**"""

df.info()

"""# **Categorical -> Numerical**"""

def attrition(x):
    if x=='Voluntary Resignation':
        x=1
    else:
        x=0
    return x
df.Attrition=df.Attrition.apply(attrition)
df.Attrition = pd.to_numeric(df.Attrition,errors='coerce')
df.Attrition.value_counts()

df.EducationField.value_counts()

#group test with other
def edufield(x):
    if  x=='Test':
        x='Other'
    return x
df.EducationField=df.EducationField.apply(edufield)

#convert numerics in object to numeric data type
df.HourlyRate.value_counts()
df.HourlyRate = pd.to_numeric(df.HourlyRate,errors='coerce')
df.JobSatisfaction = pd.to_numeric(df.JobSatisfaction,errors='coerce')
df.MonthlyIncome = pd.to_numeric(df.MonthlyIncome,errors='coerce')
df.DistanceFromHome = pd.to_numeric(df.DistanceFromHome,errors='coerce')

def overtime(x):
    if x=='Yes':
        x=1
    elif x=='No':
        x=0
    return x
df.OverTime=df.OverTime.apply(overtime)
df.OverTime = pd.to_numeric(df.OverTime,errors='coerce')

df.info()

df.PercentSalaryHike.value_counts()
df.PercentSalaryHike = pd.to_numeric(df.PercentSalaryHike,errors='coerce')

df['Employee Source'].value_counts()

def empsou(x):
    if x=='Test':
        x='Referral'
    return x
df['Employee Source']= df['Employee Source'].apply(empsou)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Initialize the LabelEncoder
label_encoder = LabelEncoder()
categorical_columns

# Apply label encoding to each categorical column
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

"""# **Save cleaned file**"""

df.to_csv('HR_Analyst_File.csv', index=False)

"""# **Test train split**"""

y=df['Attrition']
X = df.drop(columns=['Attrition'])

X.head()
y.head()

y.value_counts()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, stratify=y, random_state=0)

"""# **Model training**"""

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve
def model_eval(algo,xtrain,xtest,ytrain,ytest):
    algo.fit(xtrain,ytrain)
    y_train_pred=algo.predict(xtrain)
    y_train_prob=algo.predict_proba(xtrain)[:,1]

    y_test_pred=algo.predict(xtest)
    y_test_prob=algo.predict_proba(xtest)[:,1]
    print("MODEL USED FOR CLASSIFICATION :"+ algo)
    print('Confusion Matrix-Train:\n'+confusion_matrix(ytrain,y_train_pred))
    print('Accuracy Score-Train:\n'+accuracy_score(ytrain,y_train_pred))
    print('Classification Report-Train:\n'+classification_report(ytrain,y_train_pred))
    print('AUC Score-Train:\n'+roc_auc_score(ytrain,y_train_prob))
    print('\n')
    print('Confusion Matrix-Test:\n'+confusion_matrix(ytest,y_test_pred))
    print('Accuracy Score-Test:\n'+accuracy_score(ytest,y_test_pred))
    print('Classification Report-Test:\n'+classification_report(ytest,y_test_pred))
    print('AUC Score-Test:\n'+roc_auc_score(ytest,y_test_prob))
    print('\n')
    print('Plot')
    fpr,tpr,thresholds= roc_curve(ytest,y_test_prob)
    fig,ax1 = plt.subplots()
    ax1.plot(fpr,tpr)
    ax1.plot(fpr,fpr)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax2=ax1.twinx()
    ax2.plot(fpr,thresholds,'-g')
    ax2.set_ylabel('TRESHOLDS')
    plt.show()
    print('-x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x-')

lr=LogisticRegression()
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
knn=KNeighborsClassifier()
rf=RandomForestClassifier()

dt=DecisionTreeClassifier()


models=[]
models.append(('MVLC',lr))
models.append(('RFC',rf))
models.append(('DT',dt))
models.append(('KNNC',knn))

results=[]
names=[]
ypred=[]
for name,model in models:
    model.fit(X_train,y_train)
    ypred= model.predict(X_test)
    print(name,'\n:')
    print(classification_report(y_test,ypred))
    kfold=KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)"%(name,np.mean(cv_results)*100,np.var(cv_results,ddof=1)))
    print('-x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x-')

"""# Plot the important features for models"""

#decision tree and random forest important features
models=[dt,rf]
for i in models:
    i.fit(X,y)
    i.feature_importances_
    print(i)
    #Plot the data:
    #my_colors = 'rgbkymc'  #red, green, blue, black, etc.
    feature_ranks = pd.Series(i.feature_importances_,index=X.columns)
    plt.figure(figsize =(10,10))
    feature_ranks.nlargest(8).sort_values(ascending=True).plot(kind='barh')

    plt.show()

feature_names = df.columns.tolist()
# Get the coefficients (weights) of the features
feature_importance = abs(lr.coef_[0])

# Print feature importance scores
for i, feature_name in enumerate(feature_names):
    print(f"Feature: {feature_name}, Importance: {feature_importance[i]}")

"""# **Find optimal alpha for decision tree**"""

# find optimal alpha for Decision tree
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#Using matplotlib.pyplot to plot the effect of varying ccp_alpha on error
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1],impurities[:-1],marker='o', drawstyle="steps-post")
#blue lines drowan to highlight step changes; orange is the graph
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.plot(ccp_alphas, impurities)

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post", color = "#0097b2")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post",  color = "#0097b2")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post", color = "#00B287")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post", color = "#0097b2")
ax.legend()
plt.show()

ccp_alpha_df = pd.DataFrame({"ccp_alpha":ccp_alphas, "test_accuracy":test_scores, "train_accuracy":train_scores, "max_depth":depth, "node_counts":node_counts})
ccp_alpha_df
diff= [a-b for a,b in zip(train_scores,test_scores)]
minimum=min(diff)
print(diff.index(minimum))
#Here both test and train accuracy goes in the same direction as each other. we shall use ccp_alpha as 0.003637

ccp_alpha_df.iloc[[224]]

"""# **Final decision tree with optimal apha**"""

#FINAL MODEL BUILDING
dt=DecisionTreeClassifier(criterion='gini',ccp_alpha=0.001611		)
dt.fit(X_train, y_train)
y_pred= dt.predict(X_test)
y_train_pred = dt.predict(X_train)
print(f'Model training with train data, fitting accuracy is : {(accuracy_score(y_train, y_train_pred))}')
print(f'Model accuracy score with test data : {(accuracy_score(y_test, y_pred))}')
confusion_matrix(y_pred, y_test)

"""# **Choose n for KNN**"""

param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
accuracy=[]

for i in range(1,40):
  # Initialize the KNN classifier
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  y_pred_i= knn.predict(X_test)
  accuracy.append(accuracy_score(y_test, y_pred_i))

plt.figure(figsize=(8, 6))
plt.plot(range(1,40),accuracy, color='blue', linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('K vs Accuracy in KNN')

# Define a range of values for k (number of neighbors)
k_values = np.arange(1, 40)  # You can adjust the range as needed

# Initialize lists to store accuracy scores for training and test sets
train_scores = []
test_scores = []

# Iterate over different values of k
for k in k_values:
    # Create and train a KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions on the training and test sets
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    # Calculate accuracy for the training and test sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Append the accuracy scores to the respective lists
    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, marker='o', label='Train Set')
plt.plot(k_values, test_scores, marker='o', label='Test Set')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

"""# **Final KNN Model**"""

#FINAL MODEL BUILDING

knn = KNeighborsClassifier(n_neighbors=1)


# Train the KNN classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with 11 neighbors: {accuracy:.2f}")
knn.n_neighbors

"""# **Finetuning random forest**

"""

from sklearn.model_selection import RandomizedSearchCV
rfrs_cv=RandomForestClassifier(max_depth=20, min_samples_leaf=10, min_samples_split=20,
                       n_estimators=50)
rfrs_cv.fit(X_train,y_train)
y_train_pred=rfrs_cv.predict(X_train)
y_train_prob=rfrs_cv.predict_proba(X_train)[:,1]

y_test_pred=rfrs_cv.predict(X_test)
y_test_prob=rfrs_cv.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
thresholds[0] = thresholds[0]-1
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr,label='ROC CURVE')
ax1.plot(fpr,fpr,label='AUC CURVE')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
plt.legend(loc='best')
ax2=ax1.twinx()
plt.show()

"""# **Final Accuracy Check**"""

lr=LogisticRegression()
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
knn=KNeighborsClassifier(n_neighbors=1)

rf=RandomForestClassifier(max_depth=20, min_samples_leaf=10, min_samples_split=20,
                       n_estimators=50)

dt=DecisionTreeClassifier(criterion='gini',ccp_alpha=0.001611)


models=[]
models.append(('MVLC',lr))
models.append(('RFC',rf))
models.append(('DT',dt))
models.append(('KNNC',knn))

results=[]
names=[]
ypred=[]
for name,model in models:
    model.fit(X_train,y_train)
    ypred= model.predict(X_test)
    print(name,'\n:')
    print(classification_report(y_test,ypred))
    kfold=KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)"%(name,np.mean(cv_results)*100,np.var(cv_results,ddof=1)))
    print('Confusion Matrix-Test\n',confusion_matrix(y_test,ypred))
    print('-x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x--x-x-x-x-x-')