#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd                 
import matplotlib.pyplot as plt       
import seaborn as sns                 

credit_risk_data = pd.read_csv("C:\\Users\\User\\Downloads\\credit_risk_dataset (1).csv")
credit_risk_data.head()


# In[4]:


credit_risk_data.describe()


# In[5]:


credit_risk_data.info()


# In[6]:


categorical_columns = credit_risk_data.select_dtypes(include=['object']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_columns):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=col, data=credit_risk_data)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()


# In[7]:


numeric_columns = credit_risk_data.select_dtypes(include=['int64', 'float64']).columns
numeric_data = credit_risk_data[numeric_columns]

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_data.columns):
    plt.subplot(3, 4, i+1)
    sns.histplot(numeric_data[col], kde=True)
    plt.title(col)
    plt.tight_layout()
plt.show()


# In[8]:


missing_values = credit_risk_data.isnull().sum()

print(missing_values)


# In[9]:


credit_risk_data['person_emp_length'].fillna(credit_risk_data['person_emp_length'].median(), inplace=True)
credit_risk_data['loan_int_rate'].fillna(credit_risk_data['loan_int_rate'].median(), inplace=True)

missing_values = credit_risk_data.isnull().sum()

print(missing_values)


# In[10]:


credit_risk_data['person_age'] = credit_risk_data['person_age'].apply(lambda x: min(x, 75))
credit_risk_data['person_emp_length'] = credit_risk_data['person_emp_length'].apply(lambda x: min(x, 40))

plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
sns.histplot(credit_risk_data['person_age'], kde=True)
plt.title('person_age')
    
plt.subplot(1, 2, 2)
sns.histplot(credit_risk_data['person_emp_length'], kde=True)
plt.title('person_emp_length')    
    
    
plt.tight_layout()
plt.show()


# In[11]:


credit_risk_data = pd.get_dummies(credit_risk_data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

credit_risk_data.info()


# In[12]:


credit_risk_data['income_to_loan_ratio'] = credit_risk_data['loan_amnt'] / credit_risk_data['person_income']


sns.histplot(credit_risk_data['income_to_loan_ratio'], kde=True)
plt.title('income_to_loan_ratio')    
plt.show()


# In[13]:


credit_risk_data.to_csv("processed_data.csv")


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data = pd.read_csv('processed_data.csv')

X = data.drop(['loan_status', 'Unnamed: 0'], axis=1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model Doğruluğu: {accuracy}')

report = classification_report(y_test, y_pred)
print(report)


# In[15]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred_mnb = mnb.predict(X_test)

accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print(f'MultinomialNB Model Doğruluğu: {accuracy_mnb}')

report_mnb = classification_report(y_test, y_pred_mnb)
print(report_mnb)


# In[16]:


from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred_bnb = bnb.predict(X_test)

accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
print(f'BernoulliNB Model Doğruluğu: {accuracy_bnb}')

report_bnb = classification_report(y_test, y_pred_bnb)
print(report_bnb)


# In[17]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


fpr_gnb, tpr_gnb, _ = roc_curve(y_test, gnb.predict_proba(X_test)[:,1])
fpr_mnb, tpr_mnb, _ = roc_curve(y_test, mnb.predict_proba(X_test)[:,1])
fpr_bnb, tpr_bnb, _ = roc_curve(y_test, bnb.predict_proba(X_test)[:,1])


auc_gnb = auc(fpr_gnb, tpr_gnb)
auc_mnb = auc(fpr_mnb, tpr_mnb)
auc_bnb = auc(fpr_bnb, tpr_bnb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_gnb, tpr_gnb, label=f'GaussianNB (AUC = {auc_gnb:.2f})')
plt.plot(fpr_mnb, tpr_mnb, label=f'MultinomialNB (AUC = {auc_mnb:.2f})')
plt.plot(fpr_bnb, tpr_bnb, label=f'BernoulliNB (AUC = {auc_bnb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri')
plt.legend(loc="lower right")
plt.show()


# In[18]:


import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {'var_smoothing': np.logspace(0, -9, num=1000)}

gnb = GaussianNB()
gnb_grid = GridSearchCV(gnb, param_grid, cv=5, scoring='accuracy')

gnb_grid.fit(X_train, y_train)

print(f"En İyi Parametre: {gnb_grid.best_params_}")
print(f"En İyi Skor: {gnb_grid.best_score_}")

y_pred_optimized_gnb = gnb_grid.predict(X_test)
optimized_gnb_accuracy = accuracy_score(y_test, y_pred_optimized_gnb)
print(f"Optimize Edilmiş GaussianNB Model Doğruluğu: {optimized_gnb_accuracy}")


# In[19]:


import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {'var_smoothing': np.logspace(0, -9, num=1000)}

gnb = GaussianNB()
gnb_grid = GridSearchCV(gnb, param_grid, cv=5, scoring='accuracy')

gnb_grid.fit(X_train, y_train)

print(f"En İyi Parametre: {gnb_grid.best_params_}")
print(f"En İyi Skor: {gnb_grid.best_score_}")

y_pred_optimized_gnb = gnb_grid.predict(X_test)
optimized_gnb_accuracy = accuracy_score(y_test, y_pred_optimized_gnb)
print(f"Optimize Edilmiş GaussianNB Model Doğruluğu: {optimized_gnb_accuracy}")


# In[20]:


from sklearn.model_selection import GridSearchCV

param_grid_mnb = {
    'alpha': np.linspace(0.1, 2, 20),
    'fit_prior': [True, False]
}

mnb = MultinomialNB()
mnb_grid = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='accuracy')

mnb_grid.fit(X_train, y_train)

print(f"En İyi Parametreler: {mnb_grid.best_params_}")
print(f"En İyi Skor: {mnb_grid.best_score_}")

y_pred_optimized_mnb = mnb_grid.predict(X_test)
optimized_mnb_accuracy = accuracy_score(y_test, y_pred_optimized_mnb)
print(f"Optimize Edilmiş MultinomialNB Model Doğruluğu: {optimized_mnb_accuracy}")


# In[21]:


from sklearn.model_selection import GridSearchCV

param_grid_bnb = {
    'alpha': np.linspace(0.1, 2, 20),
    'binarize': np.linspace(0.0, 1.0, 10)
}

bnb = BernoulliNB()
bnb_grid = GridSearchCV(bnb, param_grid_bnb, cv=5, scoring='accuracy')

bnb_grid.fit(X_train, y_train)

print(f"En İyi Parametreler: {bnb_grid.best_params_}")
print(f"En İyi Skor: {bnb_grid.best_score_}")

y_pred_optimized_bnb = bnb_grid.predict(X_test)
optimized_bnb_accuracy = accuracy_score(y_test, y_pred_optimized_bnb)
print(f"Optimize Edilmiş BernoulliNB Model Doğruluğu: {optimized_bnb_accuracy}")


# In[ ]:




