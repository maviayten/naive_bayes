#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


df = pd.read_csv("C:\\Users\\\\User\\\Downloads\\IMDB Dataset.csv")


df= df.sample(n=500, random_state=42)

print(df.head())


# In[7]:


print(df.info())


# In[8]:


print(df['sentiment'].value_counts())


# In[9]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

X = df['cleaned_review']
y = df['sentiment']

count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(X)

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

X_count_df = pd.DataFrame(X_count.toarray(), columns=count_vectorizer.get_feature_names_out())
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("Count Vectorization Örnek Özellikler:")
print(X_count_df.head())

print("\nTF-IDF Vectorization Örnek Özellikler:")
print(X_tfidf_df.head())


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['sentiment'], test_size=0.2, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_count, df['sentiment'], test_size=0.2, random_state=42)


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

X_train_dense = X_train.todense()
X_test_dense = X_test.todense()

X_train_array = np.asarray(X_train_dense)
X_test_array = np.asarray(X_test_dense)

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_array)
X_test_scaled = scaler.transform(X_test_array)

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred_gnb = gnb.predict(X_test_scaled)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("GaussianNB Performansı:")
print(f"Doğruluk: {accuracy_gnb}")
print(classification_report(y_test, y_pred_gnb))


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

X_train_dense2 = X_train2.todense()
X_test_dense2 = X_test2.todense()

X_train_array2 = np.asarray(X_train_dense2)
X_test_array2 = np.asarray(X_test_dense2)

scaler = MaxAbsScaler()
X_train_scaled2 = scaler.fit_transform(X_train_array2)
X_test_scaled2 = scaler.transform(X_test_array2)

gnb = GaussianNB()
gnb.fit(X_train_scaled2, y_train2)
y_pred_gnb2 = gnb.predict(X_test_scaled2)
accuracy_gnb2 = accuracy_score(y_test2, y_pred_gnb2)
print("GaussianNB Performansı (Count Vectorization):")
print(f"Doğruluk: {accuracy_gnb2}")
print(classification_report(y_test2, y_pred_gnb2))


# In[13]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred_mnb = mnb.predict(X_test)
accuracy_mnb_tfidf = accuracy_score(y_test, y_pred_mnb)

print("MultinomialNB Performansı (TF-IDF):")
print(f"Doğruluk: {accuracy_mnb_tfidf}")
print(classification_report(y_test, y_pred_mnb))


# In[14]:


mnb2 = MultinomialNB()
mnb2.fit(X_train2, y_train2)

y_pred_mnb2 = mnb2.predict(X_test2)
accuracy_mnb_count = accuracy_score(y_test2, y_pred_mnb2)

print("MultinomialNB Performansı (Count Vectorization):")
print(f"Doğruluk: {accuracy_mnb_count}")
print(classification_report(y_test2, y_pred_mnb2))


# In[15]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


bnb_tfidf = BernoulliNB()
bnb_tfidf.fit(X_train, y_train)


y_pred_bnb_tfidf = bnb_tfidf.predict(X_test)
accuracy_bnb_tfidf = accuracy_score(y_test, y_pred_bnb_tfidf)

print("BernoulliNB Performansı (TF-IDF):")
print(f"Doğruluk: {accuracy_bnb_tfidf}")
print(classification_report(y_test, y_pred_bnb_tfidf))


# In[16]:


bnb_count = BernoulliNB()
bnb_count.fit(X_train2, y_train2)


y_pred_bnb_count = bnb_count.predict(X_test2)
accuracy_bnb_count = accuracy_score(y_test2, y_pred_bnb_count)

print("BernoulliNB Performansı (Count Vectorization):")
print(f"Doğruluk: {accuracy_bnb_count}")
print(classification_report(y_test2, y_pred_bnb_count))


# In[18]:


from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

alpha_range1 = np.linspace(0, 2, 100)
alpha_range2 = np.linspace(2, 4, 100)
alpha_range3 = np.linspace(4, 6, 100)
alpha_range4 = np.linspace(6, 8, 100)

alpha_values = np.concatenate((alpha_range1, alpha_range2, alpha_range3, alpha_range4))

param_grid = {'alpha': alpha_values}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train2, y_train2)

print("En iyi alpha değeri:", grid_search.best_params_)
print("Optimize edilmiş MultinomialNB modelinin doğruluğu:", grid_search.best_score_)

y_pred_optimized = grid_search.predict(X_test2)
print("Optimize edilmiş MultinomialNB Performansı (Count Vectorization):")
print(classification_report(y_test2, y_pred_optimized))


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Lojistik Regresyon Modeli Performansı:")
print(f"Doğruluk: {accuracy_lr}")
print(classification_report(y_test, y_pred_lr))


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train2, y_train2)

y_pred_lr2 = log_reg.predict(X_test2)

accuracy_lr2 = accuracy_score(y_test2, y_pred_lr2)
print("Lojistik Regresyon Modeli Performansı:")
print(f"Doğruluk: {accuracy_lr2}")
print(classification_report(y_test2, y_pred_lr2))


# In[21]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-En Yakın Komşu Modeli Performansı:")
print(f"Doğruluk: {accuracy_knn}")
print(classification_report(y_test, y_pred_knn))


# In[22]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train2, y_train2)


y_pred_knn2 = knn.predict(X_test2)

accuracy_knn2 = accuracy_score(y_test2, y_pred_knn2)
print("K-En Yakın Komşu Modeli Performansı:")
print(f"Doğruluk: {accuracy_knn2}")
print(classification_report(y_test2, y_pred_knn2))


# In[ ]:




