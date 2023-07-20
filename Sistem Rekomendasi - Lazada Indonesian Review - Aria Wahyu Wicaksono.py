#!/usr/bin/env python
# coding: utf-8

# Name: Aria Wahyu Wicaksono
# 
# # Content-Based Filtering and Collaborative Filtering for Lazada Indonesian Review
# 
# Recommend top 5 items for user

# ## 1. Data Loading

# In[2]:


get_ipython().system('sudo pip install -q imbalanced-learn')

from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df_items = pd.read_csv('/kaggle/input/lazada-indonesian-reviews/20191002-items.csv')
df_users = pd.read_csv("/kaggle/input/lazada-indonesian-reviews/20191002-reviews.csv")


# ## 2. Data Understanding
# ### 2.1. Missing and Duplicate Value
# 
# Remove duplicate itemID on `df_items`, user name on `df_users`, remove guest account (Lazada Customer and Lazada Guest), and remove any NaN values on itemID, user name, and user rating.

# In[4]:


df_items = df_items.drop_duplicates(subset="itemId", keep='first')
df_items.head()


# In[5]:


df_items.shape


# In[6]:


df_users['old_name'] = df_users['name']
df_users['name'] = df_users['name'].str.lower().str.replace(".", "", regex=False).str.strip()
df_users = df_users.drop(df_users[df_users['name'] == "lazada customer"].index)
df_users = df_users.drop(df_users[df_users['name'] == "lazada guest"].index)


# In[7]:


df_users = df_users.dropna(how='any', subset=["itemId", "name", "rating"])


# In[8]:


df_users.head()


# In[9]:


df_users.shape


# In[10]:


df_users[["itemId", "name", "rating"]].head()


# ### 2.2. Univariate Analysis for Ratings' Column

# In[11]:


df_users['rating'].describe()


# It looks that most ratings are 5 stars. To visualize it, plot histogram of rating's distribution.

# In[12]:


plt.hist(df_users['rating'], bins=range(1,7), align='left')
plt.title("Rating's Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.xticks(range(1,6))
plt.show()


# In[13]:


print("Actual shape:", df_users.shape[0])


# ## 3. Data Preparation

# ### 3.1. User ID and Item ID Encoding

# In[14]:


item_encoder = {x: i for i, x in enumerate(df_items['itemId'].unique().tolist())}
user_encoder = {x: i for i, x in enumerate(df_users['name'].unique().tolist())}
item_decoder = {i: x for i, x in enumerate(df_items['itemId'].unique().tolist())}
user_decoder = {i: x for i, x in enumerate(df_users['name'].unique().tolist())}
item_id_to_name = {df_items.loc[i, 'itemId']: df_items.loc[i, 'name'] for i in df_items.index}


# In[15]:


df_users['item'] = df_users['itemId'].map(item_encoder)
df_users['user'] = df_users['name'].map(user_encoder)


# In[16]:


df_users[['item', 'user']].head(10)


# ### 3.2. Data Splitting and Balancing

# In[17]:


X = df_users[['user', 'item']].values
y = df_users['rating']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=83)
y_train.value_counts()


# 100,397 out of 153,593 are labelled as 5 stars. Use SMOTE-ENN sampling (combination of oversampling and undersampling) to balance the distribution.

# In[18]:


oversample = SMOTEENN(random_state=83)
X_train, y_train = oversample.fit_resample(X_train, y_train)
y_train.value_counts()


# In[19]:


X_train


# In[20]:


y_train


# ### 3.3. Data Normalization

# Normalize ratings into (0, 1) because the model will be using sigmoid function for the output.

# In[21]:


scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler.fit_transform(np.reshape(y_train.values, (-1,1)))
y_val_scaled = scaler.transform(np.reshape(y_val.values, (-1,1)))


# In[22]:


y_val_scaled


# ### 3.4. Text Preprocessing

# In[23]:


stop_words = set(nltk.corpus.stopwords.words('indonesian'))

def preprocess_text(text, stop_words=stop_words):
    text = text.lower()
    text = re.sub(r'[^\w\s]|_', "", text)
    text = text.strip()
    text = re.sub(r'\s+', " ", text)
    text =  ' '.join(w for w in text.split() if w not in stop_words)
    return text

df_items['clean_name'] = df_items['name'].apply(preprocess_text)


# In[24]:


df_items[['name', 'clean_name']].head()


# ## 4. Modeling

# ### 4.1. Content Based Filtering

# In[25]:


tfid = TfidfVectorizer()
tfid.fit(df_items['clean_name'])
tfidf_matrix = tfid.fit_transform(df_items['clean_name'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[26]:


def recommendations(name, N=10, df=df_items, cos_sim = cos_sim):
    
    recommendation = []
    idx = df[df['clean_name'] == name].index[0]
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)
    top_N_indexes = list(score_series.iloc[1:N+1].index)
    for i in top_N_indexes:
        recommendation.append(list(df['name'])[i])
    return score_series.iloc[1:N+1], recommendation


# ### 4.2. Collaborative Filtering

# Modified from: [Collaborative Filtering for Movie Recommendations by Siddhartaha Banerjee](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)

# In[27]:


class Recommender(tf.keras.Model):

    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer = 'random_normal',
            embeddings_regularizer = tf.keras.regularizers.l2(1e-5)
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.item_embedding = tf.keras.layers.Embedding(
            num_items,
            embedding_size,
            embeddings_initializer = 'random_normal',
            embeddings_regularizer = tf.keras.regularizers.l2(1e-5)
        )
        self.item_bias = tf.keras.layers.Embedding(num_items, 1) 
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0])
        user_bias = self.user_bias(inputs[:, 0]) 
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2) 
        x = dot_user_item + user_bias + item_bias
        return tf.nn.sigmoid(x)


# For the sake of execution time, I limit the embedding size to 64.

# In[28]:


num_users = len(user_encoder)
num_items = len(item_encoder)
model = Recommender(num_users, num_items, 64)
 
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae']
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                            restore_best_weights=True, patience=5)


# In[29]:


history = model.fit(
    x = X_train,
    y = y_train_scaled,
    batch_size = 64,
    epochs = 100,
    callbacks=[callback],
    validation_data = (X_val, y_val_scaled)
)


# ## 5. Result and Evaluation
# 
# ### 5.1. Content-Based Filtering

# In[30]:


test_item = df_items.sample(1, random_state=83).iloc[0]
print("Items similar to", test_item['name'], ":")
print("=====" * 20)
similarities, recommendations_result = recommendations(test_item['clean_name'], N=5)
for i, recommendation in enumerate(recommendations_result):
    print(f"{i+1}. {recommendation}")


# In[31]:


print("Top 5 cosine similarities:")
print("=====" * 10)
for i, similarity in enumerate(similarities):
    print(f"{i+1}. {similarity}")


# ### 5.2. Collaborative Filtering

# In[32]:


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.scatter(callback.best_epoch, history.history['root_mean_squared_error'][callback.best_epoch])
plt.scatter(callback.best_epoch, history.history['val_root_mean_squared_error'][callback.best_epoch])
plt.title('RMSE Value')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[33]:


plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.scatter(callback.best_epoch, history.history['mae'][callback.best_epoch])
plt.scatter(callback.best_epoch, history.history['val_mae'][callback.best_epoch])
plt.title('MAE Value')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[34]:


print("RMSE from model's rating:")
print("Train:", history.history['root_mean_squared_error'][callback.best_epoch])
print("Test:", history.history['val_root_mean_squared_error'][callback.best_epoch])


# In[35]:


print("MAE from model's rating:")
print("Train:", history.history['mae'][callback.best_epoch])
print("Test:", history.history['val_mae'][callback.best_epoch])


# In[36]:


test_name = df_users['name'].sample(1, random_state=1).to_numpy()[0]
test_raw_name = df_users[df_users['name'] == test_name]['old_name'].iloc[0]
test_user_id = df_users[df_users['name'] == test_name]['user'].iloc[0]


# In[37]:


item_id_bought = df_users[df_users['name'] == test_name]['itemId'].unique()
print("Item bought by", test_raw_name)
print("=====" * 20)
print(*["-  "+item_id_to_name[x] for x in item_id_bought], sep='\n')


# In[38]:


item_id_not_bought = df_items[~df_items['itemId'].isin(item_id_bought)]['itemId']
item_not_bought = [[item_encoder.get(x)] for x in item_id_not_bought]
arr = np.hstack(([[test_user_id]] * len(item_not_bought), item_not_bought))
ratings = model.predict(arr).flatten()


# In[39]:


recommended_items_id = [item_decoder.get(x) for x in ratings.argsort()[-5:]]
recommended_items = [item_id_to_name[x] for x in recommended_items_id]
print("Recommendation for ",test_raw_name)
print("=====" * 20)
recommendation = [f"{i+1}. {x}\n" for i, x in enumerate(recommended_items)]
print(*recommendation, sep='\n')


# In[40]:


predicted_ratings = scaler.inverse_transform(np.reshape(np.sort(ratings)[-5:], (-1,1)))
print("Rating for each item:")
for i, rating in enumerate(predicted_ratings.flatten()):
    print(i+1, ": ", rating)

