#!/usr/bin/env python
# coding: utf-8

# 
# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import math
import json
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
get_ipython().run_line_magic('matplotlib', 'inline')

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# #### we start off by taking a better look at our data sets

# In[2]:


portfolio.head()


# In[3]:


portfolio.isna().sum()


# In[ ]:





# In[4]:


profile.head()


# In[5]:


profile.isna().sum()/profile.shape[0]


# In[6]:


transcript.head()


# In[7]:


portfolio.describe()


# ### Cleaning Data
# #### Now that we have looked at our data, we take a deeper and start cleaning the data

# In[8]:


#let's start with the portfolio data set. 
#one-hot encording for channels

for index, row in portfolio.iterrows():
    for channel in ['web', 'email', 'social', 'mobile']:
        if channel in portfolio.loc[index, 'channels']:
            portfolio.loc[index, channel] = 1
        else:
            portfolio.loc[index, channel] = 0


# In[9]:


for index, row in portfolio.iterrows():
    for offertype in ['bogo', 'informational', 'discount']:
        if channel in portfolio.loc[index, 'offer_type']:
            portfolio.loc[index, offertype] = 1
        else:
            portfolio.loc[index, offertype] = 0


# In[10]:


#rename person to customer_id
portfolio = portfolio.rename(columns = {'id' : 'offer_id'})

portfolio.shape


# In[11]:


#check for null values
portfolio.isnull().sum()


# In[12]:


# working with the profile data
profile.isnull().sum()


# In[13]:


profile.head(10)


# In[14]:


#taking another look at the profile dataset, we realize that the age 118 has Nans on the gender and the income, therefore we 
#choose to drop it

profile = profile[profile.age != 118]
profile.isna().sum()


# In[15]:


#change the datatype(became_member_on) into string
profile['became_member_on'] = profile['became_member_on'].astype(str)

#make a column for the year
profile['membership_year'] = profile['became_member_on'].apply(lambda x: x[0:4])

#change into datetime
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'])

#rename id to customer_id
profile = profile.rename(columns = {'id' : 'customer_id'})


# In[16]:


profile.head()


# In[17]:


# working with the transcript data

transcript.head()


# In[18]:


transcript[transcript["person"]=='78afa995795e4d85b5d9ceeca43f5fef'].value.values


# In[19]:


# Extract each key that exist in 'value' column to a seperate column.
# getting the different keys  that exists in the 'value' column
keys = []
for idx, row in transcript.iterrows():
    for k in row['value']:
        if k in keys:
            continue
        else:
            keys.append(k)


# In[20]:


keys


# In[21]:


transcript['offer_id'] = '' 
transcript['amount'] = 0  
transcript['reward'] = 0  


# In[22]:


# Iterating over clean_transcript dataset and checking 'value' column
# then updating it and using the values to fill in the columns created above
for idx, row in transcript.iterrows():
    for k in row['value']:
        if k == 'offer_id' or k == 'offer id': 
            transcript.at[idx, 'offer_id'] = row['value'][k]
        if k == 'amount':
            transcript.at[idx, 'amount'] = row['value'][k]
        if k == 'reward':
            transcript.at[idx, 'reward'] = row['value'][k]


# In[23]:


transcript['offer_id'] = transcript['offer_id'].apply(lambda x: 'N/A' if x == '' else x)
transcript.drop('value', axis=1, inplace=True)


# In[24]:


transcript.head()


# In[25]:


dummy = pd.get_dummies(transcript['event'])
transcript = pd.concat([transcript, dummy], axis = 1)

#rename person to customer_id
transcript = transcript.rename(columns = {'person' : 'customer_id'})

#drop amount column
transcript = transcript.drop(columns = 'amount')
transcript = transcript.drop(columns = 'reward')


# In[26]:


transcript.head()


# In[27]:


#checking for null values
transcript.isnull().sum()


# ### Data Exploration and Data Visualization

# Merging all the data sets together

# In[28]:


df = pd.merge(transcript, profile, on= 'customer_id')


# In[29]:


df.head()


# In[30]:


df = pd.merge(df, portfolio, on = 'offer_id', how = 'left')


# In[31]:


df.head()


# In[32]:


df.describe()


# In[33]:


df = df.dropna()


# In[34]:


# ensuring we have clean data
df.isnull().sum().sum()


# In[35]:


df.describe()


# In[36]:


df.groupby('event').customer_id.count()


# In[37]:


sns.countplot(df['event'])
plt.title('events')
plt.ylabel('Transcripts')
plt.xlabel('Transcript type')
plt.xticks(rotation = 0)
plt.show();


# #### The above graph clearly shows us that more offers were received than those that were completed, 
# #### Thus most people received and viewed the offers but didnot actually complete the transactions.

# In[38]:


sns.countplot(x = df[df['gender'] != 'NA']['gender'])
plt.title('Income Vs Gender')
plt.ylabel('Income')
plt.xlabel('Gender')
plt.xticks(rotation = 0)
plt.show();


# In[39]:


df.age.describe()


# In[40]:


df.age.hist(bins = 30)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Age Group Distribution');


# ### Observations
# ##### According to the chart above, we can clearly see that the majority of the customers are in the range of 50-65 years
# ##### The mean age is 55 years

# In[41]:


ax = df.gender.value_counts()
ax.plot(kind = 'bar')
plt.ylabel('No. of People')
plt.xlabel('Gender')
plt.title('Gender Distribution')


# ##### We have more male users of the app than the ladies

# In[42]:


ax = df.offer_type.value_counts()
ax.plot(kind = 'bar')
plt.ylabel('No. of People')
plt.xlabel('Offers')
plt.title('Offers Distribution')


# In[43]:


#Offer Types vs. Offer Events

plt.figure(figsize=(14,6))
g = sns.countplot(x= 'event', hue = 'offer_type', data = df)
Event = ['Offer Completed', 'Offer Received', 'Offer Viewed']
plt.title('Offer Types vs. Offer Event')
plt.ylabel('Total')
plt.xlabel('Offer Event')
plt.legend(['Bogo','Informational','Discount'])
g.set_xticklabels(Event)
plt.show();


# Observations:
# More Discount and BOGO offers were responded to than informational offers

# In[44]:


df.head()


# In[45]:


df[['offer_type', 'offer received', 'offer viewed', 'offer completed']].groupby(['offer_type']).sum().reset_index()


# ### Which demographic groups respond best to which offer type?

# In[46]:


# Making age groups
df['age_groups'] = pd.cut(x=df['age'], bins=[10, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109])


# In[47]:


df['age_groups'].unique()


# In[48]:


df['age_by_decade'] = pd.cut(x=df['age'], bins=[10,19, 29, 39, 49, 59, 69, 79, 89, 99, 109], 
                             labels=['Late Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '110s'])


# In[49]:


df.head()


# In[50]:


# Finding the most popular offers according to each age group
plt.figure(figsize=(14, 6))
g = sns.countplot(x='age_by_decade', hue="offer_type", data= df)
plt.title('Most Popular Offers to Each Age Group')
plt.ylabel('Total')
plt.xlabel('Age Group')
plt.legend(['BOGO', 'Informational','Discount'])
plt.show();


# ### Observations
# #### We clearly observe that the 50's and 60's age groupgs are more responsive to the BOGO and Discount Offers and Informational offers generally generate less attention.
# #### In general it can be concluded that all age groups best respond to BOGO and Discount Offers.

# ## Data Modeling

# In[51]:


df.head()


# A machine learning model that predicts which offer type people are more likely to respond to. We split The 
# split the dataframe into training and test data.
# 

# In[52]:


#to prepare the data for modeling, create dummies for the different offer types

# converting offer type to numerical form
offer_type = df['offer_type'].astype('category').cat.categories.tolist()
num_offer_type = {'offer_type' : {k: v for k,v in zip(offer_type,list(range(1,len(offer_type)+1)))}}
df.replace(num_offer_type, inplace=True)


# In[53]:


# converting gender to numerical form
gender = df['gender'].astype('category').cat.categories.tolist()
num_gender = {'gender' : {k: v for k,v in zip(gender,list(range(1,len(gender)+1)))}}
df.replace(num_gender, inplace=True)


# In[54]:


#to predict which model would best be get a reaction when it has been received, we only consider those people that have received
# offers

df = df[~(df['offer received'] == 0)]
df.head()


# In[55]:


df.describe()


# In[56]:


# Split into training and testing data
X= df[['duration','age','gender','reward','email','mobile','social','web', 'difficulty','income']]
y = df['offer_type']


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                        test_size=0.20, 
                                                        random_state=42)


# In[58]:


clf1 = DecisionTreeClassifier()                
clf2 = RandomForestClassifier()
classifier_list = [clf1,clf2]


# In[59]:


clf_dict = {}
for clf in classifier_list:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred, average='weighted')
    clf_dict[clf.__class__.__name__] = accuracy


# In[60]:


print(clf_dict)


# In[61]:


parameters = {#'clf__estimator__n_estimators': [10]
             'n_estimators': [5,10],
            'min_samples_split': [2,3,4]}
             #'clf__estimator__min_samples_split': [2,3]}

cv = GridSearchCV(estimator = clf2, param_grid = parameters)


# In[62]:


cv.fit(X_train,y_train)


# In[63]:


print('Training F1_score:', cv.score(X_train,y_train))
print('Test F1_score:', cv.score(X_test,y_test))


# #### What to consider before particular offers are sent.

# In[64]:


cv.best_estimator_.feature_importances_


# In[65]:


feature_importances = pd.DataFrame(cv.best_estimator_.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[66]:


feature_importances


# ### Conclusion 
# 
# 1. The people in the age range of 50-65 are more likely to visit a Starbucks
# 2. Overall, Bogo is the most popular kind of Offer Type
# 3. Looking at different age groups, we can see that Bogo is popular than any other type of offers except for the ones in theirs 30s where it is as popular as the Discount offer and the ones who are in their 60s where Discount is more popular. But Informational is the least popular of all in all age groups
# 4. In most of the cases, the offers were received but not completed. Discount offer was the which was received by most and also completed followed by BOGO
# 
# And created a Machine Learning model using Random Forest Classifier  with the accuracy of 1. 
# I may be getting an accuracy of 1 due to considering only the most important features and dropping all unnecessary features.
# 
# #### 5. Its also key to note that unlike what would be expected, income doesnot affect the choice on whether to act complete an offer or not, rather the duration, difficulty and reward are the key factors.
# 
# There may be overfitting which can be solved by considering more data.
# As more rows were eliminated due to Nan values and duplicates the model had less data to work with.
# The data available on the customer should also be indepth to define each individual customer. The features of the customer would have helped in producing better classification model results.
# 

# In[ ]:




