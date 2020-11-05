# Starbucks Capstone Project

### The link to the Blog post is here [Blog Post](https://evelyxne-en.medium.com/).

Table of Contents
1. Project Overview
2. Project Motivation
3. Project Components
4. Files
5. Software Requirements
6. Conclusion
7. Credits and Acknowledgements

### 1. Project Overview
The data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

My task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Because this is a capstone project, I am free to analyze the data any way I see fit. For example, I could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or I could build a model that predicts whether or not someone will respond to an offer.


### 2. Project Motivation
This project is the Capstone project of my Data Scientist nanodegree with Udacity. As students in the nanodegree, we have the option to take part in the Starbucks Capstone Challenge. For the challenge, Udacity provided simulated data that mimics customer behavior on the Starbucks rewards mobile app.

In this project, I use the data to answer a business questions:

Which demographic groups respond best to which offer type.
And create Machine Learning model to predict which offer_type people are most likely to respond to.

### 3. Project Components
Importing Libraries and Reading Dataset
Data Wrangling
Exploratory Data Analysis
Model Modeling
Conclusion

### 4. Files
├── notebook 
├── data - contains dataset in JSON format



### 5. Software Requirements
This project uses Python 3.6 and the following necessary libraries:

pandas
numpy
math
json
matplotlib
sklean
seaborn
scipy

### 6. Conclusion
Conclusion
The people in the age range of 50-65 are more likely to visit a Starbucks
Overall, Bogo is the most popular kind of Offer Type
Looking at different age groups, we can see that Bogo is popular than any other type of offers except for the ones in theirs 30s where it is as popular as the Discount offer and the ones who are in their 60s where Discount is more popular. But Informational is the least popular of all in all age groups
In most of the cases, the offers were received but not completed. Discount offer was the which was received by most and also completed followed by BOGO
And created a Machine Learning model using Random Forest Classifier with the accuracy of 1. I may be getting an accuracy of 1 due to considering only the most important features and dropping all unnecessary features.

Its also key to note that unlike what would be expected, income doesnot affect the choice on whether to act complete an offer or not, rather the duration, difficulty and reward are the key factors.
There may be overfitting which can be solved by considering more data. As more rows were eliminated due to Nan values and duplicates the model had less data to work with. The data available on the customer should also be indepth to define each individual customer. The features of the customer would have helped in producing better classification model results. Deploy Machine Learning model to web.

### 7. Credits and Acknowledgements
Data for working with the project was provided by Udacity in association with Starbucks.