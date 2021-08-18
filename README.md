# Classifying Tweets Based on Climate Change Stance
**Aim:**

Twitter is a popular social networking website where users posts and interact with messages known as "tweets". This serves as a means for individuals to express their thoughts or feelings about different subjects. Various parties such as consumers and marketers have done sentiment analysis on such tweets to gather insights into products or conduct market analysis. Furthermore, with the recent advancements in machine learning algorithms,the accuracy of our sentiment analysis predictions can improve.

`            `In this project, we attempted to Categorize tweets as climate change believers and deniers utilizing NLP and Classification techniques.The project aims to analyze how people's perceptions have changed over the years about climate change using Twitter data.The data provided comes with emoticons, usernames, and hashtags that must be processed and converted into a standard form. It also needs to extract useful features from the text, such as unigrams and bigrams, representing the "tweet". We use various machine learning algorithms to conduct sentiment analysis using the extracted features. However, just relying on individual models did not give high accuracy, so we pick the top few models to generate a model.  
## **Prerequisites :**
You need to have installed the following software and libraries on your machine before running this project.

Python 3 Anaconda: It will install a python notebook and most of the needed libraries likepandas, seaborn, matplotlib, NumPy,nltk,Scikit.
## **Libraries Used:**
**Pandas**: 

For creating and manipulating dataframes.

**WordCloud**: 

visualization technique for texts that are natively used for visualizing the tags or keywords from the websites.

**NLTK:** 

Text processing libraries for tokenization, parsing, classification,stemming,tagging etc.

Eg: nltk.tokenize , nltk.stem , nltk.corpus , nltk.probability

**Scikit.learn\_selection**:

Split data into train/test sets.

**Scikit.learn\_feature\_extraction.text:**

Used to extract features in a format supported by machine learning algorithms (MNB.Unigram and Bigram)

**DATA**

`		`To obtain climate change-related tweets, we used Twitter API via TweetPy, where the labeled data is about 14K tweets and unlabeled data about 38K tweets.

From the dataset, tweeted message categorized the tweets into the following classes:

- [ 2 ] News: Tweets linked to factual news about climate change.
- [ 1 ] Pro: Tweets that support the belief in human-caused climate change.
- [ 0 ] Neutral: Tweets that neither support nor refuse beliefs of climate change.
- [-1 ] Anti: Tweets that do not support the belief in human-caused climate change.


**Data Overview**

Data Source → Twitter Dataset/

Data Points → 43944 rows

Dataset Attributes:

- sentiment - class in which of the 3 category the tweet lies ( +ve = 1, -ve = -1 & neutral = 0 )
- message - tweet or message posted by user
- tweetid - user’s twitter id

**Data Preprocessing**

`		`To gain more insights from the dataset before starting exploration, we have to preprocess the data



Plan of Action:

- Copy the data frame and rename the class labels for better data visualization
- Extract hashtags and store them in separate data frames for each class
- Remove 'noisy entities' such as URLs, punctuations, mentions, numbers, and extra white space.
- Tokenization
- Stemming and Lemmatization
- Word Frequency
- Named entity extraction

**Exploratory Data Analysis (EDA)**

EDA is an essential step to apply Machine Learning. The primary purpose is to look at the data before making any assumptions. 

Following are some examples of EDA performed while exploring the data:

Target variable Distribution
Climate change buzzwords
Visualize top hashtags for each climate


**Data Modelling**

So, after the exploratory data analysis, we started modelling. So, for modelling, we used Machine Learning algorithms on the datasets to build a model that will generate output for analysis prediction.In this step, we have divided the data into train and test as 80%-20%, respectively.

In this process, we have used many algorithms and applied some hyperparameter tuning to do better. The algorithms which we have tried are:

1. Naïve bayes self-training
1. Multinomial Naïve bayes Unigram
1. Multinomial Naïve bayes Bigram

**Naïve Bayes**

**Naïve Bayes**is a probabilistic classifier, which means it predicts based on the probability of an object. It is called **Naïve** because it assumes that the occurrence of a particular feature is independent of the event of other features. It is 

- P(A|B) is the posterior probability of class (A, target) given predictor (B, attributes).
- P(A) is the prior probability of class.
- P(B|A) is the likelihood which is the probability of the predictor given class.
- P(B) is the prior probability of the predictor.
### **Train - Validation split**
We will now split the data for training and testing to check how well our model has performed. Also, we will randomize the data if our data includes all positive first and then all negative or some other kind of bias. We will then use: scikit\_learn's [**train_test_split()**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) for splitting.


**1.Naïve bayes self-training**

Naive Bayes classifier is a general term that refers to the conditional independence of each of the features in the model. In contrast,a Multinomial Naive Bayes classifier is a specific instance of a Naive Bayes classifier that uses a multinomial distribution for each component.

#### ***Term Frequency - Inverse Document Frequency (Tf-IDF)***
The *term frequency-inverse document frequency (Tf-IDF)* is another alternative for characterizing text documents. It can be understood as a weighted *term frequency*, which is especially useful if stop words have not been removed from the text corpus. The Tf-IDF approach assumes that the importance of a comment is inversely proportional to how often it occurs across all documents. Although Tf-idf is most commonly used to rank documents by relevance in different text mining tasks, such as page ranking by search engines, it can also be applied to text classification via naive Bayes.


Let tfn(d,f)tfn(d,f) be the normalized term frequency, and idfidf, the inverse document frequency, which can be calculated as follows

where

- ndnd: The total number of documents.

nd(t)nd(t): The number of records that contain the term 

**2. Multinomial Naïve bayes Unigram**

Multinomial Naive Bayes algorithm is a probabilistic learning method primarily used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and is a collection of many algorithms where all the algorithms share one common principle: each feature being classified is not related to any other feature. 

To proceed further with the sentiment analysis, we need to do text classification. We can use the **'bag of words (BOW)**' model for the examination. In laymen terms, the BOW model converts text into numbers, which can then be used in an algorithm for analysis.

we will generate DTM using the CountVectorizer module of sci-kit-learn

**Using CountVectorizer to prepare the 'bag of words**

- tokenizer = Overrides the string tokenization step, we generate tokenizer from NLTK's Regex tokenizer (by default: None)
- lowercase = True (no need to use, as it is set True by default)
- stop\_words = 'English (by default None is used, to improve the result, we can provide a custom made list of stop words)
- ngram\_range = (1,1) (by default its (1,1) i.e strictly monograms will be used, (2,2) only bigrams while (1,2) uses both)
# Our analysis will be in 5 steps:
# **1.Defining the Model**
# The first two steps of defining and compiling the model are reduced to identifying and importing the model from sklearn (as sklearngives as precompiled models).

We will use one of the [**Naive Bayes (NB)**](https://scikit-learn.org/stable/modules/naive_bayes.html) classifiers for defining the model. Specifically, we will use a [**MultinomialNB classifier**](https://scikit-learn.org/stable/modules/naive_bayes.html).

**2.Compilation**

Importing Multinomial Naive Bayes model from sklearn library

**3.Model-Fitting**

**It fits the data in Multinomial Naive Bayes.**

4.Evaluating the Models

Here we quantify the quality of our model. We use the [**metrics**](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) module from the sklearn library to evaluate the predictions.


Finally, we do Hyperparameter tuning to get the best-predicted results.

**5.Hyperparameter Tuning:**

The Accuracy score in Multinomial Naïve Bayes is **74.49%**
And After Hyperparameter Tuning, it increased to **76.29%**
# **Tweaking the model**

We have observed that the accuracy of our model is over 76%. We can now play with our model to increase its' accuracy.
## Trying different n-grams
**2. Bigram**

The same procedures above are followed; we only have to change the range to (2,2).

**After Hyperparameter Tuning**


The Accuracy score before and after is **70%itself.**
## **Task Performed By Every Member of Team During Internship**
**Steps that we performed:**

- Importing Packages
- Data Loading
- Data pre-processing
- Exploratory data analysis
- Feature engineering
- Feature selection
- Feature transformation
- Model building
- Model evaluation
- Model tuning
- Deployment

**Tools used:**

- Python
- Pycharm
- Jupyter Notebook
- Google Colab
- GitHub
- GitBash
- SublimeTextEditor

**Team Members**

1. Ashutosh Chandra
1. Joan Jose M
1. Harsha KG
1. Sharon Shelke
1. Shubratha Dutta
1. Disha Sonkar
1. Mona Kumari
### **Team Leader**
Ashutosh Chandra
### **Coordinator Name**
- Mr Yasin Shah






