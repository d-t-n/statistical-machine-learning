# Statistical Machine Learning
 Investigation of data mining and statistical pattern recognition that support artificial intelligence. Main topics covered include supervised learning; unsupervised learning; and deep learning, including major components of machine learning and the data analytics that enable it
 
 ## 1. Distinguish between supervised learning and unsupervised learning

Supervised learning and unsupervised learning are two major categories of machine learning algorithms, and they differ in terms of the type of input data they work with, the nature of the learning process, and the outcomes they produce.

Supervised learning is a type of machine learning in which the algorithm learns to map input data to a set of predefined output labels or values, based on a set of labeled examples. In other words, the algorithm is given a set of input-output pairs, called a training dataset, and it learns to generalize from these examples to make predictions on new, unseen data. The goal of supervised learning is to build a model that can accurately predict the output for new, unseen input data. Examples of supervised learning algorithms include linear regression, logistic regression, decision trees, and neural networks.

Unsupervised learning, on the other hand, is a type of machine learning in which the algorithm learns to find patterns and relationships in the input data, without the use of predefined output labels or values. In other words, the algorithm is given a set of input data, called an unlabeled dataset, and it learns to discover the underlying structure and organization of the data. The goal of unsupervised learning is to identify clusters, patterns, or trends in the data that can be used to gain insights or make better decisions. Examples of unsupervised learning algorithms include clustering algorithms (e.g., k-means clustering), dimensionality reduction algorithms (e.g., principal component analysis), and anomaly detection algorithms.

In summary, supervised learning requires labeled data with known outputs, while unsupervised learning works with unlabeled data to identify patterns and relationships. Supervised learning is used for prediction tasks, while unsupervised learning is used for exploratory analysis and data mining.

### - 1.1 Project 1: Applying Probability Distributions in Machine Learning - Logistic Regression
The objective of this project is to develop a understanding of probability distributions and their application in solving real-world problems using machine learning algorithms.
The project would involve the following steps:

1. Research and identify common probability distributions:
- Gaussian (normal) distribution
- Poisson distribution
- Bernoulli distribution
- etc.


2. Select a machine learning algorithm:
- Classification algorithm (e.g. logistic regression)
- Clustering algorithm (e.g. k-means)


3. Implement the algorithm:
- Load dataset
- Split into training and testing sets
- Train the model using a programming language such as Python


4. Apply probability distributions:
- Calculate probability density function (PDF) or cumulative distribution function (CDF)
- Use identified probability distributions to model the data


5. Evaluate the performance:
- Use metrics such as accuracy, precision, recall, and F1 score
- Analyze the results to determine the effectiveness of using probability distributions in machine learning applications.


Overall, this project would provide an opportunity to develop a deeper understanding of probability distributions and their application in machine learning algorithms. It would also allow them to gain hands-on experience in implementing machine learning algorithms and applying probability distributions to real-world datasets.

### - 1.2 Project 2: Predicting housing prices with supervised learning - Linear Regression

1. Problem Definition:
The goal of this project is to build a machine learning model that can predict housing prices based on various features such as location, size, number of rooms, etc. This is an important problem in real estate as it can help both buyers and sellers make informed decisions about the value of a property.

2. Data Gathering:
The data for this project will be gathered from web open sources datasets such as Kaggle or UCI Machine Learning Repository. The dataset should include information about various properties such as location, size, number of rooms, number of bathrooms, age of the property, and the sale price. The data will need to be cleaned and preprocessed before modeling.

3. Data Cleaning and Preparation:
The data will need to be checked for missing or incorrect values and outliers. Any missing values should be imputed or dropped, and outliers should be dealt with appropriately. The data should also be normalized or standardized if necessary. Features that are not relevant to the prediction of housing prices should be removed.

4. Modeling:
Once the data is cleaned and prepared, several supervised learning algorithms can be used to build the prediction model. Some examples of algorithms that can be used include linear regression, decision trees, random forests, and gradient boosting. The model will need to be trained on a portion of the dataset and validated on a separate portion to avoid overfitting.

5. Evaluation:
The performance of the model will be evaluated using various metrics such as mean squared error, R-squared, and accuracy. The evaluation should be performed on a test set that the model has not seen before.

6. Deployment and Visualization:
Once the model has been trained and evaluated, it can be deployed to make predictions on new, unseen data. The predictions can be visualized using various tools such as scatter plots or heat maps to show the predicted prices compared to the actual prices. This can provide insights into the accuracy and effectiveness of the model.

Conclusion:
In summary, this project aims to build a machine learning model that can accurately predict housing prices based on various features. By following the steps of problem definition and understanding, data gathering, cleaning and preparation, modeling, evaluation, deployment and visualization, this project can help buyers and sellers make informed decisions about the value of a property.

### - 1.3 Project 3: Clustering Analysis for Customer Segmentation - Unsupervised Learning
Project: Clustering Analysis for Customer Segmentation

1. Problem Definition and Understanding:
A retail company wants to segment its customers into different groups based on their purchasing behavior. The goal is to identify distinct customer segments with similar purchasing patterns, in order to tailor marketing strategies and improve customer satisfaction. To achieve this, the company has collected a dataset of customer purchase history, including the product categories, quantities, and prices of items purchased.

2. Data Gathering, Cleaning and Preparation:
The dataset can be obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/online+retail). The data contains information on over 500,000 transactions made by customers of a UK-based online retail company during a period of one year. The data includes attributes such as customer ID, product ID, product description, quantity, unit price, and transaction date.

To prepare the data for analysis, the following steps may be taken:

Remove any duplicate records or transactions
Remove any missing or invalid data
Convert categorical variables (e.g., product description) into numerical variables using one-hot encoding
Normalize or standardize the data to eliminate any differences in scale or range between the variables

3. Modeling:
The clustering algorithm used for this project will be k-means clustering. The k-means algorithm is an unsupervised learning algorithm that partitions the data into k distinct clusters based on the similarity of the data points.

4. Evaluation:
The performance of the clustering algorithm will be evaluated using the silhouette score. The silhouette score measures the similarity of a data point to its own cluster compared to other clusters, and ranges from -1 to 1, with higher values indicating better clustering.

5. Deployment and Visualization:
The results of the clustering analysis can be visualized using scatter plots or heat maps, with each data point colored according to its assigned cluster. The identified customer segments can then be used to develop targeted marketing strategies and improve customer satisfaction.

### - 1.4 Project 4: Housing Prices Prediction - Decision Trees and Random Forests (Kaggle)
Decision Trees and Random Forests project from Kaggle

## 2. Apply common probability distributions in machine learning applications
Probability distributions are an essential concept in machine learning that helps us model and analyze the data. They describe the likelihood of a random variable taking on different values, and they provide a mathematical framework for understanding the data.

Here are some common probability distributions used in machine learning applications:

Normal distribution: Also known as the Gaussian distribution, it is one of the most widely used probability distributions. It is used to model continuous variables that are symmetric and have a bell-shaped curve. The normal distribution has two parameters, mean (μ) and standard deviation (σ). For example, the height of people in a population can be modeled by a normal distribution.

Bernoulli distribution: It is a discrete probability distribution that models the probability of a binary event (success or failure). It has only one parameter, p, which represents the probability of success. For example, the result of a coin flip can be modeled by a Bernoulli distribution.

Binomial distribution: It is used to model the number of successes in a fixed number of trials of a Bernoulli experiment. It has two parameters, n (number of trials) and p (probability of success). For example, the number of heads in ten coin flips can be modeled by a binomial distribution.

Poisson distribution: It is used to model the number of occurrences of an event in a fixed interval of time or space. It has one parameter, λ (rate parameter). For example, the number of phone calls received by a call center in an hour can be modeled by a Poisson distribution.

Exponential distribution: It is used to model the time between two successive events in a Poisson process. It has one parameter, λ (rate parameter). For example, the time between two phone calls in a call center can be modeled by an exponential distribution.

Beta distribution: It is used to model the probability distribution of a random variable that is bounded between 0 and 1. It has two parameters, α and β, which can be interpreted as the number of successes and failures, respectively. For example, the probability of a website user clicking on an advertisement can be modeled by a beta distribution.

These probability distributions have many applications in machine learning, such as:

In Bayesian inference, the prior and posterior distributions are often chosen from the family of probability distributions based on the problem domain.
In regression analysis, the residual errors are often assumed to follow a normal distribution.
In classification problems, the class probabilities can be modeled by a binomial or a multinomial distribution.
In clustering problems, the distribution of the data points can be modeled by a mixture of normal distributions.
In reinforcement learning, the rewards can be modeled by a Poisson or an exponential distribution.
In conclusion, understanding and applying common probability distributions is crucial in machine learning applications to model and analyze the data accurately.


### 2.1 Project
Here is an outline of what the project will cover:

Introduction: In this section, we will introduce the concept of probability distributions and their importance in machine learning applications. We will also provide an overview of the common probability distributions used in machine learning.

Normal Distribution: In this section, we will discuss the normal distribution and its properties. We will also provide examples of how the normal distribution is used in machine learning.

Bernoulli Distribution: In this section, we will discuss the Bernoulli distribution and its properties. We will also provide examples of how the Bernoulli distribution is used in machine learning.

Binomial Distribution: In this section, we will discuss the binomial distribution and its properties. We will also provide examples of how the binomial distribution is used in machine learning.

Poisson Distribution: In this section, we will discuss the Poisson distribution and its properties. We will also provide examples of how the Poisson distribution is used in machine learning.

Exponential Distribution: In this section, we will discuss the exponential distribution and its properties. We will also provide examples of how the exponential distribution is used in machine learning.

Beta Distribution: In this section, we will discuss the beta distribution and its properties. We will also provide examples of how the beta distribution is used in machine learning.

Applications of Probability Distributions in Machine Learning: In this section, we will provide examples of how probability distributions are used in various machine learning applications, such as Bayesian inference, regression analysis, classification problems, clustering problems, and reinforcement learning.

Conclusion: In this section, we will summarize the key points of the project and emphasize the importance of understanding and applying common probability distributions in machine learning.


## 3. Use cross validation to select parameters
Cross-validation is a statistical method used to estimate the performance of a machine learning model. It is used to evaluate how well a model will perform on new, unseen data. The purpose of cross-validation is to train the model on a subset of the data and test it on the remaining data. The data is divided into training and validation sets, and the model is trained on the training set and validated on the validation set. The process is repeated several times, with different subsets of the data used for training and validation each time.

The primary use of cross-validation is to select the best hyperparameters for a machine learning model. Hyperparameters are parameters that are set before training a model and can have a significant impact on the model's performance. Examples of hyperparameters include the learning rate, the number of hidden layers in a neural network, and the regularization parameter.

Cross-validation is important because it helps prevent overfitting of the model to the training data. Overfitting occurs when the model is too complex and captures noise in the training data, which results in poor performance on new, unseen data. By using cross-validation to select the best hyperparameters, we can reduce the risk of overfitting and improve the generalization performance of the model.

In summary, cross-validation is a powerful tool for selecting the best hyperparameters for a machine learning model. By using cross-validation, we can improve the model's performance on new, unseen data and reduce the risk of overfitting.

### 3.1 Project
Project Title: Predicting Customer Churn in Telecom Industry

Project Overview (Kaggle):

The aim of this project is to develop a model that can predict customer churn in the telecom industry. The dataset used in this project contains information about customers' demographic information, service usage, and their churn status. 

Subtopics:

- Business Problem
- Exploratory Data Analysis
- Data Preprocessing & Feature Engineering
- Modelling


Data Preprocessing & Feature Engineering: In this step, we will clean the dataset, check for missing values, and perform feature scaling if required. We will also perform one-hot encoding for categorical features, and split the dataset into training and testing sets.

Cross-Validation: Cross-validation is a technique used to evaluate the performance of a model by dividing the dataset into k-folds, training the model on k-1 folds, and testing it on the remaining fold. This process is repeated k times, with each fold being used as the test set once. We will use 10-fold cross-validation to evaluate our model's performance.


Conclusion:

In conclusion, this project applies cross-validation to select the best hyperparameters for a Random Forest classifier to predict customer churn in the telecom industry. The project demonstrates the importance of hyperparameter tuning in improving the model's performance and how cross-validation can be used to select the best hyperparameters.

Code in 3-project.ipynb file

## 4. Use maximum likelihood estimate (MLE) for parameter estimation
Maximum likelihood estimate (MLE) is a statistical method used in machine learning to estimate the parameters of a statistical model. It involves finding the parameter values that maximize the likelihood of the observed data given the model. In other words, MLE determines the values of the model's parameters that make the data most probable.

In machine learning, MLE is often used for parameter estimation in models such as regression, classification, and clustering. The goal of these models is to learn the underlying patterns in the data and use them to make predictions on new, unseen data. MLE provides a method for estimating the optimal parameters of these models based on the observed data.

The importance of MLE in machine learning lies in its ability to estimate the most likely values of the model's parameters, given the observed data. This is crucial for building accurate and reliable models that can generalize well to new data. By optimizing the likelihood function, MLE ensures that the model's parameters are adjusted to fit the data as closely as possible, while also avoiding overfitting or underfitting.

Moreover, MLE is a widely used method in machine learning because it provides a straightforward and computationally efficient way of estimating model parameters. It also has a solid theoretical foundation, and its properties are well understood, which makes it a reliable and trusted method for parameter estimation.

In summary, MLE is a powerful statistical method that plays a vital role in machine learning by providing a way to estimate the optimal parameters of a model based on the observed data. Its importance lies in its ability to improve the accuracy and reliability of machine learning models, making them more effective in solving real-world problems.

### 4.1 Project: Maximum Likelihood Estimate (MLE) for parameter estimation in machine learning - Optimal Dosage of Drug
One real-life applied project that involves the use of maximum likelihood estimate (MLE) for parameter estimation in machine learning is the development of drug dosage estimator.

Business Understanding:
A pharmaceutical company has developed a new drug that is supposed to lower blood pressure in patients. They need to determine the optimal dosage of the drug to achieve the desired effect while minimizing any potential side effects. They have conducted a clinical trial with a sample of patients, and they want to use machine learning to estimate the optimal dosage.

Data Understanding:
The company has collected data from a clinical trial conducted on a sample of patients. The dataset contains the patient's age, gender, weight, and blood pressure measurements before and after taking the drug. The data is in a CSV file, and it is clean and ready for analysis.

Data Preparation:
The data needs to be split into training and testing sets. The training set will be used to train the machine learning model, while the testing set will be used to evaluate the performance of the model. We will use the scikit-learn library to split the data.

Modeling:
We will use the Maximum Likelihood Estimate (MLE) to estimate the parameters of a linear regression model. The linear regression model will predict the change in blood pressure based on the patient's age, gender, weight, and the dosage of the drug. The MLE will estimate the optimal values for the model parameters that maximize the likelihood of observing the training data. We will use the statsmodels library to perform the MLE.

Evaluation:
We will evaluate the performance of the model on the testing set using the mean squared error (MSE) metric. The MSE measures the average squared difference between the predicted and actual blood pressure measurements in the testing set. We will compare the MSE of the linear regression model with the MSE of a baseline model that always predicts the mean blood pressure measurement in the training set.

Code in 4-project.ipynb

## 5. Implement fundamental learning algorithms such as logistic regression and k-means clustering

Logistic regression is a popular classification algorithm used to predict binary or categorical outcomes. It works by estimating the probability of an outcome based on a set of input features. The algorithm optimizes a cost function to find the parameters that best fit the data. Logistic regression is widely used in many fields, including healthcare, finance, and marketing.

K-means clustering is a type of unsupervised learning algorithm used to group data points into clusters based on their similarities. The algorithm iteratively partitions the data into k clusters by minimizing the sum of squared distances between each data point and the centroid of its assigned cluster. K-means clustering is commonly used in image segmentation, customer segmentation, and anomaly detection.

To implement these fundamental learning algorithms, one needs to have a strong understanding of the underlying mathematical concepts and the ability to code them from scratch. The first step is to preprocess the data by cleaning and normalizing it. Then, the data can be split into training and testing sets. The logistic regression and k-means clustering models can be trained using the training data. Once trained, the models can be evaluated using the testing data to determine their performance.

In addition to coding from scratch, there are also many open-source libraries available for logistic regression and k-means clustering. For example, scikit-learn is a popular Python library that provides implementations of these algorithms and other machine learning models.

In conclusion, implementing fundamental learning algorithms such as logistic regression and k-means clustering is essential for building a strong foundation in machine learning. These algorithms are widely used and form the basis for more complex models. It is important to have a deep understanding of these algorithms and the ability to implement them from scratch.

### 5.1 Project: E-commerce classification

Business understanding:
Suppose we are working for an e-commerce company that sells a wide range of products. The company wants to identify the customer segments that are most likely to purchase a new product line they are introducing. To achieve this, the company needs a model that can classify customers into different segments based on their purchase history and demographics.

Data understanding:
We have access to the company's customer transaction database, which includes the customer demographics (age, gender, income, etc.) and the products they have purchased in the past. We also have some additional data from third-party sources, such as social media activity and online search history, that can be used to enrich the customer profiles. The data is relatively clean but requires some preprocessing, such as removing missing values and scaling the numerical features.

Data preparation:
We will use Python and the scikit-learn library to prepare and model the data. We will start by cleaning and scaling the data using the StandardScaler from the preprocessing module. Then, we will use the KMeans algorithm from the cluster module to group the customers into different segments based on their purchase history and demographics. Finally, we will use the logistic regression algorithm from the linear_model module to predict which customer segments are most likely to purchase the new product line.

Modeling:
We will start by using the KMeans algorithm to cluster the customers based on their purchase history and demographics. We will use the elbow method to determine the optimal number of clusters. Once we have the clusters, we will use the logistic regression algorithm to predict which customer segments are most likely to purchase the new product line. We will use the accuracy metric to evaluate the performance of the model.

Code in 5-project.ipynb



