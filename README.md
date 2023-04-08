# Statistical Machine Learning
 Investigation of data mining and statistical pattern recognition that support artificial intelligence. Main topics covered include supervised learning; unsupervised learning; and deep learning, including major components of machine learning and the data analytics that enable it
 
 ## Distinguish between supervised learning and unsupervised learning

Supervised learning and unsupervised learning are two major categories of machine learning algorithms, and they differ in terms of the type of input data they work with, the nature of the learning process, and the outcomes they produce.

Supervised learning is a type of machine learning in which the algorithm learns to map input data to a set of predefined output labels or values, based on a set of labeled examples. In other words, the algorithm is given a set of input-output pairs, called a training dataset, and it learns to generalize from these examples to make predictions on new, unseen data. The goal of supervised learning is to build a model that can accurately predict the output for new, unseen input data. Examples of supervised learning algorithms include linear regression, logistic regression, decision trees, and neural networks.

Unsupervised learning, on the other hand, is a type of machine learning in which the algorithm learns to find patterns and relationships in the input data, without the use of predefined output labels or values. In other words, the algorithm is given a set of input data, called an unlabeled dataset, and it learns to discover the underlying structure and organization of the data. The goal of unsupervised learning is to identify clusters, patterns, or trends in the data that can be used to gain insights or make better decisions. Examples of unsupervised learning algorithms include clustering algorithms (e.g., k-means clustering), dimensionality reduction algorithms (e.g., principal component analysis), and anomaly detection algorithms.

In summary, supervised learning requires labeled data with known outputs, while unsupervised learning works with unlabeled data to identify patterns and relationships. Supervised learning is used for prediction tasks, while unsupervised learning is used for exploratory analysis and data mining.

### Project 1: Applying Probability Distributions in Machine Learning - Logistic Regression
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

### Project 2: Predicting housing prices with supervised learning - Linear Regression

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

### Project 3: Clustering Analysis for Customer Segmentation - Unsupervised Learning

### Project 4: Housing Prices Prediction - Decision Trees and Random Forests (Kaggle)

## Apply common probability distributions in machine learning applications
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


