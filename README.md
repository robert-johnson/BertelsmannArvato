# Bertelsmann Arvato Project - Udacity

Data Scientist Nanodegree Capstone Project

All the code is in a public repository at the link below:

https://github.com/robert-johnson/BertelsmannArvato

A blog post on Medium summarising the work can be found at the link below:

<medium url to post>

## Installation

- Libraries included in Anaconda distribution of Python 3.8.
- Packages versions:

    - pandas Version: 1.2.4
    - numpy Version: 1.20.3
    - matplotlib Version: 3.4.2
    - scikit-learn Version: 0.24.2
    - imbalanced-learn Version 0.8.0

## Project Description

In this capstone project we were given information on the general German population provided by Bertelsmann Arvato Analytics.  We were asked to identify customers who were most like existing customers, using this German population information.  The goal of this analysis is to focus the marketing efforts/budget for future advertisements on the customer segements that were most likely to respond favorably.
  
This data was provided solely for the durations of this project only and will be deleted after completion of the project
in accordance with Bertelsmann Arvato terms and conditions.

The project has 3 phases:

**Customer Segmentation**
During this part, techniques to clean, categorize, reduce dimensionality and cluster similar customers were used to correlate the general German population with the customers provided.  The outcome is the identification of features within the dataset, that correspond to target customers.

**Supervised Learning Model**
During this part a supervised learning machine learning model was created and tuned to predicted customer response to a marketing campaign.  Most of the techniques used Customer Segmentation were also applied to the training and test data provided.

**Kaggle Competition**
During this part, the model was used to identify customers, and submit them to the Kaggle website for scoring.
    
## File Descriptions

The data provided is not publicly available according to Bertelsmann Arvato terms and conditions.

The code is contained in following files:

    utils/cluster.py - Python file with all the clustering functions defined
    utils/groups.py - Python file with lists defined to aid in the filling of missing values and imputing data
    utils/impute.py - Python file with Simple and KNN imputing functions
    utils/transform.py - Python file with transforming of columns/rows
    data_cleaning.py - Python file for calling the cleaning and transforming functions for data files
    learning.py - Python file for creating, tuning and testing the Classifiers

## Results

**Customer Segmentation Report**

In processing the data, it is hard not to wonder if the lack of understanding of the data facilitated the mistaken inclusion or exclusion of important features.  Therefore the validity of the clusters may be questionable.  However, the clusters I was able to identify are:

- Owners of higher end cars (BMW/Benz) but not between the ages of 46–60. Defined as middle aged and affluent.
- Affluent people who own their homes, but don't save or invest their money.
- Owners of a car who are less than 25 years old, and do not own a high end car.


**Supervised Learning Model**

I trained a GradientBoostingClassifier with a performance of ROC AUC score of 0.78308.

All the code is in a public repository at the link below:

https://github.com/robert-johnson/BertelsmannArvato

My blog post on Medium outlining the steps I took to complete the project is located at:

https://rajjr-tx.medium.com/customer-segmentation-of-bertelsmann-arvato-financial-solutions-data-udacity-capstone-860ecf2c0a3b

## Licensing, Authors and Acknowledgements

I would like to give credit to:

- Bertelsmann Arvato Analytics for providing the data for the project.
- Stackoverflow community for making code questions and answers publicly available.
- Udacity Data Scientist Nanodegree Peers for their project and blog posts.
