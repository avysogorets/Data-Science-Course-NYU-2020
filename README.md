# Introduction to Data Science, Fall 2020

Welcome to the repository of Intro to Data Science for PhD Students (DS-GA 3001), Fall'20!  
We meet on Wednesdays, 11 - 11:50 am in room 650, 60 5th Ave.
I will post lab materials here right before each meeting.

### Before the first meeting (9/16):
Clone the repository to your local machine by writing ```git clone https://github.com/avysogorets/IDS2020.git``` in your preferred folder.
We will often work with interactive python (3+) in jupyter notebooks, so please make sure to have installed:
 1. Jupyter ([Install](https://jupyter.org/install));
 2. Common python libraries: ```math```, ```numpy```, ```pandas```, ```matplotlib```, ```scipy```;
 3. For the first lab, we will also need ```pytrends```
 

To download lab materials, ```git pull``` to synchronize at the start of each lab.

### Lab 2 (09/23):
Lab2 folder is now availbale. If you want to follow along on your local machine during the lab, please install the following python libraries (in addition to those used last time): ```pytrends```, ```pydotplus```, ```graphviz```, ```matplotlib```, ```scikit-learn```.

### Lab 3 (09/30):
Lab3 folder is now available. No additional python libraries will be required today. See you in class!

### Lab 4 (10/07):
Lab4 folder is now available. You should be able to run this notebook without any special libraries. Today we will
 - understand **COMPAS** and ProPublica's defendant profiling data;
 - explore the data and assess recidivism predictions produced by **COMPAS**.

### Lab 5 (10/14):
Lab5 folder is now available. You will need to install ```pandas-profiling``` library to execute the notebook.

### Lab 6 (10/21):
Lab6 folder is now available. No extra libraries are required. In this lab, we will
- explore the ```scikit-learn```'s implementation of the Support Vector Classifier;
- revisit the 2016 elections dataset from Lab2 (and Homework 1).

### Lab 7 (10/28):
Lab7 folder is now availbale. In this section, we will:
- derive the bias-variance trade-off in statistical learning theory; 
- explore the resulting formula using ```sklearn```'s SVC with RBF kernels;
- predict the outcome of the upcoming Presidential Elections using the tree model from HW1!

### Lab 8 (11/04):
Lab8 folder is now up! ***Disclaimer***: you will not be able to run the notebook on your local machine this time as it depends on functions you will write in Homework2 (hidden on my computer in a different folder). I will upload these files after the due date. Today, we will talk about multiclass classification; in particular,
- combinations of binary classifiers in a multiclass setting (*one-vs-one, one-vs-all*);
- discuss evaluation metrics for multiclass models;
- extend the ROC space to handle these scenarios.

### Lab 9 (11/11):
Lab9 folder is now available! In it, you will find:
 - K-Means clustering algorithms: theory, examples, and extensions (K-Means++, Global K-Means);
 - Agglomerative clustering as the most common hierarchical approach: different linkage functions in theory and practice.
 
### Lab 10 & 11 (11/18-11/25):
These two labs focus on text classification. In particular:
 - Naive Bayes: probabilistic formulation, examples (Gaussian, Bernoulli, Multinomial);
 - full text classification pipeline using ```nltk``` library and Naive Bayes models from scikit-learn.
To execute this notebook, you will need to install ```nltk``` (natural language toolkit) and download the following modules: (1) averaged perceptron tagger, (2) punkt, (3) stopwords, (4) wordnet. To do this, type ```nltk.download()``` in your interactive python session, which will open a GUI with all available modules. In addition, you will need to download two data files (```artists-data.csv``` and ```lyrics-data.csv```) from the Resources tab on NYU classes (these files were to big to upload here).

### Lab 12 (12/02):
The last lab will be dedicated to studying Restricted Boltzmann Machines as Graphical Models, their assumptions, training, and inference. We will review aspects of Markov chains necessary for the discussion about Gibbs sampling. Finally, we will finish by showcasing a use-case of RBMs as data generators.
