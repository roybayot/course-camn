
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import sys
import os

import sklearn
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from scipy import stats

import re

reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
sys.setdefaultencoding("UTF-8")

defaultFileNames = {'age': 'age-important-words-using-info-gain.txt',
                    'gender': 'gender-important-words-using-info-gain.txt'
                   }


# In[2]:

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))


# In[3]:

def clean_all_text(allText, numLines):
    clean_train_data = []
    for i in xrange(0, numLines):
        clean_train_data.append(clean_text(allText[i]))
    return clean_train_data


# In[4]:

def featureSelection(train_x, task, train_y):
    rows, cols = train_x.shape
    top_info_words_numbers = [100, 200, 300, 500, 700, 1000, 2000, 5000, 7000, 8000, 9000, 10000, cols-1]
    top_info_words_numbers =  sorted(top_info_words_numbers, reverse=True)

    feature_selection_result = {}
    
    task_to_filenames = {'age': ['age-important-words-using-info-gain.txt', 'age-important-words-using-gain-ratio.txt'],
                 'gender': ['gender-important-words-using-info-gain.txt', 'gender-important-words-using-gain-ratio.txt']
                }
    
    filenames = task_to_filenames[task]
    for filename in filenames:
        with open(filename) as f:
            alist = [line.rstrip() for line in f]
        all_indices_ranked = alist[0].split(',')
        all_indices_ranked = [int(x) for x in all_indices_ranked]
        all_indices_ranked = [x-1 for x in all_indices_ranked]
        
        list_of_scores = []
        for num_info_words in top_info_words_numbers:
            clf = svm.SVC(kernel='linear', C=1)
            scoring_function = 'accuracy'
            
            xx = [all_indices_ranked[x] for x in range(0, num_info_words)]
            xx = tuple(xx)
            smaller_train_x = train_x[:, xx]

            scores = cross_validation.cross_val_score(clf, smaller_train_x, train_y, cv=10, scoring=scoring_function)
            list_of_scores.append(scores)
            
            feature_selection_result[filename] = list_of_scores
    return feature_selection_result
            


# In[5]:

def doSVMwithRBF(smaller_train_x, train_y, task):
    params = {'age': {'gammas': [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4],
                      'C': [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4]},
              'gender': {'gammas': [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4],
                      'C': [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4]}
             }
    
    gammas = params[task]['gammas']
    C = params[task]['C']
    
    list_of_scores = []
    results_with_params = {}
    
    for g in gammas:
        for one_C in C:
            clf = svm.SVC(kernel='rbf', gamma=g, C=one_C)
            scoring_function = 'accuracy'
            scores = cross_validation.cross_val_score(clf, smaller_train_x, train_y, cv=10, scoring=scoring_function)
            list_of_scores.append(scores)
            label = str(g)+','+str(one_C)
            results_with_params[label] = scores
    
    svm_rbf_result_list_of_scores = list_of_scores
    
    return svm_rbf_result_list_of_scores, results_with_params


# In[6]:

def calculatePValue(input_dictionary):
    p_values_dictionary = {}
    for each_key in input_dictionary.keys():
        list_of_scores = input_dictionary[each_key]
        i = range(0,len(list_of_scores))
        list_of_pvalues = []
        for x, i  in zip(list_of_scores,i):
            z_stat, p_val = stats.ranksums(list_of_scores[0], x)
            list_of_pvalues.append( p_val)
        p_values_dictionary[each_key] = list_of_pvalues
    return p_values_dictionary
        


# In[7]:

def getAccuracies(feature_selection_result):
    accuracies_dictionary = {}
    for each_key in feature_selection_result.keys():
        list_of_accuracies = feature_selection_result[each_key]
        accuracies = [a.mean() for a in list_of_accuracies]
        accuracies_dictionary[each_key] = accuracies
    return accuracies_dictionary


# In[8]:

def getListOfRankedFeatures(train_x, num_features, task, fileNames=defaultFileNames):
    fileName = fileNames[task]
    
    with open(fileName) as f:
        alist = [line.rstrip() for line in f]
    all_indices_ranked = alist[0].split(',')
    all_indices_ranked = [int(x) for x in all_indices_ranked]
    all_indices_ranked = [x-1 for x in all_indices_ranked]    
    xx = [all_indices_ranked[x] for x in range(0, num_features)]
    xx = tuple(xx)
    return xx
    


# In[9]:

def getSmallerTrainingSet(train_x, task, num_features, fileNames=defaultFileNames):
    xx = getListOfRankedFeatures(train_x, num_features, task, fileNames)
    smaller_train_x = train_x[:, xx]    
    return smaller_train_x


# In[10]:

def doSVMwithPoly(train_x, train_y, task):
    params = {'age': {'degrees': [1,2,3],
                      'C': [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4]},
              'gender': {'degrees': [1,2,3],
                      'C': [10**-4, 10**-1, 1, 10**1, 10**4]}
             }
    
    svm_poly_result = {}
    
    degrees = params[task]['degrees']
    C = params[task]['C']
    
    list_of_scores = []
    results_with_params = {}
    for degree in degrees:
        for one_C in C:
            clf = svm.SVC(kernel='poly', degree=degree, coef0=one_C, gamma=1)
            scoring_function = 'accuracy'
            scores = cross_validation.cross_val_score(clf, smaller_train_x, train_y, cv=10, scoring=scoring_function)
            list_of_scores.append(scores)
            label = str(degree)+','+str(one_C)
            results_with_params[label] = scores
    
    svm_poly_result_list_of_scores = list_of_scores
    
    return svm_poly_result_list_of_scores, results_with_params


# In[11]:

def doFeatureWithResultsofOther(train_x, train_y_task, train_y_prior, task, optimal_params):
    num_features_dict = {'age': 9000,
                    'gender': 7000}
    # optimal_params = {'age': ['poly', 3, 10, 1], 'gender': ['poly', 2, 10000, 1]}
    
    accuracies_dictionary = {}
    
    age_classification_kernel_type = optimal_params["age"][0]
    p1 = optimal_params["age"][1]
    p2 = optimal_params["age"][2]
    
    gender_classification_kernel_type = optimal_params["age"][0]
    q1 = optimal_params["age"][1]
    q2 = optimal_params["age"][2]
    
    
    if age_classification_kernel_type == "poly":
        clf_age_classification = svm.SVC(kernel='poly', degree=p1, coef0=p2, gamma=1)
    else:
        clf_age_classification = svm.SVC(kernel='rbf', degree=q1, gamma=q2)
        
        
    if gender_classification_kernel_type == "poly":
        clf_gender_classification = svm.SVC(kernel='poly', degree=p1, coef0=p2, gamma=1)
    else:
        clf_gender_classification = svm.SVC(kernel='rbf', degree=q1, gamma=q2)
    

    # age classification:
    #    clf1 = svm.SVC(kernel='poly', degree=3, coef0=10, gamma=1)
    
    # gender classification:
    #    clf2 = svm.SVC(kernel='poly', degree=2, coef0=10000, gamma=1)
    
    scoring_function = 'accuracy'
    
    if task == "age":
        prior = "gender"
        clf_prior = clf_gender_classification
        clf_task = clf_age_classification
    else:
        prior = "age"
        clf_prior = clf_age_classification
        clf_task = clf_gender_classification

    num_features_task = num_features_dict[task]
    num_features_prior = num_features_dict[prior]
    
    smaller_training_set_for_prior = getSmallerTrainingSet(train_x, task, num_features_prior) 
    smaller_training_set_for_task = getSmallerTrainingSet(train_x, task, num_features_task)
    
    clf_prior.fit(smaller_training_set_for_prior, train_y_prior)
    results_prior = clf_prior.predict(smaller_training_set_for_prior)
    
    combined = np.column_stack((smaller_training_set_for_task, results_prior))
    
    scores = cross_validation.cross_val_score(clf_task, combined, train_y_task, cv=10, scoring=scoring_function)
    accuracies_dictionary[task] = scores
    
    return accuracies_dictionary


# In[12]:

def doSVMwithPreprocessedText(train, task, num_features, optimal_params):
    accuracy_dictionary = {}
    
    age_classification_kernel_type = optimal_params["age"][0]
    p1 = optimal_params["age"][1]
    p2 = optimal_params["age"][2]
    
    gender_classification_kernel_type = optimal_params["age"][0]
    q1 = optimal_params["age"][1]
    q2 = optimal_params["age"][2]
    
    
    if age_classification_kernel_type == "poly":
        clf_age_classification = svm.SVC(kernel='poly', degree=p1, coef0=p2, gamma=1)
    else:
        clf_age_classification = svm.SVC(kernel='rbf', degree=q1, gamma=q2)
        
        
    if gender_classification_kernel_type == "poly":
        clf_gender_classification = svm.SVC(kernel='poly', degree=p1, coef0=p2, gamma=1)
    else:
        clf_gender_classification = svm.SVC(kernel='rbf', degree=q1, gamma=q2)
    
    
    newFileNames = {'age': 'new-age-important-words-using-info-gain.txt',
                 'gender': 'new-gender-important-words-using-info-gain.txt'
                }
    
    train_y = train[task]
    
    clean_train_data = []
    urls = []
    hashtags = []
    num_text = train["text"].size
    for i in xrange( 0, num_text):
        one_clean_line = clean_text( train["text"][i] )
        new_clean_line = ""
        #replacing links
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', one_clean_line)
        for one_url in url:
            new_clean_line = one_clean_line.replace(one_url, " LINK_HERE ")
            one_clean_line = new_clean_line
        urls.append(url)
    
        hashtag = re.findall('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', one_clean_line)
    
        for one_hashtag in hashtag:
            new_clean_line = one_clean_line.replace(one_hashtag, " HASHTAG_HERE ")
            one_clean_line = new_clean_line
        
        hashtags.append(hashtag)
        clean_train_data.append( one_clean_line )
    
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None)
    train_x = vectorizer.fit_transform(clean_train_data)
    train_x = train_x.toarray()
    
    smaller_train_x = getSmallerTrainingSet(train_x, task, num_features, newFileNames)
    
    if task == "age":
        clf = clf_age_classification
    else:
        clf = clf_gender_classification
    
    scores = cross_validation.cross_val_score(clf, smaller_train_x, train_y, cv=10, scoring='accuracy')
    
    accuracy_dictionary[task] = scores
    return accuracy_dictionary


# In[13]:

tasks = ["age", "gender"]
allResults = {}

datafile = "summary-english-truth.txt"
train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)
clean_train_data = clean_all_text(train["text"], train["text"].size)

vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None)
train_x = vectorizer.fit_transform(clean_train_data)
train_x = train_x.toarray()


for task in tasks:
    train_y = train[task]

    # experiment 1.1: Feature Selection
    feature_selection_result = featureSelection(train_x, task, train_y)
    accuracies_dictionary = getAccuracies(feature_selection_result)
    p_values_dictionary = calculatePValue(feature_selection_result)

    
    label=task+"-experiment-1.1"
    allResults[label] = feature_selection_result
    # experiment 1.2: SVM Poly
    num_features_dictionary = {'age': 9000,
                               'gender': 7000
                              }
    num_features = num_features_dictionary[task]
    smaller_train_x = getSmallerTrainingSet(train_x, task, num_features)

    svm_poly_result_list_of_scores, svm_poly_results_with_params = doSVMwithPoly(smaller_train_x, train_y, task)
    svm_poly_accuracies_dictionary = getAccuracies(svm_poly_results_with_params)
    
    label=task+"-experiment-1.2"
    allResults[label] = svm_poly_results_with_params

    #  experiment 1.3: SVM RBF
    svm_rbf_result_list_of_scores, svm_rbf_results_with_params = doSVMwithRBF(smaller_train_x, train_y, task)
    svm_rbf_accuracies_dictionary = getAccuracies(svm_rbf_results_with_params)

    label=task+"-experiment-1.3"
    allResults[label] = svm_rbf_results_with_params
    
    
#    doRandomForest()
#    doBoosting()


# In[14]:

# optimal params taken when looking at stats from the others
optimal_params = {'age': ['poly', 3, 10, 1], 'gender': ['rbf', 10, 1, 1]} 
for task in ["age", "gender"]:   
    # experiment 2 7000 gender + age info, 9000 age + gender info
    if task == "age":
        prior = "gender"
    else:
        prior = "age"
    
    train_y_task = train[task]
    train_y_prior = train[prior]
    res_for_other = doFeatureWithResultsofOther(train_x, train_y_task, train_y_prior, task, optimal_params)

    label=task+"-experiment-2"
    allResults[label] = res_for_other

    # experiment 3: turning hashtags/hyperlinks to HASHTAG_HERE and LINK_HERE
    
    res_with_preprocessed_text = doSVMwithPreprocessedText(train, task, num_features, optimal_params)
    
    label=task+"-experiment-3"
    allResults[label] = res_with_preprocessed_text


# In[15]:

allResults["age-experiment-2"]["age"].mean()


# In[16]:

#part where we change compare results

# Experiment 1.1: feature selection

results = allResults["age-experiment-1.1"]
age_feat_select_accuracies_dictionary = getAccuracies(results)
age_feat_select_p_values_dictionary = calculatePValue(results)

results = allResults["gender-experiment-1.1"]
gender_feat_select_accuracies_dictionary = getAccuracies(results)
gender_feat_select_p_values_dictionary = calculatePValue(results)


# In[ ]:




# In[17]:

def extractMeanAccuraciesForPolyAndRBF(results, Cs=[], gammas=[]):
    svm_mean_accuracy = {}
    svm_list_of_accuracies = []
    for gamma in gammas:
        for C in Cs:
            label = str(gamma)+","+str(C)
            svm_mean_accuracy[label] = results[label].mean()
            svm_list_of_accuracies.append(results[label])
    return svm_mean_accuracy, svm_list_of_accuracies


# In[18]:

def makePValMatrix(list_of_accuracies):
    list_length = len(list_of_accuracies)
    p_value_matrix = np.zeros((list_length, list_length))
    i = range(0, list_length)
    #sig values
    for treatment1,x in zip(svm_poly_age_list_of_accuracies,i):
        for treatment2,y in zip(svm_poly_age_list_of_accuracies,i):
            z_stat, p_val = stats.ranksums(treatment1, treatment2)
            p_value_matrix[x,y] = p_val
    return p_value_matrix


# In[19]:

def turnPValMatrixToExcel(fileName, p_value_matrix, list_of_accuracies):
    df = pd.DataFrame(data = p_value_matrix, columns=list_of_accuracies)
    df.index = list_of_accuracies
    null_disproved = df[df < 0.05]
    null_disproved.to_csv(fileName, sep='\t', encoding='utf-8')


# In[20]:

# Experiment 1.2: SVM Poly 

# Age
results = allResults["age-experiment-1.2"]

# getting mean accuracy for each and putting in a dictionary        
svm_poly_age_mean_accuracy, svm_poly_age_list_of_accuracies = extractMeanAccuraciesForPolyAndRBF(results, [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4] , [1,2,3])
svm_poly_age_mean_accuracy = [a.mean() for a in svm_poly_age_list_of_accuracies]
p_val_matrix = makePValMatrix(svm_poly_age_list_of_accuracies)
turnPValMatrixToExcel("pval-null-disproved-age-poly.csv", p_val_matrix, svm_poly_age_mean_accuracy)


# In[21]:

# Gender
results = allResults["gender-experiment-1.2"]
svm_poly_gender_mean_accuracy, svm_poly_gender_list_of_accuracies = extractMeanAccuraciesForPolyAndRBF(results, [10**-4, 10**-1, 1, 10**1, 10**4] , [1,2,3])
svm_poly_gender_mean_accuracy = [a.mean() for a in svm_poly_gender_list_of_accuracies]
p_val_matrix = makePValMatrix(svm_poly_gender_list_of_accuracies)
turnPValMatrixToExcel("pval-null-disproved-gender-poly.csv", p_val_matrix, svm_poly_gender_mean_accuracy)


# In[22]:

# Experiment 1.3: SVM RBF 

# Age
results = allResults["age-experiment-1.3"]
svm_rbf_age_mean_accuracy, svm_rbf_age_list_of_accuracies = extractMeanAccuraciesForPolyAndRBF(results, [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4] , [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4] )
svm_rbf_age_mean_accuracy = [a.mean() for a in svm_rbf_age_list_of_accuracies]
p_val_matrix = makePValMatrix(svm_rbf_age_list_of_accuracies)
turnPValMatrixToExcel("pval-null-disproved-age-rbf.csv", p_val_matrix, svm_rbf_age_mean_accuracy)


# In[23]:

# Gender
results = allResults["gender-experiment-1.3"]
svm_rbf_gender_mean_accuracy, svm_rbf_gender_list_of_accuracies = extractMeanAccuraciesForPolyAndRBF(results, [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4] , [10**-4, 10**-3, 10**-1, 1, 10**1, 10**3, 10**4] )
svm_rbf_gender_mean_accuracy = [a.mean() for a in svm_rbf_gender_list_of_accuracies]
p_val_matrix = makePValMatrix(svm_rbf_gender_list_of_accuracies)
turnPValMatrixToExcel("pval-null-disproved-gender-rbf.csv", p_val_matrix, svm_rbf_gender_mean_accuracy)


# In[24]:

# Experiment 2: Using the other classification result as prior
# Age
age_classification_results_with_priors = allResults["age-experiment-2"]["age"]
best_result_age_classification = allResults["age-experiment-1.2"]['3,10']
z_stat, p_val = stats.ranksums(age_classification_results_with_priors, best_result_age_classification)
[age_classification_results_with_priors.mean(), best_result_age_classification.mean(), p_val]


# similar since greater than 0.05


# In[25]:

# Gender
gender_classification_results_with_priors = allResults["gender-experiment-2"]["gender"]
best_result_gender_classification = allResults["gender-experiment-1.3"]['1,10']
z_stat, p_val = stats.ranksums(gender_classification_results_with_priors, best_result_gender_classification)
[gender_classification_results_with_priors.mean(), best_result_gender_classification.mean(), p_val]

# similar since greater than 0.05


# In[26]:

# Experiment 3: Different text
# Age
age_classification_with_string_sub = allResults["age-experiment-3"]["age"]
best_result_age_classification = allResults["age-experiment-1.2"]['3,10']
z_stat, p_val = stats.ranksums(age_classification_with_string_sub, best_result_age_classification)
[age_classification_with_string_sub.mean(), best_result_age_classification.mean(), p_val]
# similar since greater than 0.05


# In[27]:

# Gender
gender_classification_with_string_sub = allResults["gender-experiment-3"]["gender"]
best_result_gender_classification = allResults["gender-experiment-1.3"]['1,10']
z_stat, p_val = stats.ranksums(gender_classification_with_string_sub, best_result_gender_classification)
[gender_classification_with_string_sub.mean(), best_result_gender_classification.mean(), p_val]
# similar since greater than 0.05


# In[28]:

# Additional comparisons:
# Comparing age_classification_results_with_priors and age_classification_with_string_sub
z_stat, p_val = stats.ranksums(age_classification_results_with_priors, age_classification_with_string_sub)
[ age_classification_results_with_priors.mean(), age_classification_with_string_sub.mean(), p_val]


# In[29]:

# Comparing gender_classification_results_with_priors and gender_classification_with_string_sub
z_stat, p_val = stats.ranksums(gender_classification_results_with_priors, gender_classification_with_string_sub)
[ gender_classification_results_with_priors.mean(), gender_classification_with_string_sub.mean(), p_val]


# In[30]:

# Experiment 4: Random Forest

from sklearn.ensemble import RandomForestClassifier

num_estimators = [10, 100, 1000, 2000, 5000, 10000]

for task in ["age", "gender"]:
    oneResult = {}
    for x in num_estimators:
        train_y = train[task]
        clf = RandomForestClassifier(n_estimators=x)
        scores = cross_validation.cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
        oneResult[str(x)] = scores
    label = task+"-experiment-4"
    allResults[label] = oneResult


# In[31]:

x = [allResults["age-experiment-4"][a].mean() for a in allResults["age-experiment-4"].keys()] 
x


# In[32]:

x = [allResults["gender-experiment-4"][a].mean() for a in allResults["gender-experiment-4"].keys()] 
x


# In[33]:

# Experiment 5: AdaBoost

from sklearn.ensemble import AdaBoostClassifier

#num_estimators = [10, 100, 1000, 10000, 100000]
num_estimators = [50, 100, 150, 200, 250]

for task in ["age", "gender"]:
    oneResult = {}
    for x in num_estimators:
        train_y = train[task]
        clf = AdaBoostClassifier(n_estimators=x)
        scores = cross_validation.cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
        oneResult[str(x)] = scores
    label = task+"-experiment-5"
    allResults[label] = oneResult    


# In[34]:

x = [allResults["age-experiment-5"][a].mean() for a in allResults["age-experiment-5"].keys()]
x


# In[35]:

x = [allResults["gender-experiment-5"][a].mean() for a in allResults["gender-experiment-5"].keys()]
x


# In[ ]:



