
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Invoke libraries
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
import numpy as np
import pandas as pd
import mne
import glob
from sklearn import preprocessing, metrics

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ParameterGrid
import timeit

from sklearn.model_selection import RandomizedSearchCV

import hyperopt
from hyperopt import Trials,fmin,STATUS_OK, hp, tpe
from sklearn.model_selection import cross_val_score
import warnings

#suppress warnings. This is specifically to suppress the zero division warning in the precision calculation step
warnings.filterwarnings('ignore')


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Functions
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#Convert edf data to dataframe
def edf2Df(edfFilePath, classVar):
    tempEdf = mne.io.read_raw_edf(edfFilePath)
    tempHeader = ','.join(tempEdf.ch_names)
    tempColumnNames = list(tempHeader.split(","))
    edfAsDf = pd.DataFrame(tempEdf.get_data().T, columns = tempColumnNames)    
    edfAsDf['classVar'] = classVar
    return(edfAsDf)

#Define objective function for smbo
def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = model #defined at basic grid search section    
    else:
        return 0
    accuracy = cross_val_score(clf,x_train, y_train).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}

#Run the experiments with multiple optmization methods
def runExperiment(optMethod, noOfValues, cv):
    #Define gamma and C values
    gammaValues = np.logspace(start = -3, stop = 3, num=noOfValues, endpoint=True, base=10.0, dtype=None, axis=0)
    Cvalues = np.linspace(start = 1, stop = 100, num=noOfValues)
    noOfIterations = round(((noOfValues**2)+1)/10,None)
    
    #set up grid
    grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': gammaValues, 'C': Cvalues}]
    
    #Define search method
    if(optMethod == 'gridSearch'):        
        search = GridSearchCV(estimator=model, param_grid=grid ,cv=cv, verbose = 1)
        searchSpace = (((noOfValues**2)+1)*9)
    elif(optMethod == 'randomGridSearch'):
        search = RandomizedSearchCV(estimator=model, param_distributions=grid , cv=cv, n_iter=noOfIterations , verbose = 1)
        searchSpace = (noOfIterations*9)
    elif(optMethod == 'smbo'):
        bayes_trials = Trials()
        MAX_EVALS = noOfValues        
        #place holder search space value
        searchSpace = (((noOfValues**2)+1)*9)
        #Define seach space for smbo
        search_space = hp.choice('classifier_type', [
            {
                'type': 'svm',        
                'C': hp.choice('SVM_C', Cvalues),
                'kernel': hp.choice('kernel', ['linear', 'rbf']),
                'gamma': hp.choice('gamma', gammaValues)
            }
        ])
    
    #execute search and Compute the time to run the gridsearch
    if(optMethod == ('gridSearch') or optMethod == 'randomGridSearch'):
        start = timeit.default_timer()
        result = search.fit(x_train, y_train)
        stop = timeit.default_timer()
        timetaken = stop-start
    elif(optMethod == 'smbo'):
        result = fmin(fn = objective, space = search_space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, verbose = True)    
    
    #Evaluate performance on training data
    if(optMethod == ('gridSearch') or optMethod == 'randomGridSearch'):        
        accuracy = result.best_score_
        bestKernel = result.best_params_['kernel']
        bestC = result.best_params_['C']
        bestGamma = result.best_params_['gamma']
        numOfValues = noOfValues
        bestEstimator = result.best_estimator_
    elif(optMethod == 'smbo'):
        start = timeit.default_timer()
        hyperoptResult = hyperopt.space_eval(space = search_space, hp_assignment = result)
        stop = timeit.default_timer()
        timetaken = stop-start
        #placeholder accuracy
        accuracy = None
        bestKernel = hyperoptResult['kernel']
        bestC = hyperoptResult['C']
        bestGamma = hyperoptResult['gamma']
        numOfValues = noOfValues             
        bestEstimator = SVC(kernel = bestKernel, C = bestC, gamma = bestGamma)
        bestEstimator.fit(x_train, y_train)
        
    #Predict on test data    
    y_true, y_pred = y_test, bestEstimator.predict(x_test)
    
    #Evaluate the performance on test data    
    testAccuracy = metrics.accuracy_score(y_test, y_pred)
    testRecallGood = metrics.recall_score(y_test, y_pred, average="binary", pos_label="good")
    testRecallBad = metrics.recall_score(y_test, y_pred, average="binary", pos_label="bad")
    testRecallAvg = np.mean([testRecallGood,testRecallBad])
    testPrecisionGood = metrics.precision_score(y_test, y_pred,  average="binary", pos_label="good")
    testPrecisionBad = metrics.precision_score(y_test, y_pred,  average="binary", pos_label="bad")
    testPrecisionAvg = np.mean([testPrecisionGood, testPrecisionBad])
    
    
    tempDf = {'optMethod':[optMethod], 'numOfValues': [noOfValues], 'searchSpace': [searchSpace],  
             'bestKernel': [bestKernel], 'bestC': [bestC], 'bestGamma': [bestGamma], 'optimisationTime': [timetaken], 'accuracyTrng': [accuracy], 'accuracyTest': [testAccuracy], 'goodRecallTest': [testRecallGood], 'badRecallTest': [testRecallBad], 'goodPrecisionTest': [testPrecisionGood], 'badPrecisionTest': [testPrecisionBad], 'meanPrecisionTest': [testPrecisionAvg]}        
    return (tempDf)


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Data processing
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Iterate through all the individual files and create a master df
#df=pd.read_csv(folderPath +"masterDf2.csv")
pathToData = "C:/Users/tpf3706/Documents/"    
goodPath = (pathToData + '/data/good/*_2.edf')
goodDocs = glob.glob(goodPath)

badPath = (pathToData + '/data/bad/*_2.edf')
badDocs = glob.glob(badPath)

#create an empty master dataframe
masterDf = pd.DataFrame()    

#Iteratively append master dataframe with resting and action data
for doc in goodDocs:        
    print(doc)   
    tempDf = edf2Df(edfFilePath = doc, classVar = 1)
    tempDf2 = pd.DataFrame(tempDf.apply(lambda x: x.mean(axis = 0)))    
    masterDf = masterDf.append(tempDf2.T)
    tempDf = pd.DataFrame()
    tempDf2 = pd.DataFrame()  


for doc in badDocs:        
    print(doc)   
    tempDf = edf2Df(edfFilePath = doc, classVar = 2)
    tempDf2 = pd.DataFrame(tempDf.apply(lambda x: x.mean(axis = 0)))    
    masterDf = masterDf.append(tempDf2.T)
    tempDf = pd.DataFrame()
    tempDf2 = pd.DataFrame()  

#Drop the variable "ECG ECG" as that is not needed for this study
masterDf = masterDf.drop('ECG ECG',axis=1)    

#Normalize the variables except the class var
x = masterDf.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df2 = pd.DataFrame(x_scaled, columns=masterDf.columns)

#Recode classvar to intuitive name
df2["classVar"].replace({0.0: 'good', 1.0: 'bad'}, inplace=True)
masterDfNorm = df2

#export the master dataframe as csv to disk. Drop header for NeuCom reasons    
masterDfNorm.to_csv(pathToData + 'masterDfNorm.csv', index = False, header = True)

#Split data into training and test
x=masterDfNorm.iloc[:,:-1]
y=masterDfNorm.iloc[:,-1]
x_train,x_test, y_train, y_test=train_test_split(x , y, test_size=0.30, stratify = y, random_state = 123)


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Set lookup values 
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#Build results dataframe
experimentResults = pd.DataFrame(columns = ["optMethod", "optimisationTime", "accuracyTrng", "accuracyTest", "goodRecallTest", "badRecallTest","meanRecallTest", "goodPrecisionTest", "badPrecisionTest", "meanPrecisionTest"])

#Define model
model=SVC()

#Define cross validation
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)

#Define list of gamma and C values
noOfValues = [*range(105, 200, 5)]
#noOfValues = [*range(10, 20, 5)]

#Set up the grid for the search space
#grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': gammaValues, 'C': Cvalues}]

#Define list of optimization methods to experiment
optMethod = ['gridSearch', 'randomGridSearch', 'smbo']
#optMethod = ['smbo']

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Experiments
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

for m in optMethod:
    for n in noOfValues:
        print('Now running %s method with %s number of values' % (m, n))
        tempDf = runExperiment(optMethod = m, noOfValues = n, cv = cv)
        experimentResults  = pd.concat([experimentResults, pd.DataFrame(tempDf)])
    
#export the results
experimentResults.to_csv(pathToData + 'experimentResults.csv', index = False, header = True)
