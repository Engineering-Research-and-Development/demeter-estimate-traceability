"""
Estimate Milk Quality Module - Random Forest Training/Test

Author: Luigi di Corrado
Mail: luigi.dicorrado@eng.it
Date: 12/10/2020
Company: Engineering Ingegneria Informatica S.p.A.

Introduction : This module is used to perform the training of the Random Forest algorithm,
               and create the models that will be used on prediction process to 
               estimate the milk quality using three different classes of classification:
                   - Low
                   - Medium
                   - High

Function     : initConfiguration

Description  : Set the current work directory and initialize the settings for Logger 
               and Random Forest Training/Prediction
               
Parameters   : string     confFile      contains the path to rfConf.properties to load
               string     workPath      path to work directory
               
Return       : 



Function     : measure (TP, FP, TN, FN, MFPR, MTPR, MPPV, MACC)

Description  : Execute the calculation of the following metrics data
               TP    - True Positive
               FP    - False Positive
               TN    - True Negative
               FN    - False Negative
               MFPR  - False Positive Rate (Average)
               MTPR  - True Positive Rate (Average)
               MPPV  - Precision (Average)
               MACC  - Accuracy (Average)
               
Parameters   : dataframe  y_actual      contains the real solution data provided by the user
               dataframe  y_predict     contains the test prediction data provived by the training phase
               
Return       : float TP
               float FP
               float TN
               float FN
               float MFPR
               float MTPR
               float MPPV
               float MACC



Function     : execRFTraining

Description  : Converts the JSON input into a dataframe object, then process the data and start the training phase 
               for Random Forest algorithm.
               The data is splitted using 80% of the rows for Training and 20% for Testing.
               The random state "rs" argument is used to provide randomic rows while splitting the data.
               Once the training is complete a test prediction is executed on the dedicated rows and the
               models are saved into the configured folder.
               The metrics are calculated using the measure function.
               All the tested data and metrics are returned as output using JSON format.
               The JSON output contains the following roots:
                   - rawData           Contains data about RAW milk
                   - processedData     Contains data about PROCESSED milk
                   - metricsData       Contains all the metrics data of the algorithm
               
Parameters   : str   JsonData           - String that contains the JSON data to be processed
               int   randomState        - Random state value for test phase, it choose random rows
               int   estimatorsNumbers  - Numbers of estimators to use on training phase
               
Return       : str   jsonResult   - String that contains all the JSON data to output



Function     : execRFPrediction

Description  : Converts the JSON input into a dataframe object, then process the data
               and load the models before start the prediction using Random Forest.
               At the end of the process, the output data is converted into JSON.
               The JSON output contains the following roots:
                   - rawData           Contains data about RAW milk
                   - processedData     Contains data about PROCESSED milk
               
Parameters   : str   JsonData     - String that contains the JSON data to be processed
               
Return       : str   jsonResult   - String that contains all the JSON data to output

"""

import pandas as pd
import numpy as np
import datetime as dt
import joblib
import sys
import os
import json
import configparser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from MQLogger import log

class MilkQualityRandomForest:
    def __init__(self):
       self.config = configparser.ConfigParser()
       self.myLog = log()
    
    def initConfiguration(self, confFile, workpath):
      os.chdir(workpath)
      self.config.read(confFile)
      self.myLog.initConfiguration(confFile)
      
    def measure(self, y_actual, y_predict):
        cnf_matrix = confusion_matrix(y_actual,y_predict)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = (TP/(TP+FN))*100
        TPR = [round(val,2) for val in TPR]
        MTPR = round((np.sum(TPR)/len(TPR)),2)
        # Precision or positive predictive value
        PPV = (TP/(TP+FP))*100
        PPV = [round(val,2) for val in PPV]
        MPPV = round((np.sum(PPV)/len(PPV)),2)
        # Fall out or false positive rate
        FPR = (FP/(FP+TN))*100
        FPR = [round(val,2) for val in FPR]
        MFPR = round((np.sum(FPR)/len(FPR)),2)
        # Overall accuracy
        ACC = ((TP+TN)/(TP+FP+FN+TN))*100
        ACC = [round(val,2) for val in ACC]
        MACC = round((np.sum(ACC)/len(ACC)),2)

        return(TP, FP, TN, FN, MFPR, MTPR, MPPV, MACC)

    def execRFTraining(self, JsonData, randomState, estimatorsNumbers):        
        # GET FUNCTION NAME
        functionName = sys._getframe().f_code.co_name
        # Estimate Milk Quality - Training and Testing
        with self.myLog.error_debug():
            try:
                self.myLog.writeMessage("Working directory set to: "+os.path.abspath(os.getcwd()),"DEBUG",functionName)
                self.myLog.writeMessage("Module temp directory in use: "+os.path.join(os.path.dirname(__file__)),"DEBUG",functionName)
                # LOADING CONFIGURATION FROM PROPERTIES FILE
                self.myLog.writeMessage("Loading configuration from properties file...","DEBUG",functionName)

                # TRAINING SETTINGS
                #rs = int(config.get('PyRandomForest', 'milkquality.randomForest.trainingSettings.randomState'))  # Random State
                rs = randomState
                self.myLog.writeMessage("Random state set at: "+str(rs),"DEBUG",functionName)
                #estimators = int(config.get('PyRandomForest', 'milkquality.randomForest.trainingSettings.estimators'))
                estimators = estimatorsNumbers
                self.myLog.writeMessage("Estimators set at: "+str(estimators),"DEBUG",functionName)
        
                # MODEL SETTINGS
                ModelFolderPath = self.config.get('PyRandomForest', 'milkquality.randomForest.commonSettings.modelfilePath')
                self.myLog.writeMessage("Model directory set at: "+ModelFolderPath,"DEBUG",functionName)
                PrefixModel = self.config.get('PyRandomForest', 'milkquality.randomForest.commonSettings.modelNamePrefix')
                self.myLog.writeMessage("Model name prefix set at: "+PrefixModel,"DEBUG",functionName)
                TraceabilityModelName = PrefixModel + 'Model'

        
                self.myLog.writeMessage('Preparing to execute Random Forest training phase of Estimate Milk Quality ...',"INFO",functionName)

                # Check if directory exists, or create it
                try:
                    self.myLog.writeMessage('Checking models directory ...',"DEBUG",functionName)
                    os.makedirs(ModelFolderPath)
                    self.myLog.writeMessage('Warning: Directory not found! ',"WARN",functionName)
                    self.myLog.writeMessage('Created directory: ' + os.path.realpath(ModelFolderPath), "DEBUG",functionName)
                except FileExistsError:
                    # Directory already exists
                    self.myLog.writeMessage('Directory already exists: ' + os.path.realpath(ModelFolderPath), "DEBUG",functionName)
                    pass
                
                # Dataset preparation
                self.myLog.writeMessage('Loading dataset ...',"DEBUG",functionName)
                JsonObj = json.loads(JsonData)
                dataframe = pd.DataFrame(JsonObj)
                #dataframe = dataframe.set_index('Index')
                cols = ['AciditySH', 'Casein','Density', 'Fat', 'Freezing Point mC', 'Lactose',
                        'Protein', 'SNF', 'Urea', 'Actual Quality']
                dataset = dataframe[cols]
                self.myLog.writeMessage('Dataset successfully loaded!',"DEBUG",functionName)

                # Random forest classification preparation
                # Defining the values: X contains values and actual solutions
                #                      y contain only solution column 
                #                      i is the index column
                self.myLog.writeMessage('Values definintion for classification ...',"DEBUG",functionName)
                Traceability_X = dataset.iloc[:, 0:9].values
                Traceability_y = dataset.iloc[:, 9].values
                Traceability_i = dataset.index.values
                self.myLog.writeMessage('Values definintion for classification successfully completed!',"DEBUG",functionName)

                # Split data into training and test, and keep the index for each of them
                self.myLog.writeMessage('Defining training and test data ...',"DEBUG",functionName)
                Traceability_X_train, Traceability_X_test, Traceability_y_train, Traceability_y_test, Traceability_i_train, Traceability_i_test = train_test_split(
                Traceability_X, Traceability_y, Traceability_i, test_size = 0.20, random_state= rs)
                self.myLog.writeMessage('Training and test data definition completed!',"DEBUG",functionName)

                # Training phase and testing
                self.myLog.writeMessage('Executing training and testing ...',"DEBUG",functionName)
                TraceabilityClassifier = RandomForestClassifier(n_estimators = estimators, random_state = rs)
                TraceabilityClassifier.fit(Traceability_X_train, Traceability_y_train)
                Traceability_y_pred = TraceabilityClassifier.predict(Traceability_X_test)
                self.myLog.writeMessage('Training and testing completed!',"DEBUG",functionName)

                # Savign Models
                self.myLog.writeMessage('Saving models ...',"DEBUG",functionName)
                joblib.dump(TraceabilityClassifier, ModelFolderPath + '/' + TraceabilityModelName + '.pkl')
                self.myLog.writeMessage('Models saved : ' + os.path.realpath(ModelFolderPath) + '/' + TraceabilityModelName + '.pkl', "DEBUG",functionName)
                self.myLog.writeMessage('Models successfully saved!',"DEBUG",functionName)

                # Creating an output dataset with test data results
                self.myLog.writeMessage('Preparing output dataset ...',"DEBUG",functionName)
                indices = Traceability_i_test
                df_actual = pd.DataFrame(Traceability_y_test)
                df_actual = df_actual.set_index(indices)
                df_actual.columns = ['Actual Quality']
                df_predict = pd.DataFrame(Traceability_y_pred)
                df_predict = df_predict.set_index(indices)
                df_predict.columns = ['Predicted Quality']

                cols = ['Date','Time','Product Name','AciditySH','Casein','Density','Fat',
                      'Freezing Point mC','Lactose','Protein','SNF','Urea','Remark']

                df_result = dataframe[cols].loc[indices].join([df_actual, df_predict])
                df_result.sort_index(inplace=True)
                df_result['Density'] = [float('%.3f' % val) for val in df_result['Density']]
                df_result['Date'] = pd.to_datetime(df_result['Date'])
                df_result['Date'] = pd.to_datetime(df_result['Date'], format = '%Y-%m-%d').dt.strftime('%d/%m/%Y')

                # Extract RAW and PROCESSED data into two different datasets
                self.myLog.writeMessage('Extract data for RAW and PROCESSED milk ...',"DEBUG",functionName)
                dsRaw = df_result[df_result['Remark']=='Raw']
                dsProcessed = df_result[df_result['Remark']=='Processed']
                self.myLog.writeMessage('Output dataset preparation completed!', "DEBUG",functionName)
                
                # Metrics calculation for RAW and PROCESSED milk
                Raw_y_test = dsRaw['Actual Quality']
                Raw_y_pred = dsRaw['Predicted Quality']

                Processed_y_test = dsProcessed['Actual Quality']
                Processed_y_pred = dsProcessed['Predicted Quality']
                self.myLog.writeMessage('Data Extraction complete!',"DEBUG",functionName)
                
                self.myLog.writeMessage('Executing metrics calculations ...',"DEBUG",functionName)
                Raw_TP, Raw_FP, Raw_TN, Raw_FN, Raw_FPR, Raw_TPR, RawPrecision, RawAccuracy = self.measure(Raw_y_test, Raw_y_pred)
                Processed_TP, Processed_FP, Processed_TN, Processed_FN, Processed_FPR, Processed_TPR, ProcessedPrecision, ProcessedAccuracy = self.measure(Processed_y_test, Processed_y_pred)

                metricsDict = {'RAW_TRUE_POSITIVE_RATE': [Raw_TPR], 'RAW_FALSE_POSITIVE_RATE': [Raw_FPR], 
                      'RAW_PRECISION': [RawPrecision], 'RAW_ACCURACY': [RawAccuracy],
                      'PROCESSED_TRUE_POSITIVE_RATE': [Processed_TPR], 'PROCESSED_FALSE_POSITIVE_RATE': [Processed_FPR], 
                      'PROCESSED_PRECISION': [ProcessedPrecision], 'PROCESSED_ACCURACY': [ProcessedAccuracy]}
                
                dsMetrics = pd.DataFrame(metricsDict)
                self.myLog.writeMessage('Metrics calculations completed!',"DEBUG",functionName)

                # Convert dataset predictions to json using records orientation
                self.myLog.writeMessage('Converting output datasets to JSON ...',"DEBUG",functionName)
                #jsonDataset = df_result.to_json(orient='records')
                jsonDsRaw = dsRaw.to_json(orient='records')
                jsonDsProcessed = dsProcessed.to_json(orient='records')
                jsonMetrics = dsMetrics.to_json(orient='records')
                self.myLog.writeMessage('Conversion completed!',"DEBUG",functionName)  

                # Decode the json data created to insert a custom root element
                self.myLog.writeMessage('Adding roots to JSON ...',"DEBUG",functionName)
                #jsonDataset_decoded = json.loads(jsonDataset)
                #jsonDataset_decoded = {'milkData': jsonDataset_decoded}
                jsonDsRaw_decoded = json.loads(jsonDsRaw)
                jsonDsRaw_decoded = {'rawData': jsonDsRaw_decoded}
                jsonDsProcessed_decoded = json.loads(jsonDsProcessed)
                jsonDsProcessed_decoded = {'processedData': jsonDsProcessed_decoded}
                jsonMetrics_decoded = json.loads(jsonMetrics)
                jsonMetrics_decoded = {'metricsData': jsonMetrics_decoded}
                self.myLog.writeMessage('Roots successfully added!',"DEBUG",functionName)

                # Decode json that contains metrics element and update the prediction json
                self.myLog.writeMessage('Processing JSON output ...',"DEBUG",functionName)
                #jsonDataset_decoded.update(jsonMetrics_decoded)
                jsonDsRaw_decoded.update(jsonDsProcessed_decoded)
                jsonDsRaw_decoded.update(jsonMetrics_decoded)

                #jsonResult = json.dumps(jsonDataset_decoded, indent=4, sort_keys=False)
                jsonResult = json.dumps(jsonDsRaw_decoded, indent=4, sort_keys=False)
                self.myLog.writeMessage('JSON output successfully processed!',"DEBUG",functionName)
                self.myLog.writeMessage('Estimate Milk Quality training and test completed!',"INFO",functionName)
                self.myLog.writeMessage('==============================================================',"INFO",functionName)
                return jsonResult
            except:
                self.myLog.writeMessage('An exception occured!', "ERROR",functionName)
                raise

    def execRFPrediction(self, JsonData):
        # GET FUNCTION NAME
        functionName = sys._getframe().f_code.co_name
        
        with self.myLog.error_debug():
            try:
                self.myLog.writeMessage("Working directory set to: "+os.path.abspath(os.getcwd()),"DEBUG",functionName)
                self.myLog.writeMessage("Module temp directory in use: "+os.path.join(os.path.dirname(__file__)),"DEBUG",functionName)
                
                # LOADING CONFIGURATION FROM PROPERTIES FILE
                self.myLog.writeMessage("Loading configuration from properties file...","DEBUG",functionName)
                
                # LOADING MODELS SETTINGS
                modelsNotExists = False
                ModelFolderPath = self.config.get('PyRandomForest', 'milkquality.randomForest.commonSettings.modelfilePath')
                self.myLog.writeMessage("Model directory set at: "+ModelFolderPath,"DEBUG",functionName)
                PrefixModel = self.config.get('PyRandomForest', 'milkquality.randomForest.commonSettings.modelNamePrefix')
                self.myLog.writeMessage("Model name prefix set at: "+PrefixModel,"DEBUG",functionName)
                TraceabilityModelName = PrefixModel + 'Model'
                
                self.myLog.writeMessage('Preparing to execute Random Forest Predictions of Estimate Milk Quality ...',"INFO",functionName)
                # Estimate Animal Welfare Condition - Prediction
        
        
                
                # Check if models folder exists
                self.myLog.writeMessage('Checking models folder ...',"DEBUG",functionName)
                if (os.path.exists(ModelFolderPath)):
                    pass
                else :
                    self.myLog.writeMessage('Error! Folder '+ModelFolderPath+' not found!',"ERROR",functionName)
                    self.myLog.writeMessage('Sending error message ...',"DEBUG",functionName)
                    errorData = {"Status": "Error",
                                 "Type" : "Models directory not found",
                                 "Description": "The models directory is missing. Please execute the training first."}
                    jsonResult = json.dumps(errorData, indent=4, sort_keys=False)
                    self.myLog.writeMessage('Error sent!',"DEBUG",functionName)
                    return jsonResult
                
                # Check if models file exists
                self.myLog.writeMessage('Checking saved models files ...',"DEBUG",functionName)
                if (os.path.exists(ModelFolderPath + '/' + TraceabilityModelName + '.pkl')):
                    self.myLog.writeMessage('Models files found!',"DEBUG",functionName)
                    pass
                else :
                    # Models not exists
                    self.myLog.writeMessage('Error! Models files not found!',"ERROR",functionName)
                    self.myLog.writeMessage('Listing existing fiels ...',"DEBUG",functionName)
                    filesNames = []
                    modelsNotExists = True
                    existingfiles = [f for f in os.listdir(ModelFolderPath) if os.path.isfile(os.path.join(ModelFolderPath, f))]
                    for i in existingfiles :
                        modelPrefix = i.split('_')[:-1]
                        if not(modelPrefix in filesNames):
                            filesNames.append(modelPrefix)
                    self.myLog.writeMessage('Listing existing files completed!',"DEBUG",functionName)

                if not(modelsNotExists):
                    # Dataset preparation
                    self.myLog.writeMessage('Loading dataset ...',"DEBUG",functionName)
                    JsonObj = json.loads(JsonData)
                    dataframe = pd.DataFrame(JsonObj)

                    cols = ['AciditySH', 'Casein','Density', 'Fat', 'Freezing Point mC', 'Lactose',
                    'Protein', 'SNF', 'Urea']
                    dataset = dataframe[cols]
                    self.myLog.writeMessage('Dataset successfully loaded!',"DEBUG",functionName)

                    # Loading random forest saved models
                    self.myLog.writeMessage('Loading models ...',"DEBUG",functionName)
                    TraceabilityClassifier = joblib.load(ModelFolderPath + '/' + TraceabilityModelName + '.pkl')
                    self.myLog.writeMessage('Models successfully loaded!',"DEBUG",functionName)

                    # Execute predictions
                    self.myLog.writeMessage('Executing predictions ...',"DEBUG",functionName)
                    TraceabilityPred = TraceabilityClassifier.predict(dataset)
                    self.myLog.writeMessage('Predictions successfully completed!',"DEBUG",functionName)
                    
                    # Predictions output preparation
                    self.myLog.writeMessage('Output preparation ...',"DEBUG",functionName)
                    dataframe['Predicted Quality'] = TraceabilityPred
                    dataframe['Density'] = [float('%.3f' % val) for val in dataframe['Density']]
                    
                    # Extracting data for RAW and PROCESSED
                    dsRaw = dataframe[dataframe['Remark']=='Raw']
                    dsProcessed = dataframe[dataframe['Remark']=='Processed']
                    
                    self.myLog.writeMessage('Output preparation completed!',"DEBUG",functionName)
                    
                    # Convert dataset predictions to json using records orientation
                    self.myLog.writeMessage('Converting output dataset to JSON ...',"DEBUG",functionName)
                    #jsonDataset = dataframe.to_json(orient='records')
                    jsonDsRaw = dsRaw.to_json(orient='records')
                    jsonDsProcessed = dsProcessed.to_json(orient='records')
                    self.myLog.writeMessage('Conversion completed!',"DEBUG",functionName)  

                    # Decode the json data created to insert a custom root element
                    self.myLog.writeMessage('Adding roots to JSON ...',"DEBUG",functionName)
                    #jsonDataset_decoded = json.loads(jsonDataset)
                    #jsonDataset_decoded = {'milkData': jsonDataset_decoded}
                    jsonDsRaw_decoded = json.loads(jsonDsRaw)
                    jsonDsRaw_decoded = {'rawData': jsonDsRaw_decoded}
                    jsonDsProcessed_decoded = json.loads(jsonDsProcessed)
                    jsonDsProcessed_decoded = {'processedData': jsonDsProcessed_decoded}
                    self.myLog.writeMessage('Roots successfully added!',"DEBUG",functionName)
                    
                    # Process JSON output
                    self.myLog.writeMessage('Processing JSON output ...',"DEBUG",functionName)
                    jsonDsRaw_decoded.update(jsonDsProcessed_decoded)
                    #jsonResult = json.dumps(jsonDataset_decoded, indent=4, sort_keys=False)
                    jsonResult = json.dumps(jsonDsRaw_decoded, indent=4, sort_keys=False)
                    self.myLog.writeMessage('JSON output successfully processed!',"DEBUG",functionName)
                    self.myLog.writeMessage('Estimate Milk Quality predictions completed!',"INFO",functionName)
                    self.myLog.writeMessage('==============================================================',"INFO",functionName)
                    return jsonResult
                else :
                    if len(existingfiles) < 1 :
                        self.myLog.writeMessage('Error! The folder is empty!',"ERROR",functionName)
                        self.myLog.writeMessage('Sending error message ...',"DEBUG",functionName)
                        errorData = {"Status": "Error",
                                     "Type" : "No models saved",
                                     "Description": "There are no models saved. Please execute the training first."}
                        jsonResult = json.dumps(errorData, indent=4, sort_keys=False)
                        self.myLog.writeMessage('Error sent!',"DEBUG",functionName)
                        return jsonResult
                    else :
                        self.myLog.writeMessage('Error! Model name not found!',"ERROR",functionName)
                        self.myLog.writeMessage('Found '+str(len(filesNames))+' models names:'+ str(filesNames),"ERROR",functionName)
                        self.myLog.writeMessage('Sending error message ...',"DEBUG",functionName)
                        errorData = {"Status": "Error",
                                     "Type" : "Models not found",
                                     "Description": "Models not found for the animal type: "+PrefixModel}
                        jsonResult = json.dumps(errorData, indent=4, sort_keys=False)
                        self.myLog.writeMessage('Error sent!',"DEBUG",functionName)
                        return jsonResult
            except:
                self.myLog.writeMessage('An exception occured!', "ERROR",functionName)
                raise