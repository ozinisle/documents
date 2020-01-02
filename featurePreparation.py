import os, errno, traceback, sys

from pathlib import Path

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def doBasicOperation(dataName,dataFrequency,autoConfigFileRelativePath,KEY_preProcessedDataFilePath,variation_degree):
    import os,sys,traceback    
    from datetime import datetime, timedelta

    import pandas as pd  
    import numpy as np

    
    from dataPreprocessing.basicRawDataProcess import getAutoConfigData
    from dataPreprocessing.basicRawDataProcess import setAutoConfigData

    
    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    print ("into method doBasicOperation")

    try:

        # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
        # using various python commands like os.path.abspath and os.path.join
        jupyterNodePath = None

        configFilePath = None    

        # holds data from input data file - Truth source, should be usd only for reference and no updates should happen to this variable
        inputRawProcessedDataDF = None    

        #caluclate the deployment directory path of the current juypter node in the operating system
        jupyterNodePath = getJupyterRootDirectory()
        print("jupyterNodePath >>> "+jupyterNodePath)

        configFilePath=jupyterNodePath+autoConfigFileRelativePath
        print("configFilePath >>> "+configFilePath)

        autoConfigData = getAutoConfigData(configFilePath)
        

        preProcessedDataFilePath=autoConfigData[dataName][dataFrequency][KEY_preProcessedDataFilePath]

        # read the raw processed data from csv file
        inputRawProcessedDataDF = pd.read_csv(preProcessedDataFilePath)  

        basicDf = createFundamentalFeatures(inputRawProcessedDataDF)
        
        print("before return statement of method doBasicOperation ")

        return basicDf,variation_degree,preProcessedDataFilePath,autoConfigData,configFilePath

    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise
    

def doFeatureAssessment(newFeatureDf,basicDf,variation_degree,preProcessedDataFilePath,autoConfigData,
    configFilePath,requiredMinimumCorrelation,featureIndexStamp,dataName,dataFrequency):
    import numpy as np
    import pandas as pd

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    from utilities.fileFolderManipulations import deleteFile
    
    from dataPreprocessing.basicRawDataProcess import setAutoConfigData
    
    try:

        featureOfInterest = newFeatureDf.name
        if variation_degree==0:
            newTrainingSetDf=pd.concat([basicDf,newFeatureDf],axis=1)
        else:
            newTrainingSetDf = createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) 

        correlation = newTrainingSetDf.corr()

        reasonableCorelation = correlation.loc[ (np.abs(correlation['open'])>requiredMinimumCorrelation) & 
        (np.abs(correlation['high'])>requiredMinimumCorrelation) &
        (np.abs(correlation['low'])>requiredMinimumCorrelation) & 
        (np.abs(correlation['close'])>requiredMinimumCorrelation)]

        # create necessary file folder structure for storing and filtering features
        preprocessedFolderPath = getParentFolder(preProcessedDataFilePath)
        outputFolderPath = getParentFolder(preprocessedFolderPath)

        featuresFolder = outputFolderPath+"\\features"
        createFolder(featuresFolder)

        rawFeaturesFolder = featuresFolder+"\\rawFeatures"
        createFolder(rawFeaturesFolder)

        filteredFeaturesFolder = featuresFolder+"\\filteredFeatures"
        createFolder(filteredFeaturesFolder)

        correlationsFolder = featuresFolder+"\\correlations"
        createFolder(correlationsFolder)

        reasonableCorrelationsFolder = correlationsFolder+"\\reasonableCorrelations"
        createFolder(reasonableCorrelationsFolder)

        trainableFeaturesListtFilePath = filteredFeaturesFolder+"\\"+featureIndexStamp+featureOfInterest+"_trainableFeaturesList.csv"
        currentFeatureListFilePath = rawFeaturesFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_list.csv"
        currentFeatureCorrelationListFilePath = correlationsFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_correlation_list.csv"
        reasonableCorelationListFilePath = reasonableCorrelationsFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_reasonable_correlation_list.csv"

        deleteFile(trainableFeaturesListtFilePath)
        deleteFile(currentFeatureListFilePath)
        deleteFile(currentFeatureCorrelationListFilePath)
        deleteFile(reasonableCorelationListFilePath)

        # store output information related to current 
        newTrainingSetDf.to_csv(currentFeatureListFilePath, sep=',', index=False)
        correlation.to_csv(currentFeatureCorrelationListFilePath, sep=',', index=True)
        reasonableCorelation.to_csv(reasonableCorelationListFilePath, sep=',', index=True)

        if len(reasonableCorelation.index)>4:    
            # store trainable features in global file - to be used by other training feature creation procedures    
            newFilteredTrainableFeaturesDf = newTrainingSetDf[[filteredIndex for filteredIndex in reasonableCorelation.index] ]
            trainableFeaturesDf=newFilteredTrainableFeaturesDf.drop(columns=["open","close","high","low"])    
            # trainableFeaturesDf = getTrainableFeaturesListDf(trainableFeaturesListtFilePath)
            # if trainableFeaturesDf is None:
            #     trainableFeaturesDf= newFilteredTrainableFeaturesDf
            # else:        
            #     # newFilteredTrainableFeaturesDf=newFilteredTrainableFeaturesDf.drop(columns=["open","close","high","low"])    
            #     # trainableFeaturesDf = pd.concat([trainableFeaturesDf,newFilteredTrainableFeaturesDf],axis=1)
            #     for index in reasonableCorelation:
            #         try:
            #             trainableFeaturesDf[index] = newFilteredTrainableFeaturesDf[index]
            #         except KeyError:
            #             print ('key error >>>' + index)
            
            if not trainableFeaturesDf is None or trainableFeaturesDf.shape[1]>0:
                trainableFeaturesDf.to_csv(trainableFeaturesListtFilePath, sep=',', index=False)

            # assertions
            print("newTrainingSetDf shape>>>"+str(newTrainingSetDf.shape[0])+","+str(newTrainingSetDf.shape[1]))
            print("trainableFeaturesDf shape>>>"+str(trainableFeaturesDf.shape[0])+","+str(trainableFeaturesDf.shape[1]))
            
            autoConfigData[dataName][dataFrequency].update({'trainableFeaturesListtFile':trainableFeaturesListtFilePath})
            setAutoConfigData(configFilePath,autoConfigData)
        else:
            trainableFeaturesDf = getTrainableFeaturesListDf(trainableFeaturesListtFilePath)

        return correlation, reasonableCorelation ,newTrainingSetDf ,trainableFeaturesDf
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise

def getTrainableFeaturesListDf(filePath):
    import pandas as pd
    df = None
    try:
        df=pd.read_csv(filePath) 
    except FileNotFoundError:
        df=None
        
    return df
    
def createFundamentalFeatures(rawDf):    
    import pandas as pd

    #initialize the straight forward input features
    df = pd.DataFrame({            
        'open':rawDf['open'],
        'high':rawDf['high'],
        'low':rawDf['low'],
        'close':rawDf['close']    
    })

    print("added INPUT FEATURES >>> 4 count >>> open-high-low-close")    
    
    return df

def getFeatureVariations(row, featureName, variation_degree_count):
    import numpy as np

    try:
    
        featureVal=row[featureName]    
        
        for iterator in range(1,variation_degree_count+1):
            row[featureName+'_exp_1'] = np.exp(featureVal)
            row[featureName+'_exp_inv_1'] = np.exp(-1*featureVal)
        
            if iterator>1:
                val= np.power(featureVal,iterator)
                valInv = 0
                if not val==0:
                    valInv = 1/val
                    row[featureName+'_times_inv_'+str(iterator)] = 1/(val*iterator)
                # correlation of X::mY does not change for the value m and hence commenting out the following code
                # else:
                #     row[featureName+'_times_inv_'+str(iterator)] = 0

                row[featureName+'_pow_'+str(iterator)] = val
                row[featureName+'_pow_inv_'+str(iterator)] = valInv
                row[featureName+'_exp_'+str(iterator)] = np.exp(iterator*featureVal)
                row[featureName+'_exp_inv_'+str(iterator)] = np.exp(-iterator*featureVal)

                # correlation of X::mY does not change for the value m and hence commenting out the following code
                # row[featureName+'_times_'+str(iterator)] = val*iterator        

                if val>0:
                    row[featureName+'_log_times_'+str(iterator)] = np.log(val*iterator)
                elif val<0:
                    row[featureName+'_log_times_'+str(iterator)] = -np.log(-1*val*iterator)
            
        return row 
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise

def createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) :
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    try:
        # Create and register a new `tqdm` instance with `pandas`
        # (can use tqdm_gui, optional kwargs, etc.)
        tqdm.pandas()



        newTrainingSetDf = pd.concat([basicDf,newFeatureDf],axis=1)

        newTrainingSetDf = newTrainingSetDf.progress_apply(lambda row,
                        featureName, variation_degree_count:getFeatureVariations(row,featureOfInterest,variation_degree),axis=1,
                        args=[featureOfInterest,variation_degree])

        return newTrainingSetDf
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise

