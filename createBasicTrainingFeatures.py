traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

# Note : Configure values as need :: keyword for search - Configuration
# @Param : minLongCandleLength_config - this will vary for different stocks as per market flow history - for crude it can be 5 Rupees, for Adani it can be 1 Rupee
# @Param : relative minimum for open and close values to be considered for doji features relativeOpenCloseValuePercent_config (1% for shares like Adani(1-2 Rs) and 0.01% for crude(.5 to 1 Rs))
def createInputData(dataName,dataFrequency, outputFileName='preparedTrainingData.csv', minLongCandleLength_config=1, relativeOpenCloseValuePercent_config=1):    
    
    import os,sys,traceback    
    from datetime import datetime, timedelta

    import pandas as pd  

    from dataPreprocessing.basicRawDataProcess import getAutoConfigData
    from dataPreprocessing.basicRawDataProcess import setAutoConfigData


    # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
    # using various python commands like os.path.abspath and os.path.join
    jupyterNodePath = None

    configFilePath = None

    # @Return Type
    success = False # help mark successful execution of function 
    outputFilePath = None

    # holds data from input data file - Truth source, should be usd only for reference and no updates should happen to this variable
    inputRawProcessedDataDF = None
    # Variable to hold a dataframe created with the data from input data files 
    # Will be used for data preparation
    preparedTrainingDataDF = None

    # declaring other variables necessary for further manipulations
    bufferDF = None;  #for interim data frame manipulations
    
    try:
        #caluclate the deployment directory path of the current juypter node in the operating system
        jupyterNodePath = getJupyterRootDirectory()
        
        configFilePath=jupyterNodePath+'\\src\config\\autoConfig\\config.json'

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

    finally:
        return success, outputFilePath, outputFileName,preparedTrainingDataDF