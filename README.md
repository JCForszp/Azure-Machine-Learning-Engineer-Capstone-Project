This capstone project is the third and last of Udacity's "Machine Learning Engineer with Microsoft Azure" nanodegree program.  
The aim of this project is to compare two models: one based on a given algorithm whose hyperparameters are tuned using HyperDrive, and a second one chosen and optimized using the "Automated ML" (denoted as AutoML from now on) using the same metric ('accuracy' in our case). 


# Predicting survival of patients with heart failure.  


## Project Overview  
As stated in [BMC Medical Informatics and Decision Making](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec10), "Cardiovascular diseases kill approximately 17 million people globally every year, and they mainly exhibit as myocardial infarctions and heart failures. Heart failure (HF) occurs when the heart cannot pump enough blood to meet the needs of the body."  

The aim of this project is to train models on a dataset of 299 patients and 13 features (including the target one, i.e the "death event") collected in 2015 that can predict, with the highest accuracy, the survival of patients with heart failures up to the end of the follow-up period.  

The method used to determine the best model is to compare the best-of-classe candidates using a common metric: 'accuracy', i.e the total number of correct predictions over the total number of predictions. We will take the best model identified by the 2 tools available in Azure Machine Learning Studio.  

> Tool 1: Hyperdrive  
> In this scenario, we provide the model we chose ("Logistics Regression") and we ask HyperDrive to determine the set of hyperparameters that maximizes the accuracy.  

> Tool 2: AutoML  
> With AutoML, we let Azure run through a portfolio of models, optimize each one of them and select the model with the highest accuracy. 

Once the best model is identified, we will deploy it and consume the corresponding endpoint to check that our web service is able to accept json requests that contain the values of 12 medical features for one or a group of patient, and send back a prediction of the patients' survival.  
(NB: the model will send back a binary answer, i.e 'will probably survive' or 'will probably die', but it won't send back the probabilities behind those predictions)  

The graph below summarizes the different phases of this project:

![image](https://user-images.githubusercontent.com/36628203/126818290-a8ba04c3-2855-43a3-bd71-2c2e73f41a88.png)




## Project Set Up and Installation
For this project, we will use a CPU-based Compute Instance (Neural Nets are scoped out) configured using Azure Standard_DS3_v2 general purpose profile (4 cores, 14GB RAM, 28GB storage), meant for classifcal ML model training, AutoML runs and pipeline runs.  

The notebooks and python script necessary to re-run the 2 steps above (HyperDrive & AutoML) have been stored in the folder ["Scripts & notebooks"](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/tree/master/Scripts%20%26%20notebooks). Each step has been handled as separate experiments. The first run corresponds to the HyperDrive run. It requires the notebook [hyperparameter_tuning.ipynb](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/hyperparameter_tuning.ipynb) as well as the python script [Train.py](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/train.py).  
The second experiment (AutoML), only requires the [AutoML.ipynb](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/automl.ipynb) notebook.   


## Dataset

### Overview
I will use Kaggle "Heart Failure Prediction dataset".  
This dataset is related to a study that focused on survival analysis of 299 heart failure patients who were admitted to Institute  
of Cardiology and Allied hospital Faisalabad-Pakistan during April-December (2015).  
All the patients were aged 40 years or above, having left ventricular systolic dysfunction.  

The dataset contains the following 12 clinical features, plus one target feature ("death event"):  
A data analysis report is available onmy github repo, here  
  
### Clinical features:  

- **age**: age of the patient (years)  
- **anaemia**: decrease of red blood cells or hemoglobin (boolean)  
- **high blood pressure**: if the patient has hypertension (boolean)  
- **creatinine phosphokinase (CPK)**: level of the CPK enzyme in the blood (mcg/L)  
- **diabetes**: if the patient has diabetes (boolean)  
- **ejection fraction**: percentage of blood leaving the heart at each contraction (percentage)  
- **platelets**: platelets in the blood (kiloplatelets/mL)  
- **sex**: woman or man (binary)  
- **serum creatinine**: level of serum creatinine in the blood (mg/dL)  
- **serum sodium**: level of serum sodium in the blood (mEq/L)  
- **smoking**: if the patient smokes or not (boolean)  
- **time**: follow-up period (days)  

**Target feature**  
- **death event**: if the patient deceased during the follow-up period (boolean)  

The table below gives the unit of measure, for each (non-categorical) feature, as well as the range of values observed. 
![image](https://user-images.githubusercontent.com/36628203/126629671-cd6cd9e0-b8d3-4be1-97f9-0f3247ca2f6b.png)
Source: [BMC Medical Informatics and Decision Making](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5/tables/1), published: 03 February 2020

A correlation matrix shows only weak correlations between features and no missing data.    
![image](https://user-images.githubusercontent.com/36628203/126821310-83152bb8-12c3-4010-bc18-24f442237f22.png)

So, the Kaggle / BMC dataset has a good overall quality and is suitable for our analysis.  

### Importing the Kaggle dataset into Azure ML studio  
The BMC [dataset](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Datasets/heart_failure_clinical_records_dataset.csv) whas unpacked and saved in a dedicated [folder](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/tree/master/Datasets).  
I accessed the raw version directly from the web-link and imported it into my Azure workspace using command `Dataset.Tabular.from_delimited_files(example_data)`, where the variable "example_data" contained the http: link to the dataset on my Github repo.  

For the first experiment (HyperDrive), only the command in the python script is actually needed.  
I used the `Dataset.Tabular.from_delimited_files()` command to benefit from Azure data wrangling facilities. The dataset is then immediately converted into a pandas dataframe format to be fed to the Sk-learn Train-Test-split function and model. 
The notebook of the first experiment also contains a dataset import section. This was only done to be used as template for the automl.ipynb notebook, but is not necessary stricto-sensu and can be overlooked.


### Tasks
The target feature is a binary variable (life or death).  
We are dealing here with a classification task, i.e trying to predict the outcome of the follow-up period, based on the given clinical features.
As mentioned in the project-overview, the aim of this project is:   
> 1. to train models on the same BMC dataset, using separate HyperDrive and AutoML experiments,   
> 2. keep, from the both experiments, the model that reached the highest accuracy,  
> 3. deploy it and, last,  
> 4. feed the endpoint with a new json request to obtain the prediction(s). 



### Access
The source file is directly accessed from this repo, processed by the Dataset Tabular.from_delimited_files() method as we are dealing with a plain csv file.   
![image](https://user-images.githubusercontent.com/36628203/126844923-a602f809-9f49-4006-8427-feca9cde6fcc.png)  

The dataset is immediately registered, in order to be made available to the AutoML experiment.  
Below the details of the registered dataset:  
![image](https://user-images.githubusercontent.com/36628203/126833631-99321462-4455-44c3-b2f7-41fd7873070b.png)  


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
This complete overview of the AutoML settings is taken from the [AutoML Jupyter notebook](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/automl.ipynb).  
The settings used are listed below, with a short justification:  

![image](https://user-images.githubusercontent.com/36628203/126845421-9686ecc7-8a82-41ba-8032-af8dcbd5b79f.png)  
### Note on automl settings selection:  ¶
- **n_cross_validations**: 4, 10 is a usual value for cross-validations, but the size of the dataset is relatively small.  
Hence, a 90/10% split seems a bit disproportionate. I prefer to take a 75/25% split,  
which will end of with testing sets of 75 patients, so probably more reasonable and keeping the  
number of runs low.  
- **primary_metric**: 'accuracy',  
'accuracy' is the most frequent and easiest metrics to use for classification tasks  
- **enable_early_stopping**: True,  
According to Azure documentation, this settings allows automl to terminate a score determination if the score is not improving.  
The default value is 'False' and hence, needs to be set to 'True' at config level.  
Microsoft documentation mentions that Early stopping window starts on the 21st iteration  
and looks for early_stopping_n_iters iterations (currently set to 10).  
This means that the first iteration where stopping can occur is the 31st.  
Hence, this setting is a nice to have, but won't be critical for our limited exercice.  
- **max_concurrent_iterations**: 4,  
According to Microsoft documentation Represents the maximum number of iterations that would be executed in parallel.  
The default value is 1.  
In our compute_config, we chose a value of 4, and the number of concurrent values needs to be less or equal to that number.  
Hence, the value of this setting.  
- **experiment_timeout_minutes**: 20,  
Defines how long, in minutes, the experiment should continue to run. Looking at Azure documentation on how-to-configure-auto-train,  
20mn seemed to be a reasonable trade-off.  
- **verbosity**: logging.INFO  
The verbosity level for writing to the log file. The default is INFO or 20. So, we could basically have skipped this setting, but it seems  
good practice to specify it every time, to assess if it's really the optimal level of detail.  


![image](https://user-images.githubusercontent.com/36628203/126845473-f7e2c338-8c21-4638-ba4d-0422d781d963.png)

### Note on automl_config settings:
- **compute_target** = compute_target,  
This is the Azure Machine Learning compute target to run the Automated Machine Learning experiment on.  
It corresponds to the compute_target we defined above in the script, right after the import of the dependencies.  
- **task**='classification',  
Three types of tasks are allowed here:'classification', 'regression', or 'forecasting'.  
As mentioned in the Dataset section, we are clearly here in a classification task.  
- **training_data**=dataset,  
This is the dataset we registered in previous cell.  
- **label_column_name**='0DEATH_EVENT',  
This is the name of the target column, i.e the column we will are training our model to predict.    
The original dataset on Kaggle clearly defines 'DEATH_EVENT' as being the label column.  
- **path** = project_folder,  
We set this project_folder to './capstone-project'  
- **featurization**= 'auto',  
Two values allowed: 'auto' and 'off'. Based on Microsoft doc, setting featurization to off would mean re-doing manually  
all one-hot encoding, managing missing values,... It meakes total sense to leave automl dealing with that on a pre-cleaned dataset.  
- **debug_log** = "automl_errors.log",  
The log file to write debug information to. If not specified, 'automl.log' is used.  
I just set the name to one I chose.  
- **enable_onnx_compatible_models**=False,  
ONNX is presented as a way to optimize the inference of the ML model.(doc)  
We are dealing with a small-sized dataset, so I chose to leave this setting of False and I will investigate this feature separately later.  
- **automl_settings**  
Brings the automl_settings dictionary we defined above in the automl_config object.  

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
**Results obtained**:
The AutoML experiment run came out with a Voting Ensemble model, and an accuracy of 0.88972, as shown on the Azure Machine Learning Studio:  
![image](https://user-images.githubusercontent.com/36628203/126847833-48f120b2-b072-42c6-9397-a5b3dc3c4c47.png)

**Parameters of the model**
![image](https://user-images.githubusercontent.com/36628203/126847135-59d0a876-c578-47e2-b2ff-5982273853b4.png)

**Possible ways to improve it**  
Probably, the most interesting would be to bring neural nets into the game and see how they would do, compared to the Voting Ensemble outcome (0.89).  
It can't be added at this stage, as NN require a GPU-based compute instance to produce the analysis within reasonable time. 

**RunDetails widget**:
![image](https://user-images.githubusercontent.com/36628203/126846146-2806dd50-04a9-4134-89e2-b5bcbe616bd9.png)

**Best model trained with its parameters:**

- From Azure Machine Learning Studio:
![image](https://user-images.githubusercontent.com/36628203/126846771-cad0d3f6-477b-4a19-9fc4-68c546f83295.png)
![image](https://user-images.githubusercontent.com/36628203/126846369-5bcb4aaf-c805-4d0b-9855-c8dc5dff2986.png)

- From the AutoML notebook:
![image](https://user-images.githubusercontent.com/36628203/126846935-8da862bb-992b-46a0-9d81-a7fa5a9470d0.png)
(full details in notebook [AutoML.ipynb](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/automl.ipynb)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.


## Udacity mentor helpers  

### Step 2: Model Training and Deployment  

**HyperDrive**  
- I have finished all the TODOS in the hyperparameter_tuning.ipynb file  
- I have used one type of sampling method: grid sampling, random sampling or Bayesian sampling  
- Specify an early termination policy (not required in case of Bayesian sampling)  
- I have tuned at least 2 hyperparameters in my model  
- I have visualized the progress and performance of my hyperparameter runs (Logs metrics during the training process)  
- I have saved the best model from all the hyperparameter runs  
- Take a screenshot of the RunDetails widget that shows the progress of the training runs of the different experiments  
- Take a screenshot of the best model with its run id and the different hyperparameters that were tuned  

**AutoML**  
- I have finished all lhe TODOs in the automl.ipynb file  
- Take a screenshot of the RunDetails widget that shows the progress of the training runs of the different experiments  
- Take a screenshot of the best model with its run id  

**Deployment of the best model**  
- I have deployed 
 as a webservice
- I have tested the webservice by sending a request to the model endpoint
- I have deleted webservice and shut down all the computes that I have used
- Take a screenshot showing the model endpoint as active


### Step 3: Complete the README  
- I have completed all of the required TODOs in the README file


### Step 5: Create a screencast  
- A working model
- Demo of the deployed model
- Demo of a sample request sent to the endpoint and its response 
- Demo of any additional feature of your model 
- Add the screencast link to the README file
