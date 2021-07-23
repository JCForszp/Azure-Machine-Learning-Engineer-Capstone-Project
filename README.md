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

The notebooks and python script necessary to re-run the 2 steps above (HyperDrive & AutoML) have been stored in the folder ["Scripts & notebooks"](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/tree/master/Scripts%20%26%20notebooks). Please note that the notebook [hyperparameter_tuning.ipynb](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/hyperparameter_tuning.ipynb) requires the python script [Train.py](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/train.py)


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


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.  
We are dealing here with a classification task, i.e trying to predict the outcome of the follow-up period based on the given clinical features.



### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

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
