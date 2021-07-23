This capstone project is the third and last of Udacity's "Machine Learning Engineer with Microsoft Azure" nanodegree program.  
The aim of this project is to compare two models: one based on a given algorithm whose hyperparameters are tuned using HyperDrive, and a second one chosen and optimized using the "Automated ML" (denoted as AutoML from now on) using the same metric ('accuracy' in our case). 


# Predicting survival of patients with heart failure.  

As stated in [BMC Medical Informatics and Decision Making](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec10), "Cardiovascular diseases kill approximately 17 million people globally every year, and they mainly exhibit as myocardial infarctions and heart failures. Heart failure (HF) occurs when the heart cannot pump enough blood to meet the needs of the body."  

The aim of this project is to train models on a dataset of 299 patients and 13 features collected in 2015 that can predict, with the highest accuracy, the survival of patients with heart failures up to the end of the follow-up period.  

The method used to determine the best model is to compare the best-of-classe candidates using a common metric: 'accuracy', i.e the total number of correct predictions over the total number of predictions. We will take the best model identified by the 2 tools available in Azure Machine Learning Studio.  

> Tool 1: Hyperdrive  
> In this scenario, we provide the model we chose ("Logistics Regression") and we ask HyperDrive to determine the set of hyperparameters that maximizes the accuracy.  

> Tool 2: AutoML  
> With AutoML, we let Azure run through a portfolio of models, optimize each one of them and select the model with the highest accuracy. 

Once the best model is identified, we will deploy it and consume the corresponding endpoint to check that our web service is able to accept json requests that contain the values of 13 medical features for one or a group of patient, and send back a prediction of the patients' survival.  
(NB: the model will send back a binary answer, i.e 'will probably survive' or 'will probably die', but it won't send back the probabilities behind those predictions)  

The graph below summarizes the different phases of this project:

![image](https://user-images.githubusercontent.com/36628203/126818290-a8ba04c3-2855-43a3-bd71-2c2e73f41a88.png)




## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

![image](https://user-images.githubusercontent.com/36628203/126629671-cd6cd9e0-b8d3-4be1-97f9-0f3247ca2f6b.png)
Source: [BMC Medical Informatics and Decision Making](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5/tables/1), published: 03 February 2020

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

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
