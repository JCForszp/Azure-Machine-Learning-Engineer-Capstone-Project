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
This is, by the way, confirmed by the AutoML run "Data Guardrails" analysis, below:  
![image](https://user-images.githubusercontent.com/36628203/127145164-acd3f342-e068-495d-9953-782c1d0ee110.png)


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
- **label_column_name**='DEATH_EVENT',  
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

![image](https://user-images.githubusercontent.com/36628203/127148373-2d0ec59d-a3e0-4f54-b81c-f82488e8b2b5.png)

**Results obtained**:  
The AutoML experiment run came out with a Voting Ensemble model, and an accuracy of 0.87633, as shown on the Azure Machine Learning Studio:  
![image](https://user-images.githubusercontent.com/36628203/127149386-23f4b135-e529-4177-a838-d2eba3ba4971.png)  

**Screenshot of the best model from the ML Studio**   

![image](https://user-images.githubusercontent.com/36628203/127149319-128a2859-6d49-47b4-81e9-88c13a8c1fd8.png)


**Parameters of the model**  
![image](https://user-images.githubusercontent.com/36628203/126847135-59d0a876-c578-47e2-b2ff-5982273853b4.png)

**Possible ways to improve it**  
Probably, the most interesting would be to bring neural nets into the game and see how they would do, compared to the Voting Ensemble outcome (0.89).  
It can't be added at this stage, as NN require a GPU-based compute instance to produce the analysis within reasonable time.  

**RunDetails widget**:  
![image](https://user-images.githubusercontent.com/36628203/127156612-d41ac251-8506-47bd-9756-f5d67da299fd.png)
![image](https://user-images.githubusercontent.com/36628203/127156957-5020ad1f-ab53-4173-80ff-bd78718c976a.png)


**Best model trained with its parameters:**  

- From Azure Machine Learning Studio:  
![image](https://user-images.githubusercontent.com/36628203/127148541-73d0a2a7-f042-40df-a6fa-f7328e06b854.png)
![image](https://user-images.githubusercontent.com/36628203/126846771-cad0d3f6-477b-4a19-9fc4-68c546f83295.png)
![image](https://user-images.githubusercontent.com/36628203/127157642-2da046a7-3c7d-4709-b878-9771a4f21944.png)  


- From the AutoML notebook:  
![image](https://user-images.githubusercontent.com/36628203/127158740-de168a6e-e0d7-4355-84f9-6444f21e2130.png)  
(full details in notebook [AutoML.ipynb](https://github.com/JCForszp/Azure-Machine-Learning-Engineer-Capstone-Project/blob/master/Scripts%20%26%20notebooks/automl.ipynb)  
![image](https://user-images.githubusercontent.com/36628203/127158441-17429cb0-bb45-4db7-a9fa-9b6aa6cfd2f1.png)

### Saving & Registering

We finish the AutoML run by saving & registering the best model, which is called 'fitted model' in our case:  
![image](https://user-images.githubusercontent.com/36628203/126879042-d79d2f7b-9ded-41fe-9eeb-05df4ebbae26.png)




## Hyperparameter Tuning  
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search  

For this part of the project, I chose a Logistic Regression model that fits well binary classification problems.  
Features are weakly correlated and the presence of categorical variables (anemia, diabetes, sex) is supported by this algorithm.  
So, LR seems well suited.  

The aim here is to fine-tune the model hyper-parameters using Azure HyperDrive. HyperDrive configuration will be split below into 3 sections:  
1. early termination policy  
2. creation of the estimator and of the different parameters that will be used during the training  
3. the hyper drive configuration run in itself  

We are going to comment the following piece of code:  
![image](https://user-images.githubusercontent.com/36628203/126878639-5f01bc8a-8354-4be8-967a-1299409a1dc6.png)

### 1. early_termination_policy
Regarding early termination of poorly performing runs, I used the BanditPolicy.
The BanditPolicy defines a slack factor (defined here to 0.1).
All runs that fall outside the slack factor with respect to the best performing run will be terminated, saving time and budget.

### 2. estimator and parameters sampling  
**estimator**  
> We will use a Logistic Regression and use 'accuracy'.  
> AuC would also have been an option.  

**Hyperparameter space**  
> I chose the RandomParameterSampling, mainly for speed reason, as the usual alternative, GridParameterSampling, would have triggered  
> an exhaustive search over the complete space, for a gain that proved to be relatively small at the end.  
> Also, GridParameterSampling only allows discrete values, while random sampling is more open as it allows also the use of continuous values.  
> Finally, RandomParameterSampling supports early termination of low-performance runs.  
> For those three reasons, and within the given context of this analysis, RandomParameterSampling appeared as the best option.  
**Note on setting values for the 'Inverse of regularization strength' ("C") and 'max_iter'**  
> I started from the default value of each of those settings, ie. '1' for "C" and '100' for 'max_iter'.    
> C is a continuous variable, so I chose to let Hyperdrive pick float values in a range centered on 1, and with values distributed according to a uniform law.  
> for max_iter, as we are dealing with integers (maximum number of iterations taken for the solvers to converge), I let Hyperdrive pick integer values in a range centered on the default value again (100 +/-50). 

### 3. HyperDrive run configuration  
This configuration object aggregates the settings defined for the policy, the choice of estimator and the hyper-parameters space definition.  
We define here also the primary metric used ('accuracy') that we want to maximize (primary_metric_goal).  
max_total_runs sets a limit of the maximum number of runs that can be created. We set it here to 100.  
max_concurrent_runs is aligned to the compute target available resources available (4).  

### Results  

We can see below the best run metrics that shows the outcome (accuracy) as well as the corresponding hyperparameters:  
![image](https://user-images.githubusercontent.com/36628203/126878735-19f61f63-33cc-4de9-9085-db48ca6d7bba.png)  
We are significantly below the accuracy reached by AutoML, so this is not the model we are going to deploy.  

The RunDetails widget allow to see the accuracy per child run (on the left side, column 'Best Metric'), as well as the related hyperparameters (last two columns on the right).  
![image](https://user-images.githubusercontent.com/36628203/126878827-b7bc46b2-e18a-4dd7-9a8b-56e2c8c3f215.png)

We can obtain a similar level of detail in Azure Machine Learning Studio:  
![image](https://user-images.githubusercontent.com/36628203/126878877-0bad4b13-378e-41df-9dec-8702c84680e2.png)

**Improvements**  
It seems that we reached a "plateau" using a Logistics Regression model.   
When looking at the child runs, 0.833 seems to be the maximum achievable value.  
We can still attempt to replace random sampling by grid search to see if we can 'squeeze' some more accuracy,  
and increase the maximum number of iterations (hyperparameter 'max_iter' to 300 or even 500)
but it seems improbable that we will get an accuracy greater than the one determined by AutoML.  

### Saving & Registering
The code below registers the model optimized by HyperDrive, even if it's clear that's not going to be the one we will deploy in the next section:  
![image](https://user-images.githubusercontent.com/36628203/126879168-1d6fac8d-7364-425b-8a65-253e27d1fbc6.png)




## Model Deployment

### Overview of the deployed model
Below the screenshot that shows the deploymment of the AutoML model and its deployment status ('healthy'):
![image](https://user-images.githubusercontent.com/36628203/126879339-f7c8abe3-3582-4371-b9da-d8fc928ea13c.png)
![image](https://user-images.githubusercontent.com/36628203/126879402-dea754d9-981a-4443-abe8-59d00e4d5269.png)

# Instructions on how to query the endpoint
I chose not to create a request manually, but to send a batch of 10 requests picked at random from the dataset.
Generating such request could be done in a single line:  
![image](https://user-images.githubusercontent.com/36628203/127027798-ac08ac4e-cc59-47de-a557-7a39478edbc2.png)
The first instruction, `df.sample(10)`, picks 10 records from the pandas dataframe we created earlier.  
Please note that, at this stage, the target column is already removed, so we have records ready to be inputted to the webservice.  
We convert this selection of 10 records into a dictionary format using the command `.to_dict(orient='records')`.   
Finally, we convert this dictionary into the json format (`input_data`) we need to post our request:   
![image](https://user-images.githubusercontent.com/36628203/127028468-e1543519-9f53-40df-8dfb-1c27dd8e08ee.png)  

The scoring_uri is directly provided by Azure when the web service is activated, using the command `webservice.scoring_uri`.  
The screenshot above shows that we receive a list of 10 answers, 0 ('lived') or 1('died'), one per record.   

This method of querying the endpoint can be easily adapted to generate a request from any pandas dataframe.  
When replacing the `.sample()` command by `.loc[]` or `.iloc[]`, any record or group of record from a given dataset can be used to generate a request. 


## Screen Recording

You will find below 3 videos that demo each of the 3 parts of this capstone project, including the demo of the working models and the demo of a sample request sent to the endpoint and its response:  

01. [HyperDrive Experiment demo](https://youtu.be/BntDQjLNIME)
02. [AutoML Experiment demo](https://youtu.be/p_qk5BFxTaI)
03. [Deployment & demo of the webservice](https://youtu.be/JJqE7ZWk65Q)

## Standout Suggestions
As mentioned in the section related to the query of the endpoint, I chose not to generate a manual request, but to build one from an existing Dataframe.  
In a production environment, the json payload will either be built dynamically by a routine, or, most probably - to keep track of the requests sent and the answers received, from pre-existing records, stored in an already existing data structure.  

I succeeded to generate a request dynamically.  Now, I will keep looking at the response.  
As shown on the screenshot (below for convenience), the webservice sends back a list of answers in a dict (unique key "result"), but the whole answer is sent back as a string (see the quotes below).   
![image](https://user-images.githubusercontent.com/36628203/127030403-e235c509-66b2-445c-9d7e-25cd57c89491.png)

So, the next piece of code will be to update the source dataframe with the answer received from the webservice.  
That will give us the ability to match predicted vs actual survivals, and determine if we have **data drift** and assess if the model performance remains reasonable or needs to be retrained.

