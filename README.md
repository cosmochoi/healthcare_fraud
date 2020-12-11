# healthcare_fraud
## Authors: Annie Choi (cosmochoi), Marcus Choi (mwc201) , Truong Pham (gottingen411)
### This repo contains files for the capstone project at the NYCDSA bootcamp. In this project, we analyzed health insurance data obtained from Kaggle (https://www.kaggle.com/rohitrox/healthcare-provider-fraud-detection-analysis) and created machine learning models trained on these data to predict fraudulent providers. 

- Folder Fraud_Detection_EDA contains our EDA jupyter notebooks: EDA_Annie (outpatient data and analysis of duplicate claims), EDA_Marcus (beneficiary data and size-related analysis), EDA_Truong (inpatient data and time-related analysis)

- The raw datasets are processed using the following pipeline: 

raw data =====> [preprocess_module.py] =====> full_df (containing all features for each claim) =====> [Feature_Engineering.py] =====>  Features (containing all features for each provider) =====> Machine learning models to predict fraudulent providers (shown in Jupyter notebooks "ML_.* ")

- NetworkFraudAnalysis.ipynb: Further network analysis was done on providers to generate a network graph of providers connected by shared physicians. The output is csv "networkdfnew.csv" which is then added to the Features dataframe in Feature_Engineering.py

- businessscoring.py: function to calculate a business metric used to evaluate machine learning models

- In the machine learning notebooks "ML_.* ", different machine learning models were tuned. The most comprehensive one is "ML_models_FINAL.ipynb". For reference, we kept all the previous ones, which served different purposed as we moved forward in our project. 



 
