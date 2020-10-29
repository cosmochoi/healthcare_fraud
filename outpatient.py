import pandas as pd
import matplotlib.pyplot as plt

outpatient_test = pd.read_csv("./dataset/Test_Outpatientdata-1542969243754.csv")
outpatient_train = pd.read_csv("./dataset/Train_Outpatientdata-1542865627584.csv")

outpatient = pd.concat((outpatient_train, outpatient_test)).reset_index(drop=True)

#IMPUTATION
#Looking at Percentage NA for each column
nullnum = {}
percentage = {}
for element in outpatient.columns:
    nullnum[element] = outpatient[element].isna().sum()
    percentage[element] = outpatient[element].isna().sum()/outpatient[[element]].shape[0] * 100

pd.DataFrame(sorted(percentage.items(), key=lambda x: x[1], reverse=True), columns = ['Feature', '% NA']).head(30)

#Impute all NA with None
naColumns = outpatient.columns[outpatient.isna().any()].tolist()
for column in naColumns:
    outpatient[column] = outpatient[column].fillna('None')

#Convert dates into datetime 
from datetime import datetime
outpatient['ClaimStartDt'] = [datetime.strptime(date, '%Y-%m-%d') for date in outpatient.ClaimStartDt]
outpatient['ClaimEndDt'] = [datetime.strptime(date, '%Y-%m-%d') for date in outpatient.ClaimEndDt]
outpatient['duration'] = outpatient['ClaimEndDt'] - outpatient['ClaimStartDt']
outpatient['duration'] = list(map(lambda x: x.days, outpatient['duration']))

# Make a column called month for the month of claim
outpatient['month'] = list(map(lambda x: x.month, outpatient['ClaimStartDt']))

# Change Dollar amount to float type
outpatient['InscClaimAmtReimbursed'] = outpatient['InscClaimAmtReimbursed'].astype(float)
outpatient['DeductibleAmtPaid'] = outpatient['DeductibleAmtPaid'].astype(float)

print(outpatient.dtypes)
print(outpatient.head())
