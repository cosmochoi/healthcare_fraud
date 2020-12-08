## Routine to clean and change data dtypes for health care fraud datasets
# input: flag telling the function which preprocessing steps to take 
# output: combined data frame containing both inpatient and outpatient datasets, arranged in order of provider ids 

def fraud_preprocessor(i_flag=0):
	
	## Truong's script:
		
	# script to preprocess inpatient data
	# Import ds packages
	import numpy as np
	import pandas as pd 
	import re
	from datetime import date


	# import inpatient datasets
	ip_train_df = pd.read_csv('Train_Inpatientdata-1542865627584.csv',na_values=[''])
	n_trainip = ip_train_df.shape[0]
	ip_test_df = pd.read_csv('Test_Inpatientdata-1542969243754.csv',na_values=[''])


	# combine test and train sets
	ip_df = pd.concat([ip_train_df,ip_test_df], axis=0).reset_index(drop = True)


	# replace NAs in deductible amount with 0
	ip_df.loc[:,'DeductibleAmtPaid'].fillna(0, inplace = True)


	# replace NAs in the remaining columns with 'None'
	ip_df.fillna('None', inplace = True)


	# change data types of the data frame
	ip_df = ip_df.astype(str)
	ip_df.DeductibleAmtPaid=ip_df.DeductibleAmtPaid.astype(float)
	ip_df.InscClaimAmtReimbursed=ip_df.InscClaimAmtReimbursed.astype(float)


	# change date strings to datetime type
	ip_df['AdmissionDt'] = ip_df['AdmissionDt'].map(lambda x: date.fromisoformat(x))
	ip_df['DischargeDt'] = ip_df['DischargeDt'].map(lambda x: date.fromisoformat(x))
	ip_df['ClaimStartDt'] = ip_df['ClaimStartDt'].map(lambda x: date.fromisoformat(x))
	ip_df['ClaimEndDt'] = ip_df['ClaimEndDt'].map(lambda x: date.fromisoformat(x))
		
	# Make a column called month for the month of claim
	ip_df['ClaimMonth'] = ip_df['ClaimStartDt'].map(lambda x: x.month)

	# calculate time duration
	ip_df['HospitalDuration'] = ip_df['DischargeDt'] - ip_df['AdmissionDt']
	ip_df['ClaimDuration'] = ip_df['ClaimEndDt'] - ip_df['ClaimStartDt']

	# convert durations to integers
	ip_df['HospitalDuration'] = ip_df['HospitalDuration'].map(lambda x: x.days)
	ip_df['ClaimDuration'] = ip_df['ClaimDuration'].map(lambda x: x.days)



		
	### Annie's script:
	# Read in original data
	op_test = pd.read_csv("Test_Outpatientdata-1542969243754.csv")
	op_train = pd.read_csv("Train_Outpatientdata-1542865627584.csv")
	n_trainop = op_train.shape[0]
	# Combine test and train set
	op_df = pd.concat((op_train, op_test)).reset_index(drop=True)

	# Imputing NA values with None (should only be None)
	op_df.fillna('None', inplace = True)

	# Convert dates into datetime
	op_df = op_df.astype(str)
	op_df['ClaimStartDt'] = op_df['ClaimStartDt'].map(lambda x: date.fromisoformat(x))
	op_df['ClaimEndDt'] = op_df['ClaimEndDt'].map(lambda x: date.fromisoformat(x))
	op_df['ClaimDuration'] = op_df['ClaimEndDt'] - op_df['ClaimStartDt']
	op_df['ClaimDuration'] = op_df['ClaimDuration'].map(lambda x: x.days)

	# Make a column called month for the month of claim
	op_df['ClaimMonth'] = op_df['ClaimStartDt'].map(lambda x: x.month)

	# Change Dollar amount to float type
	op_df['InscClaimAmtReimbursed'] = op_df['InscClaimAmtReimbursed'].astype(float)
	op_df['DeductibleAmtPaid'] = op_df['DeductibleAmtPaid'].astype(float)


	### Marcus's script:
		
	# import beneficiary datasets
	bene_train_df = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
	n_benetrain = bene_train_df.shape[0]
	bene_test_df = pd.read_csv('Test_Beneficiarydata-1542969243754.csv')

	# combine test and train datasets
	bene_df = pd.concat([bene_train_df, bene_test_df], axis=0)

	# replace NAs in Date of Death (DOD) to 0 (alive) and 1 (dead)
	bene_df['DOD'] = bene_df['DOD'].fillna(0)
	bene_df['DOD'] = bene_df['DOD'].str.contains('2009') == 1
	bene_df['DOD'] = bene_df['DOD'].map({False:0, True:1})

	# change Date of Birth (DOB) to datetime type
	bene_df['DOB'] = pd.to_datetime(bene_df['DOB'], format='%Y/%m/%d')

	# change reimbursement and deductible amounts to float values
	bene_df['IPAnnualReimbursementAmt'] = bene_df['IPAnnualReimbursementAmt'].astype(float)
	bene_df['IPAnnualDeductibleAmt'] = bene_df['IPAnnualDeductibleAmt'].astype(float)
	bene_df['OPAnnualReimbursementAmt'] = bene_df['OPAnnualReimbursementAmt'].astype(float)
	bene_df['OPAnnualDeductibleAmt'] = bene_df['OPAnnualDeductibleAmt'].astype(float)


	# import labels dataset
	label_train_df = pd.read_csv('Train-1542865627584.csv')
	label_test_df = pd.read_csv('Test-1542969243754.csv')

	label_train_df['PotentialFraud'] = label_train_df['PotentialFraud'].map({'No': 0, 'Yes': 1})
	
	# create column indicating patient type
	ip_df['PatientType'] = np.repeat('Inpatient', len(ip_df))
	op_df['PatientType'] = np.repeat('Outpatient', len(op_df))


	if (i_flag==0):	
		### merge all cleaned datasets ###

		# create column indicating patient type
		ip_df['PatientType'] = np.repeat('Inpatient', len(ip_df))
		op_df['PatientType'] = np.repeat('Outpatient', len(op_df))

		#combine all datasets
		ip_op_df = pd.concat((ip_df, op_df), axis=0)
		full_df = pd.merge(ip_op_df, bene_df, on = 'BeneID', how = 'left')

		return full_df
		
	elif (i_flag==1):
		iptrain_df = ip_df.iloc[:n_trainip,:]
		optrain_df = op_df.iloc[:n_trainop,:]
		iptest_df = ip_df.iloc[n_trainip:,:]
		optest_df = op_df.iloc[n_trainop:,:]
		benetrain_df = bene_df.iloc[:n_benetrain,]
		benetest_df = bene_df.iloc[n_benetrain:,]
		return iptrain_df, iptest_df, optrain_df, optest_df, benetrain_df, benetest_df, label_train_df, label_test_df
		


fraud_preprocessor(i_flag=0)









