### main script to call preprocessing routines and run ML models 

# import preprocessing routine
from preprocess_module import fraud_preprocessor




def feature_engineering(j_flag=0):
	import pandas as pd
	import numpy as np

	if(j_flag==0): #All feature engineering for the full dataframe
		full_df = fraud_preprocessor(i_flag=0)

	elif(j_flag==1):
		iptrain_df, iptest_df, optrain_df, optest_df, benetrain_df, benetest_df, label_train_df, label_test_df = fraud_preprocessor(i_flag=1)
		ip_op_train = pd.concat((iptrain_df, optrain_df), axis=0)
		trainset = pd.merge(ip_op_train, benetrain_df, on = 'BeneID', how = 'left')
		trainset = pd.merge(trainset, label_train_df, how = 'inner', on = 'Provider')
		full_df = trainset

	else:
		iptrain_df, iptest_df, optrain_df, optest_df, benetrain_df, benetest_df, label_train_df, label_test_df = fraud_preprocessor(i_flag=1)
		ip_op_test = pd.concat((iptest_df, optest_df), axis=0)
		testset = pd.merge(ip_op_test, benetest_df, on = 'BeneID', how = 'left')
		full_df = testset

	# create physician count column 
	full_df.AttendingPhysician = full_df.AttendingPhysician.replace('None',np.nan)
	phys_count = full_df.groupby(['Provider']).AttendingPhysician.nunique(dropna=True).reset_index(name='Phys_Count')

	# create patient count column 
	patient_count = full_df.groupby(['Provider']).BeneID.nunique().reset_index(name='Patient_Count')
	train_features1 = pd.merge(phys_count, patient_count, on='Provider')
	
	# create normalized patient count 
	train_features1['Norm_Patient_Count'] = round(train_features1['Patient_Count']/train_features1['Phys_Count'],2)

	# create claim count column 
	claim_count = full_df.groupby(['Provider']).ClaimID.count().reset_index(name='Claim_Count')
	train_features2 = pd.merge(train_features1, claim_count, on='Provider')
	
	# create normalized claim count
	train_features2['Norm_Claim_Count'] = round(train_features2['Claim_Count']/train_features2['Phys_Count'],2)
	
	
	##create service type column
	prov_full = full_df.groupby(['Provider', 'PatientType'])['ClaimID'].count().reset_index()
	prov_type = full_df.groupby(['Provider', 'PatientType'])['ClaimID'].count().reset_index(name='').drop('', axis=1)

	# create a dictionary provider by service type
	lst_prov_type = list(zip(prov_type['Provider'], prov_type['PatientType']))

	# feed in empty dict with values inpatient, outpatient, or both
	prov_type_dict = {}

	for i in lst_prov_type:
	    if i[0] not in prov_type_dict:
	        prov_type_dict[i[0]]= i[1]    
	    else:
	        prov_type_dict[i[0]] = 'Both_Service'
	        

	# creaete new column type of service by povider
	service_type = pd.DataFrame(prov_type_dict.keys(), prov_type_dict.values()).reset_index().\
	rename(columns={'index':'Service_Type', 0:'Provider'})
	
	# add Service column 
	train_features3 = pd.merge(train_features2, service_type, on='Provider')

	#create dummy for service types
	dummy = pd.get_dummies(service_type['Service_Type'])
	dummy_type = pd.concat([service_type, dummy], axis=1)
	dummy_type = dummy_type.drop('Service_Type', axis=1)
	
	# add Service column 
	train_features3 = pd.merge(train_features3, dummy_type, on='Provider')

	
	# add Service counts
	ip_count = np.zeros(train_features3.shape[0])
	op_count = np.zeros(train_features3.shape[0])
	provider = train_features3.Provider.unique()
	service_count_df = pd.DataFrame({'Provider':provider,'Inpatient_Count': ip_count, 'Outpatient_Count': op_count})
	for ind, provider in enumerate(service_count_df.Provider):
    		if prov_full.loc[(prov_full['Provider']==provider) & (prov_full['PatientType']=='Inpatient'),'ClaimID'].any():
        		service_count_df.loc[ind,'Inpatient_Count'] = prov_full.loc[(prov_full['Provider']==provider) & 									(prov_full['PatientType']=='Inpatient'),'ClaimID'].values[0]    
    		if prov_full.loc[(prov_full['Provider']==provider) & (prov_full['PatientType']=='Outpatient'),'ClaimID'].any():
        		service_count_df.loc[ind,'Outpatient_Count'] = prov_full.loc[(prov_full['Provider']==provider) & 										(prov_full['PatientType']=='Outpatient'),'ClaimID'].values[0]

	train_features3 = pd.merge(train_features3, service_count_df, on='Provider')
	# add Normalized Service Counts
	train_features3['Norm_Inpatient_Count'] = round(train_features3['Inpatient_Count']/train_features3['Claim_Count'],2)
	train_features3['Norm_Outpatient_Count'] = round(train_features3['Outpatient_Count']/train_features3['Claim_Count'],2)
	
	
	
	# Duplicate Claims 
	full_df2 = full_df.copy()
	full_df2['all_duplicates'] = full_df2.duplicated(subset = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
		'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
		'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
		'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
		'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
		'ClmProcedureCode_5', 'ClmProcedureCode_6', 'ClmAdmitDiagnosisCode', 'BeneID', 'Provider'], keep=False)
	#duplicate claims
	Duplicates = full_df2[full_df2.all_duplicates == True].groupby('Provider')['BeneID'].count().reset_index(name='DuplicateClaims')
	fillvalues = {'DuplicateClaims': 0} 
	train_features4 = pd.merge(train_features3, Duplicates, on = 'Provider',how='left').fillna(value=fillvalues)

	#duplicate claims percentage 
	train_features4['Duplicate_Claims_Percent'] = pd.DataFrame(round((train_features4['DuplicateClaims']/train_features4['Claim_Count']), 2))
	# Claim Duration 
	Claim_dur = full_df2.groupby('Provider')['ClaimDuration'].agg('mean').reset_index(name='AvgClaimDuration')
	train_features5 = pd.merge(train_features4, Claim_dur, on = 'Provider')

	# Average Cost (reimbursed + deductible) across Providers
	full_df2['TotalCharge'] = full_df2['InscClaimAmtReimbursed'] + full_df2['DeductibleAmtPaid']
	AvgCharge = full_df2.groupby('Provider')['TotalCharge'].agg('mean').reset_index(name='Avg_Cost')
	train_features6 = pd.merge(train_features5, AvgCharge, on = 'Provider')

	#age column
	claim_start = full_df['ClaimStartDt'].apply(pd.to_datetime, errors='coerce', format='%Y-%m-%d')
	birth_date = full_df['DOB']

	full_df['Age'] = claim_start - birth_date
	full_df['Age'] = full_df['Age'] / np.timedelta64(1, 'Y')

	#average age column 
	avg_age = full_df.groupby(['Provider', 'BeneID'])['Age'].mean().reset_index(name = "Avg_Age").dropna().groupby('Provider')['Avg_Age'].mean().reset_index()
	avg_age['Avg_Age'] = avg_age['Avg_Age'].astype(int)

	train_features7 = pd.merge(train_features6, avg_age, on='Provider')
	#create gender columns 
	gender1 = full_df[['Provider', 'BeneID', 'Gender']].loc[full_df['Gender'] == 1].groupby('Provider')['BeneID'].nunique().to_frame().reset_index().rename(columns = {'BeneID': 'Gender1'})
	gender2 = full_df[['Provider', 'BeneID', 'Gender']].loc[full_df['Gender'] == 2].groupby('Provider')['BeneID'].nunique().to_frame().reset_index().rename(columns = {'BeneID': 'Gender2'})

	#merge gender1 and gender2
	gender = pd.merge(gender1, gender2, on='Provider',how='outer').fillna(0)
	train_features9 = pd.merge(train_features7, gender, on='Provider')

	#create race columns
	race1 = full_df[['Provider', 'BeneID', 'Race']].loc[full_df['Race'] == 1].groupby('Provider')['BeneID'].nunique().to_frame().reset_index().rename(columns = {'BeneID': 'Race1'})
	race2 = full_df[['Provider', 'BeneID', 'Race']].loc[full_df['Race'] == 2].groupby('Provider')['BeneID'].nunique().to_frame().reset_index().rename(columns = {'BeneID': 'Race2'})
	race3 = full_df[['Provider', 'BeneID', 'Race']].loc[full_df['Race'] == 3].groupby('Provider')['BeneID'].nunique().to_frame().reset_index().rename(columns = {'BeneID': 'Race3'})
	race5 = full_df[['Provider', 'BeneID', 'Race']].loc[full_df['Race'] == 5].groupby('Provider')['BeneID'].nunique().to_frame().reset_index().rename(columns = {'BeneID': 'Race5'})

	#merge race columns
	race = pd.merge(race1, race2, on='Provider',how='outer').fillna(0)
	race = pd.merge(race, race3, on='Provider',how='outer').fillna(0)
	race = pd.merge(race, race5, on='Provider',how='outer').fillna(0)

	train_features13 = pd.merge(train_features9, race, on='Provider')
	# create each chronic condition count column
	chroniclist = ['ChronicCond_Alzheimer', 'ChronicCond_KidneyDisease',
		'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 
		'ChronicCond_Depression', 'ChronicCond_Diabetes',
		'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
		'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']
	conditions = train_features13[['Provider']]
	for cond in chroniclist:
    		df1 = full_df[['Provider', 'BeneID', cond]].loc[full_df[cond] == 1].groupby('Provider')['BeneID'].nunique().reset_index().rename(columns = 																	{'BeneID': cond + '_1'})
    		df2 = full_df[['Provider', 'BeneID', cond]].loc[full_df[cond] == 2].groupby('Provider')['BeneID'].nunique().reset_index().rename(columns = 																	{'BeneID': cond + '_2'})
    		conditions = pd.merge(conditions,df1,on='Provider',how='left').fillna(0)
    		conditions = pd.merge(conditions,df2,on='Provider',how='left').fillna(0)


	train_features23 = pd.merge(train_features13, conditions, on='Provider')
	#import train label df
	label_train_df = pd.read_csv('Train-1542865627584.csv')
	
	#add network degree column
	networkdf = pd.read_csv('networkdf.csv')

	if(j_flag==2):
		return train_features23
	

	#add label column
	features = pd.merge(train_features23, label_train_df, on='Provider')
	features = pd.merge(features, networkdf, on='Provider')
	

	return features

