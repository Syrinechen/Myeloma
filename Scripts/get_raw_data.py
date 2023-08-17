#%%
import pandas as pd
import numpy as np
from pathlib import Path    
from tqdm import tqdm,trange
import csv
#%%
#Some patients in the csv files do not have their id but their admission number. 
#This dictionary contains the correspondance between the two in order to change the ones with admission number
def Correspondances_interne_id():
    d={}
    with open("/home/irit/Documents/Myeloma_Syrine/Data/Utilities/nums_internes_cassioppe.tsv") as f:
        print('here')
        for line in f:
            (value, key)=line.split()
            d[key]=value[2:8]
    return d
#%% These functions are used to get the data for regression problem (using mrd rates)
def get_patients_rates(patient_id):
    """
    Retrieves the MRD rate of a specific patient.

    Args:
        patient_id (int): The ID of the patient.

    Returns:
        int or None: The rate of the patient if found, None otherwise.
    """
    # Check if the patient ID is in the rates index
    if patient_id not in rates.index:
        # If not found, append the patient ID to the rate_not_found list
        rate_not_found.append(patient_id)
        return None
    elif rates.loc[patient_id]['mrd post-conso (T3)'] == 'NE':
        # If the MRD rate is 'NE', return None
        return None
    else:
        # Return the MRD rate as an integer
        return int(rates.loc[patient_id]['mrd post-conso (T3)']) 
def get_response_rate(rate):
    """
    Classify the response rate based on the given rate into 3 different classes : 0 (MRD negatif), 1 (positif), 2 (zone grise) 

    Parameters:
        rate (int): The rate value to calculate the response rate.

    Returns:
        int or None: The response class. Returns None if rate is None.
                     Returns 0 if rate is 0.
                     Returns 2 if rate is less than 10.
                     Returns 1 for any other rate value.
    """
    if rate is None:
        return None
    elif rate == 0:
        return 0
    elif rate < 10:
        return 2
    else:
        return 1
#%% These functions are used to get the data for classification using MRD responses directly
def get_patient_response(patient_id):
    """
    Get the response for a given patient ID.
    
    Args:
        patient_id (str): The ID of the patient.
        
    Returns:
        int or None: The patient's response, or None if the response is not found or is indeterminate.
    """
    if patient_id not in responses.index:
        response_not_found.append(patient_id)
        return None
    elif responses.loc[patient_id]['post_consolidation_MRD at 10-5'] == 'INDETERMINATE':
        return None
    elif responses.loc[patient_id]['post_consolidation_MRD at 10-5'] == 'POSITIVE':
        return 1
    else:
        return 0
# %%
def get_data():
    """
    Retrieves data from a directory containing text files and processes it.

    Returns:
    - patient_list: list of patient IDs
    - count_list: list of count values
    - responses: list of patient responses
    - rates: list of patient rates
    - res_rates: list of response rates
    """
    directory = '/home/irit/Documents/Myeloma_Syrine/Data/allRawCounts'
    pathlist = Path(directory).glob('*.txt')
    patient_list = []
    count_list = []
    responses = []
    rates = []
    res_rates = []

    dict_interne_id = Correspondances_interne_id()

    for path in tqdm(pathlist):
        patient_id = str(path).split('/')[7].split('_')[2]

        if patient_id[0] == 'T':
            if patient_id not in dict_interne_id.keys():
                print(patient_id, ' not found')
            else:
                patient_id = dict_interne_id[patient_id]
                
        raw_counts = pd.read_csv(str(path), sep='\t', header=1)
        count_list.append(raw_counts[raw_counts.columns[-1]].values.tolist())
        patient_list.append(patient_id)
        responses.append(get_patient_response(patient_id))
        rate = get_patients_rates(patient_id)
        rates.append(rate)
        res_rates.append(get_response_rate(rate))

    return patient_list, count_list, responses, rates, res_rates

    

# %% execute this to get the files raw_count_mrd_response.csv and raw_count_mrd_rate.csv
if __name__ == "__main__":
    response_not_found=[]
    rate_not_found=[]
    #preprocess MRD rates file
    rates=pd.read_csv('/home/irit/Documents/Myeloma_Syrine/Data/valeurs_brutes_mrd_completed.csv')
    for i in range(len(rates)):
        if(len(rates['id'][i])>6):
            rates['id'][i]=rates['id'][i][2:8]
    rates.index=rates['id']
    rates=rates.drop('id',axis=1)
    #preprocess MRD Response file
    responses=pd.read_excel('/home/irit/Documents/Myeloma_Syrine/Data/Cassiopee Reponse-MRD.xls')[['USUBJID','post_consolidation_MRD at 10-5']]
    for i in range (len(responses)):
        responses['USUBJID'].iloc[i]=responses['USUBJID'].iloc[i][-6:]
    responses.index=responses['USUBJID']
    reponses=responses.drop('USUBJID',axis=1)
    #get data
    patient_list,raw_counts,mrd_response,mrd_rate,rate_response=get_data()
    #create final datasets
    gene_list=pd.read_csv('/home/irit/Documents/Myeloma_Syrine/Data/allRawCounts/Run_20190201_T12973_featureCounts.txt',sep='\t',header=1)['Geneid'].values.tolist()
    data_classif=pd.DataFrame(raw_counts,index=patient_list,columns=gene_list)
    data_reg=pd.DataFrame(raw_counts,index=patient_list,columns=gene_list)
    data_classif['MRD Response']=mrd_response
    data_reg['MRD Rate']=mrd_rate
    data_reg['MRD Response']=rate_response
    data_classif=data_classif.dropna()
    data_reg=data_reg.dropna()
    data_classif.to_csv('raw_count_mrd_response.csv',index_label='patient_id')
    data_reg.to_csv('raw_count_mrd_values.csv',index_label='patient_id')    
# %%
