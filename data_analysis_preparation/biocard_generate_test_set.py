"""
Generate test set for CNN models using BIOCARD data.

For each row corresponding to singleMRI scan, only MPRAGE scans are considered. Each scan is assigned a label based on
the diagnosis criteria similar to the one used in ADNI study. Used criteria are described in
data_analysis_preparation/utils.py/filter_biocard_diagnosis_criteria() function docstring.
"""
from data_analysis_preparation.utils import (filter_biocard_diagnosis_criteria,
                                             get_joined_biocard_df)

BIOCARD_SCANS_DIRECTORY = r"/media/dpolak/MRI/FULL_PIPELINE_BIOCARD"
BIOCARD_DIAGNOSIS_DATA = r"../DATA/BIOCARD_DiagnosisData_Limited_2022.05.14.xlsx"
BIOCARD_DEMOGRAPHIC_DATA = r"../DATA/BIOCARD_Demographics_Limited_Data_2022.05.10.xlsx"
BIOCARD_FUNCTIONAL_DATA = r"../DATA/BIOCARD_Functional_Evaluation_Limited_2022.05.16.xlsx"
BIOCARD_COGNITIVE_DATA = r"../DATA/BIOCARD_CognitiveData_Limited_2022.05.14.xlsx"
BIOCARD_MRI_SUBJECT_DATA = "../DATA/BIOCARD_SUBJECTS.csv"
EXPORT_FILE = r"../tabular_data/biocard_test_set.csv"

if __name__ == "__main__":
    df_biocard = get_joined_biocard_df(biocard_diagnosis_data=BIOCARD_DIAGNOSIS_DATA,
                                       biocard_mri_subject_data=BIOCARD_MRI_SUBJECT_DATA,
                                       scans_path=BIOCARD_SCANS_DIRECTORY,
                                       biocard_demographic_data=BIOCARD_DEMOGRAPHIC_DATA,
                                       biocard_functional_data=BIOCARD_FUNCTIONAL_DATA,
                                       biocard_cognitive_data=BIOCARD_COGNITIVE_DATA)
    df_biocard = df_biocard[df_biocard["SCANTYPE"] == "MPRAGE"]
    df_biocard_revised = filter_biocard_diagnosis_criteria(df_biocard)
    df_biocard_revised.to_csv(EXPORT_FILE, index=False)
