"""Helper functions for ADNI data analysis and preparation."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

base_volumetric_columns = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]
base_biocard_diag_columns = ["JHUANONID", "VISITNO", "MOFROMBL", "STARTYEAR", "BIRTHYR", "DIAGNOSIS", "NORMCOG",
                             "DEMENTED", "IMPNOMCI", "PROBAD", "PROBADIF", "POSSAD", "POSSADIF"]
base_biocard_demographic_columns = ["JHUANONID", "SEX", "EDUC"]
base_biocard_functional_columns = ["JHUANONID", "VISITNO", "MOFROMBL", "MEMORY", "CDRGLOB"]
base_biocard_cognitive_columns = ["JHUANONID", "VISITNO", "MOFROMBL", "MMSE"]


def load_adnimerge(path: str) -> pd.DataFrame:
    """
    Load ADNIMERGE.csv into a pandas dataframe.

    Drop rows with NaN values in IMAGEUID column and convert it to int. Map DX column values to AD, CN and MCI.
    :param path: path to ADNIMERGE.csv file.
    :return: pandas dataframe containing ADNIMERGE.csv.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["IMAGEUID", "DX", "DX_bl"])
    df["IMAGEUID"] = df["IMAGEUID"].astype(int)
    df["DX"] = df["DX"].map({"Dementia": "AD", "CN": "CN", "MCI": "MCI"})
    return df


def get_mci_progression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get DataFrame containing only MCI patients. Split MCI patients into two groups: progressors and non-progressors.

    Progressors are arbitrarily defined as patients who progressed to dementia within 6, 12, 18, 24 or 36 months.
    Non-progressors are patients who did not progress to dementia within 36 months. Patients who progressed to
    dementia after 36 months are removed from the dataset.
    :param df: source DataFrame containing ADNIMERGE.csv data.
    :return: DataFrame containing only MCI patients, with p-MCI and np-MCI labels in the DX column.
    """
    progressors_ptids = []
    viscode_ad = {}
    for patient in df["PTID"].unique():
        df_patient = df[df["PTID"] == patient]
        is_mci = "LMCI" in df_patient["DX_bl"].unique() or "EMCI" in df_patient["DX_bl"].unique()
        progressed_to_dementia = "AD" in df_patient["DX"].unique()
        if is_mci and progressed_to_dementia:
            progressors_ptids.append(patient)
            viscode_ad[patient] = df_patient[df_patient["DX"] == "AD"].VISCODE.min()
    _df = df[df["DX"] == "MCI"].copy(deep=True)
    _df["DX"] = _df["PTID"].apply(lambda x: "p-MCI" if x in progressors_ptids else "np-MCI")
    ptid_to_remove = {ptid for ptid, viscode in viscode_ad.items() if viscode == "m48"}
    _df = _df[~_df["PTID"].isin(ptid_to_remove)]
    return _df


def filter_df_by_problem(problem: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows from the DataFrame according to the problem.

    Problem can be one of the following:
    - MCI vs CN (remove Dementia patients)
    - AD vs MCI (remove CN patients)
    - AD vs CN (remove MCI patients)
    - AD vs MCI vs CN (keep all patients - multiclass classification)
    - p-MCI vs np-MCI (remove Dementia patients and split MCI patients into progressors and non-progressors)
    :param problem: string containing the name of the problem.
    :param df: source DataFrame containing ADNIMERGE.csv data.
    :return: filtered DataFrame.
    """
    if problem == 'MCI vs CN':
        df = df[df["DX"] != "AD"].copy(deep=True)
    elif problem == 'AD vs MCI':
        df = df[df["DX"] != "CN"].copy(deep=True)
    elif problem == 'AD vs CN':
        df = df[df["DX"] != "MCI"].copy(deep=True)
    elif problem == "AD vs MCI vs CN":
        df = df.copy(deep=True)
    elif problem == "p-MCI vs np-MCI":
        df = get_mci_progression(df.copy(deep=True))
    else:
        raise Exception("Problem not found")
    return df


def train_val_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation and test sets.

    The split is done at the patient level, so that the same patient is not present in more than one set.
    Split is stratified according to the DX column (i.e. the proportion of patients in each class is as similar as
    possible in each set).
    :param df: source DataFrame containing ADNIMERGE.csv data.
    :return: train, validation and test sets.
    """
    df.reset_index(inplace=True, drop=True)
    ptid_split = df[["PTID", "DX"]].drop_duplicates(subset="PTID").values
    train_ptid, val_ptid = train_test_split(ptid_split, test_size=0.2, random_state=42, stratify=ptid_split[:, 1])
    val_ptid, test_ptid = train_test_split(val_ptid, test_size=0.5, random_state=42, stratify=val_ptid[:, 1])

    train_df = df[df["PTID"].isin(train_ptid[:, 0])]
    val_df = df[df["PTID"].isin(val_ptid[:, 0])]
    test_df = df[df["PTID"].isin(test_ptid[:, 0])]
    return train_df, val_df, test_df


def query_images_paths(df: pd.DataFrame, images_directory: str | Path, prefix: str = "rescaled") -> pd.DataFrame:
    """
    Query the paths of the images in the dataset.

    Find matching images in the images_paths directory and return a DataFrame containing the paths of the images
    in "path" column. Filter out rows with no matching images.
    :param df: source DataFrame containing ADNIMERGE.csv data.
    :param images_directory: path to the directory containing the images.
    :param prefix: prefix of the images to be found. Prefixes comes from scans_preparation step. After every
    preprocessing step, the image is saved with a different prefix. Last step is "rescaled", so the default value
    is "rescaled".
    :return: DataFrame containing the paths of the images in "path" column.
    """
    scans_paths = list(Path(images_directory).rglob(f"{prefix}*.nii.gz"))
    print(f"Found {len(scans_paths)} images")
    paths_by_imageuid = {int(pth.name.split(".")[0].split("_")[-1][1:]): pth for pth in scans_paths}
    df['path'] = df['IMAGEUID'].map(paths_by_imageuid)
    df = df.dropna(subset=['path'])
    return df


def get_mean_image_per_diagnosis(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get mean image for each diagnosis.

    :param df: source DataFrame containing ADNIMERGE.csv data after query_images_paths step.
    :return: mean image for AD, MCI and CN patients.
    """
    ad_scans = np.zeros((182, 218))
    mci_scans = np.zeros((182, 218))
    cn_scans = np.zeros((182, 218))

    ad_scans_num = 0
    mci_scans_num = 0
    cn_scans_num = 0

    for _, row in df.iterrows():
        image = nib.load(row['path']).get_fdata()
        if row['DX'] == 'AD':
            ad_scans = ad_scans + image[:, :, 80]
            ad_scans_num += 1
        elif row['DX'] == 'MCI':
            mci_scans = mci_scans + image[:, :, 80]
            mci_scans_num += 1
        elif row['DX'] == 'CN':
            cn_scans = cn_scans + image[:, :, 80]
            cn_scans_num += 1
        else:
            print(f'Unknown diagnosis {row["DX"]}')

    ad_scans /= ad_scans_num
    mci_scans /= mci_scans_num
    cn_scans /= cn_scans_num

    return ad_scans, mci_scans, cn_scans


def get_bounding_box(img: np.ndarray) -> tuple[int, int, int, int, int, int, int, int, int, float, float]:
    """
    Get the bounding box of the brain - the smallest box containing all the brain voxels.

    :param img: 3D image of the brain.
    :return: tuple containing the coordinates of the bounding box and its dimensions.
    """
    mask = img > img.min()
    x, y, z = np.where(mask)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin
    min_val, max_val = img.min(), img.max()
    return xmin, xmax, ymin, ymax, zmin, zmax, width, height, depth, min_val, max_val


def timedelta_in_months(end: pd.Timestamp, start: pd.Timestamp) -> int:
    """Compute the number of months between two dates."""
    return 12 * (end.year - start.year) + (end.month - start.month)


def get_months_from_baseline(subject_group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of months from the baseline date for each visit data.

    :param subject_group: DataFrame containing data for a single subject.
    :return: DataFrame with the number of months from the baseline date for each visit data.
    """
    baseline_date = subject_group["Date"].min()
    subject_group["MOFROMBL"] = subject_group["Date"].apply(lambda x: timedelta_in_months(x, baseline_date))
    return subject_group


def get_biocard_df(biocard_diagnosis_data: str | Path, biocard_mri_subject_data: str | Path) -> pd.DataFrame:
    """
    Get the DataFrame containing the data from the BIOCARD study.

    :param biocard_diagnosis_data: Path to the BIOCARD_DiagnosisData_Limited_2022.05.14.xlsx file.
    :param biocard_mri_subject_data: Path to .csv file containing three columns: Scan label, Date, and Subject. This
    file can be obtained from the BIOCARD study in XNAT search tools by using "MR/Spreadsheet" function.
    Make sure to select all scans (5000 rows per tab) and remove unnecessary columns.
    Sample:
    Label,Date,Subject
    JHU001001_1,xxxx-xx-xx,JHU001001
    :return: DataFrame containing joined data from the BIOCARD study.
    """
    mri_subject_data = pd.read_csv(biocard_mri_subject_data)
    mri_subject_data["Date"] = pd.to_datetime(mri_subject_data["Date"])
    mri_subject_data.rename(columns={"Subject": "JHUANONID"}, inplace=True)
    mri_subject_data = mri_subject_data.groupby("JHUANONID").apply(get_months_from_baseline).reset_index(drop=True)
    df_diagnosis = pd.read_excel(biocard_diagnosis_data)
    return mri_subject_data.merge(df_diagnosis, on=["JHUANONID", "MOFROMBL"])


def query_biocard_image_files(path: str | Path, prefix: str = "rescaled") -> list[tuple[str, str, Path]]:
    """
    Query all image files in the given directory.

    :param path: Path to the directory with preprocessed (with run_biocard_scns_preprocessing.py script)
    images in nifti format.
    :param prefix: Prefix of the images to be found. Prefixes comes from scans_preparation step. After every
    preprocessing step, the image is saved with a different prefix. Last step is "rescaled", so the default value
    is "rescaled".
    :return: list of scans labels (JHUANONID_VISITID), scans types (MRI) and paths to the images in nifti format.
    """
    images_paths = Path(path).rglob(f"{prefix}*.nii.gz")
    rows = []
    for pth in images_paths:
        name = "_".join(pth.name.split(".")[0].split("_")[1:]).split("=")
        jhuanonid, visit_id, scan_type = name
        scan_type = scan_type.split("-")[1]
        rows.append((f"{jhuanonid}_{visit_id}", scan_type, pth))
    return rows


def get_joined_biocard_df(biocard_diagnosis_data: str | Path, biocard_mri_subject_data: str | Path,
                          scans_path: str | Path, biocard_demographic_data: str | Path | None = None,
                          biocard_functional_data: str | Path | None = None,
                          biocard_cognitive_data: str | Path | None = None) -> pd.DataFrame:
    """
    Get joined data (diagnosis, imaging, demographic, functional and cognitive) from the BIOCARD study.

    :param biocard_diagnosis_data: Path to the BIOCARD_DiagnosisData_Limited_2022.05.14.xlsx file.
    :param biocard_mri_subject_data: Path to .csv file containing three columns: Scan label, Date, and Subject. This
    file can be obtained from the BIOCARD study in XNAT search tools by using "MR/Spreadsheet" function.
    Make sure to select all scans (5000 rows per tab) and remove unnecessary columns.
    Sample:
    Label,Date,Subject
    JHU001001_1,xxxx-xx-xx,JHU001001
    :param scans_path: Path to the directory with preprocessed (with run_biocard_scns_preprocessing.py script)
    images in nifti format.
    :param biocard_demographic_data: Path to the BIOCARD_Demographics_Limited_Data_2022.05.10.xlsx file. Optional.
    If not provided, the demographic data will not be added to the DataFrame.
    :param biocard_functional_data: Path to the BIOCARD_Functional_Evaluation_Limited_2022.05.16.xlsx file. Optional.
    If not provided, the functional data will not be added to the DataFrame.
    :param biocard_cognitive_data: Path to the BIOCARD_CognitiveData_Limited_2022.05.14.xlsx file. Optional.
    If not provided, the cognitive data will not be added to the DataFrame.
    :return: DataFrame containing joined data from the BIOCARD study.
    """
    scans_data_columns_names = ["Label", "SCANTYPE", "path"]

    df_biocard = get_biocard_df(biocard_diagnosis_data, biocard_mri_subject_data)
    df_images_paths = pd.DataFrame(query_biocard_image_files(scans_path), columns=scans_data_columns_names)
    df_images_paths = df_images_paths.merge(df_biocard.copy(), on="Label", how="left")
    df_diag = df_images_paths[base_biocard_diag_columns + scans_data_columns_names].copy()
    df_diag.PROBAD = df_diag.PROBAD.fillna(0)
    df_diag["AGE"] = df_diag.STARTYEAR - df_diag.BIRTHYR
    df_diag.dropna(subset=["JHUANONID", "VISITNO", "DIAGNOSIS"], inplace=True)

    if biocard_demographic_data is not None:
        df_demographic = pd.read_excel(biocard_demographic_data)[base_biocard_demographic_columns]
        df_demographic["SEX"] = df_demographic["SEX"].map({1: "Male", 2: "Female"})
        df_diag = df_diag.merge(df_demographic.copy(), on="JHUANONID", how="left", validate="many_to_one")

    if biocard_functional_data is not None:
        df_functional = pd.read_excel(biocard_functional_data)[base_biocard_functional_columns]
        df_functional.MEMORY = df_functional.MEMORY.fillna(0)
        df_functional.CDRGLOB = df_functional.MEMORY.fillna(0)
        df_diag = df_diag.merge(df_functional.copy(), on=["JHUANONID", "VISITNO", "MOFROMBL"], how="left",
                                validate="many_to_one")

    if biocard_cognitive_data is not None:
        df_cognitive = pd.read_excel(biocard_cognitive_data)[base_biocard_cognitive_columns]
        df_diag = df_diag.merge(df_cognitive.copy(), on=["JHUANONID", "VISITNO", "MOFROMBL"], how="left",
                                validate="many_to_one")

    return df_diag


def filter_biocard_diagnosis_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query BIOCARD dataset with diagnostic criteria from ADNI study. The criteria are as follows:
    - CN: MMSE >= 24, CDRGLOB = 0, CDR MEMORY = 0, Probable AD conditions not met
    - MCI: MMSE >= 24, CDRGLOB = 0.5, CDR MEMORY >= 0.5, Probable AD conditions not met
    - AD: MMSE: 20-26, CDRGLOB >= 0.5, CDR MEMORY >= 0.5, Probable AD conditions met

    Education adjusted WMS-R Logical Memory II (Delayed Paragrash Recall) score was also used in the ADNI study, but
    it is not available in the BIOCARD at this level of detail.
    :param df: DataFrame containing joined data from the BIOCARD study (get_joined_biocard_df function).
    :return: DataFrame with revised diagnosis.
    """
    df_cn = df[(df["MMSE"] >= 24) & (df["CDRGLOB"] == 0.0) & (df["MEMORY"] == 0.0) & (df["PROBAD"] == 0)].copy()
    df_cn["DX"] = "CN"
    df_mci = df[(df["MMSE"] >= 24) & (df["CDRGLOB"] == 0.5) & (df["MEMORY"] >= 0.5) & (df["PROBAD"] == 0)].copy()
    df_mci["DX"] = "MCI"
    df_ad = df[(df["MMSE"].between(20, 26)) & (df["CDRGLOB"] >= 0.5)
               & (df["MEMORY"] >= 0.5) & (df["PROBAD"] == 1.0)].copy()
    df_ad["DX"] = "AD"
    df = pd.concat([df_ad, df_mci, df_cn])
    return df
