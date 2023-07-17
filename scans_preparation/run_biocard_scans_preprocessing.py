"""Run preprocessing on BIOCARD dataset."""
from multiprocessing import Pool
from pathlib import Path

import dicom2nifti
from scans_preparation.utils import get_scans_paths, run_processing


def get_dicom_scans_paths(root_directory: str | Path) -> list[Path]:
    """
    Get paths to DICOM directories.

    :param root_directory: Directory with DICOM files. Subdirectories will be searched recursively.
    :return: List of paths to DICOM files.
    """
    print(f"Searching for DICOM files in {root_directory}...")
    scan_dirs = list(Path(root_directory).rglob("*/*/files"))
    return scan_dirs


def convert_dicom_to_nifti(dicom_dir: Path, export_directory: str | Path) -> None:
    """
    Convert DICOM files to NIfTI format.

    :param dicom_dir: Paths to directory with DICOM files.
    :param export_directory: Path to the directory where the NIfTI files will be saved.
    """
    export_directory = Path(export_directory)
    export_directory.mkdir(parents=True, exist_ok=True)
    failed_records_log = open(Path(export_directory, "failed_dicom2nifti_records.log"), "a+")

    print(f"Converting {dicom_dir}...")
    pth_str = str(dicom_dir)
    patient_id = pth_str.split("/")[5].split("_")[0]
    scan_id = pth_str.split("/")[5].split("_")[1]
    scan_type = pth_str.split("/")[7]
    dirname = f"{patient_id}={scan_id}={scan_type}"

    output_path = Path(export_directory, dirname)
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        dicom2nifti.convert_directory(dicom_dir, output_path, compression=True, reorient=True)
    except Exception as e:
        print(f"Failed to convert {dicom_dir}. Error: {e}")
        failed_records_log.write(f"{dicom_dir},{e}\n")
        failed_records_log.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True)
    parser.add_argument("--export_directory", type=str, required=True)
    args = parser.parse_args()
    scan_directories = get_dicom_scans_paths(args.root_directory)
    export_path = args.export_directory

    def _convert(dicom_dir):
        """Wrapper for convert_dicom_to_nifti function to use with multiprocessing.Pool"""
        convert_dicom_to_nifti(dicom_dir, export_path)

    with Pool(8) as p:
        p.map(_convert, scan_directories)
    run_processing(get_scans_paths(args.export_directory), args.export_directory, overwrite=False)
