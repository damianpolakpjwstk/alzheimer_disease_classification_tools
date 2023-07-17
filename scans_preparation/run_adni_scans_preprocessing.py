"""Run preprocessing on ADNI dataset."""
from scans_preparation.utils import get_scans_paths, run_processing

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True)
    parser.add_argument("--export_directory", type=str, required=True)
    args = parser.parse_args()
    mri_paths = get_scans_paths(args.root_directory)
    run_processing(mri_paths, args.export_directory, overwrite=False)
