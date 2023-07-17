"""Script to split the ADNI dataset into train, validation and test sets."""

import argparse
from pathlib import Path

from data_analysis_preparation.utils import (load_adnimerge,
                                             query_images_paths,
                                             train_val_test_split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Script to split the ADNI dataset into train, validation and test sets. " \
                         "The split is done at the patient level, so that the same patient is not present in more " \
                         "than one set. " \
                         "Split is stratified according to the DX column (i.e. the proportion of patients in each " \
                         "class is as similar as possible in each set)."
    parser.add_argument("--adnimerge_path", type=str, required=True, help="Path to the ADNIMERGE.csv file.")
    parser.add_argument("--images_directory", type=str, required=False, default=None,
                        help="Path to the images directory. If not specified, the output .csv files will not contain "
                             "the path to the images, and all rows will be kept. Otherwise, only rows with matching "
                             "images will be kept.")
    parser.add_argument("--output_directory", type=str, required=False, default="../tabular_data")
    args = parser.parse_args()

    df_adnimerge = load_adnimerge(args.adnimerge_path)

    if args.images_directory is not None:
        df_adnimerge = query_images_paths(df_adnimerge, args.images_directory)

    train_df, val_df, test_df = train_val_test_split(df_adnimerge)
    Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    train_df.to_csv(Path(args.output_directory, "train_base.csv"), index=False)
    print(f"Train set ({len(train_df)} rows) saved to {Path(args.output_directory, 'train_base.csv')}")
    val_df.to_csv(Path(args.output_directory, "val_base.csv"), index=False)
    print(f"Validation set ({len(val_df)} rows) saved to {Path(args.output_directory, 'val_base.csv')}")
    test_df.to_csv(Path(args.output_directory, "test_base.csv"), index=False)
    print(f"Test set ({len(test_df)} rows) saved to {Path(args.output_directory, 'test_base.csv')}")
