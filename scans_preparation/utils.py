"""
Helper functions for preprocessing MRI brain scans to be used in CNN models.

Some of the functions are adapted from the following repository:
https://github.com/quqixun/BrainPrep/tree/master
"""
import os
from pathlib import Path
from time import time

import nibabel as nib
import numpy as np
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from scans_preparation.configuration import (flirt_bin_path, hd_bet_parameters,
                                             reference_template_path)
from scipy.ndimage import zoom
from scipy.signal import medfilt

from HD_BET.run import run_hd_bet


def load_nii(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load nifti file and return image data and affine.

    :param path: path to .nii or .nii.gz file.
    :return: Loaded image data as numpy array and affine matrix.
    """
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine


def save_nii(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    """
    Save data as nifti file.

    :param data: numpy array containing image data
    :param affine: affine matrix.
    :param path: path to save the nifti file to.
    """
    nib.save(nib.Nifti1Image(data, affine), path)


def downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample the image volume by the given factor.

    :param data: Image volume to be downsampled.
    :param factor: Downsampling factor.
    :return: Downsampled image volume.
    """
    return zoom(data, 1 / factor, order=1)


def bias_field_correction(src_path: str | Path, dst_path: str | Path) -> None:
    """
    Perform N4ITK bias field correction on the image at src_path and save the result at dst_path.

    :param src_path: path to the image to be corrected.
    :param dst_path: path to save the corrected image to.
    """
    print("N4ITK on: ", src_path)
    n4 = N4BiasFieldCorrection()
    n4.inputs.input_image = src_path
    n4.inputs.output_image = dst_path

    n4.inputs.dimension = 3
    n4.inputs.n_iterations = [100, 100, 60, 40]
    n4.inputs.shrink_factor = 3
    n4.inputs.convergence_threshold = 1e-4
    n4.inputs.bspline_fitting_distance = 300
    n4.run()


def denoise(volume: np.ndarray, kernel_size: int = 3):
    """Denoise using a median filter."""
    return medfilt(volume, kernel_size)


def rescale_intensity(volume: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5),
                      bins_num: int = 256) -> np.ndarray:
    """
    Rescale the intensity of the image volume.

    Quantize the image volume into bins_num bins and map them to the corresponding intensity values.
    :param volume: Image volume to be rescaled.
    :param percentiles: Lower and upper percentiles for rescaling. Defines min and max values.
    :param bins_num: Number of bins to use for rescaling. If 0, use simple min-max normalization.
    :return: Rescaled image volume.
    """
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentiles[0])
    max_value = np.percentile(obj_volume, percentiles[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume

    return volume


def equalize_hist(volume: np.ndarray, bins_num=256) -> np.ndarray:
    """
    Perform histogram equalization of the image volume.

    Redistribute the intensity values of the image volume to enhance contrast.
    :param volume: Image volume to be equalized.
    :param bins_num: Number of bins to use for equalization.
    :return: Equalized image volume.
    """
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, density=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def get_scans_paths(directory: str | Path) -> list[Path]:
    """
    Recursively get all MRI scans in a directory.
    :param directory:
    :return: list of file paths of scans in nifti format.
    """
    directory = Path(directory)
    return list(directory.rglob(r"*.nii")) + list(directory.rglob(r"*.nii.gz"))


def run_processing(mri_paths: list[str | Path], output_path: str | Path, overwrite: bool = False) -> None:
    """
    Run the preprocessing pipeline on the MRI scans.

    :param mri_paths: path to the directory with MRI scans.
    :param output_path: path to the directory where the preprocessed scans will be saved.
    :param overwrite: if True, the existing files will be overwritten. Otherwise, the existing files will be skipped.
    """
    Path(output_path).mkdir(exist_ok=True, parents=True)
    failed_records_log = open(Path(output_path, "failed_records.log"), "w+")
    for path in mri_paths:
        t0 = time()
        out_path = Path(output_path, path.parent.__str__().split("/")[-1])
        out_path.mkdir(exist_ok=True, parents=True)
        filename = f"{str(path.name).split('.')[0]}.nii.gz"

        try:
            bet_output_path = Path(out_path, f"BET_{filename}").__str__()
            n4_output_path = Path(out_path, f"N4_{filename}").__str__()
            alignment_output_path = Path(out_path, f"alignment_{filename}").__str__()
            denoised_output_path = Path(out_path, f"denoised_{filename}").__str__()
            rescaled_output_path = Path(out_path, f"rescaled_{filename}").__str__()
            equalized_output_path = Path(out_path, f"equalized_{filename}").__str__()

            if not overwrite and not Path(bet_output_path).exists():
                run_hd_bet(str(path), str(bet_output_path), **hd_bet_parameters)

            if not overwrite and not Path(n4_output_path).exists():
                bias_field_correction(bet_output_path, n4_output_path)

            if not overwrite and not Path(alignment_output_path).exists():
                alignment_command = f"{flirt_bin_path} -in {n4_output_path} -ref {reference_template_path}" \
                                    f" -out {alignment_output_path} -dof 6"
                os.system(alignment_command)

            volume, affine = load_nii(alignment_output_path)

            if not overwrite and not Path(denoised_output_path).exists():
                volume = denoise(volume)
                save_nii(volume, affine, denoised_output_path)

            if not overwrite and not Path(rescaled_output_path).exists():
                volume = rescale_intensity(volume)
                save_nii(volume, affine, rescaled_output_path)

            if not overwrite and not Path(equalized_output_path).exists():
                volume = equalize_hist(volume)
                save_nii(volume, affine, equalized_output_path)
        except Exception as e:
            print(f"Failed to process {path} in {time() - t0} seconds.")
            print(e)
            failed_records_log.write(f"{path},{e}\n")
            failed_records_log.flush()
            continue
        print(f"Processed {path} in {time() - t0} seconds.")
    failed_records_log.close()
    print(f"Done. Failed records are logged in {failed_records_log.name}.")
