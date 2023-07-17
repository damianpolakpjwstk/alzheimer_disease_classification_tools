"""
Configuration file for the scans_preparation package.

In order to use the scans_preparation package, you need to install FSL and ANTs packages.
Provide the paths to the FSL and ANTs binaries below.
"""

import os

path_env = os.environ["PATH"]
ants_dir = "/home/dpolak/ants-2.4.3/"  # Provide path to ANTs installation directory.
ants_bin = os.path.join(ants_dir, "bin")
os.environ["PATH"] = path_env + ":" + ants_bin
os.environ["FSLDIR"] = "/home/dpolak/fsl"  # Provide path to FSL installation directory.
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

flirt_bin_path = os.path.join(os.environ["FSLDIR"], "bin", "flirt")

# You can change the reference template to the one you prefer.
reference_template_path = os.path.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_brain.nii.gz")

# HD-BET parameters - check the HD-BET documentation for more details.
hd_bet_parameters = {
    "mode": "accurate",
    "keep_mask": False,
    "postprocess": True,
    "do_tta": True,
    "overwrite": True
}
