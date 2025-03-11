# NOTE: to run this code you must have the following pip packages installed (I ran using python 3.9):
# pydicom, tifffile, matplotlib, numpy , scipy, pymedphys\

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.interpolate import interp1d
from pymedphys import gamma
from pathlib import Path
from FilmLab.utils import FilmImage, NDVoxelArray
import cv2
import seaborn as sns

from FilmLab.vic_dicom import  FilmLabDoseFile

# Directories where we're working
BASEDIR = Path("")
DATADIR = BASEDIR / Path("sample_data")
OUTDIR = BASEDIR / Path("sample_output")

# The eclipse doses
python_dose_output_file = OUTDIR / "sample_Dose_Data.npz"
matlab_dose_output_file = OUTDIR / "sample_Dose_Data.mat"

# The resulting film doses
film_python_output_file_rc = OUTDIR / "sample_Film_Dose_rc.npz"
film_matlab_output_file_rc = OUTDIR / "sample_Film_Dose_rc.mat"
film_python_output_file_gc = OUTDIR / "sample_Film_Dose_gc.npz"
film_matlab_output_file_gc = OUTDIR / "sample_Film_Dose_gc.mat"
film_python_output_file_bc = OUTDIR / "sample_Film_Dose_bc.npz"
film_matlab_output_file_bc = OUTDIR / "sample_Film_Dose_bc.mat"
film_matlab_output_file_avg = OUTDIR / "sample_Film_Dose_avg.npz"

channel_suffix_map = {
    "red": "rc",
    "green": "gc",
    "blue": "bc"
}

# dicom file locations containing Eclipse AAA dose
victoria_aaa_dose_file = BASEDIR / "dicom" / "RD.$Physics539.TLDandFilm.Victoria.dcm"
vancouver_aaa_dose_file = BASEDIR / "dicom" / "RD.$Physics539.OSLDandFilm.Vancouver.dcm"
kelowna_aaa_dose_file = BASEDIR / "dicom" / "RD.$Physics539.OSLDandFilm.Kelowna.dcm"

def main():

    plt.switch_backend('TkAgg')  # or 'Qt5Agg' if TkAgg is not available
    
    # do full film calibration and calculate plan dose
    #type: channel
    calibrate_film("red")
    calibrate_film("green")

    # extract isocentre slice from dicom file, output as python and matlab formats
    #file: the .dcm file with the planar dose
    extract_dicom(victoria_aaa_dose_file)

    # do comparison of dose profiles and do gamma test
    # Channel_1: first channel
    # Channel_2: second channel
    # Channel_3: where eclipse dose is stored in NDVoxelArray
    compare_dose_file("red", "green", python_dose_output_file)



def calibrate_film(type):
    sns.set_style("whitegrid")  # Set Seaborn style
    channel_attr = f"{type}_channel"

    # Define calibration film file locations
    calib_file_paths = [DATADIR / "cal{:03}.tif".format(s) for s in range(1, 11)]
    flood_file = DATADIR / "flood.tif"
    film_DPI = 72

    # Define doses
    film_calib_doses_cgy = [0, 10, 10, 10, 25, 50, 100, 150, 200, 250]

    # Load flood image
    flood_image = getattr(FilmImage(flood_file, dpi=film_DPI), channel_attr)

    # Load calibration images
    calib = [getattr(FilmImage(s, dpi=film_DPI), channel_attr) for s in calib_file_paths]

    # Calculate optical densities
    calib_od = [np.log10(flood_image / s) for s in calib]

    # visualize all the calibration images
    # comment out this for-loop to skip the image popups
    for cal_od,fn in zip(calib_od, calib_file_paths):
        fig, ax = plt.subplots()
        ax.imshow(cal_od, origin='lower', cmap='Greys')
        ax.set_title("full image - " + os.path.basename(fn))
        plt.show()

    # use visualization above to pick ROI for calibration, best to use same image size/shape for each film!
    # I chose these pixel ranges from the images x:100->200  y:120->220

    # Crop center pixels
    calib_od_cropped = [s[120:220, 100:200] for s in calib_od]

    # inspect images for defects, make sure you chose the right pixel ranges
    # comment out this for-loop to skip the image popups
    for cal_od,fn in zip(calib_od_cropped, calib_file_paths):
        fig, ax = plt.subplots()
        ax.imshow(cal_od, origin='lower', cmap='Greys')
        ax.set_title("calibration ROI - "+ os.path.basename(fn))
        plt.show() 

    # Compute mean optical densities
    calib_pixel_values = [np.mean(s) for s in calib_od_cropped]

    # Average the 10 cGy readings
    od_10cGy = np.mean(calib_pixel_values[1:4])
    calib_pixel_values = [calib_pixel_values[0], od_10cGy] + calib_pixel_values[4:]
    film_calib_doses_cgy = [0, 10, 25, 50, 100, 150, 200, 250]

    # Compute calibration curves
    calibration_curve = interp1d(calib_pixel_values, film_calib_doses_cgy, kind='cubic')
    calibration_curve_poly = np.poly1d(np.polyfit(calib_pixel_values, film_calib_doses_cgy, 3))

    # Generate OD range for plotting
    od_range = np.linspace(calib_pixel_values[0], calib_pixel_values[-1], 100)

    # 🎨 **Plot Calibration Curve**
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(calib_pixel_values, film_calib_doses_cgy, 'o', label='Calibration Points')
    ax.plot(od_range, calibration_curve(od_range), '-', label='Cubic Spline', color='blue')
    ax.plot(od_range, calibration_curve_poly(od_range), '--', label='3rd Order Polynomial', color='red')
    ax.set_ylabel("Dose (cGy)")
    ax.set_xlabel("Optical Density")
    ax.set_title(f"{type.capitalize()} Channel - Film Calibration", fontsize=14)
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 🎨 **Load & Process Plan Film**
    plan_file = DATADIR / "plan001.tif"
    plan_image_handle = FilmImage(plan_file, dpi=film_DPI)
    new_flood_image = getattr(FilmImage(flood_file, dpi=film_DPI), channel_attr)
    plan_image = getattr(plan_image_handle, channel_attr)

    # Compute optical density & dose
    plan_od = np.log10(new_flood_image / plan_image)
    plan_dose = calibration_curve(plan_od)

    # Compute film image origin
    size_mm = np.array(np.shape(plan_dose)) * plan_image_handle.reso_mm
    origin = [-0.5 * size_mm[0], -0.5 * size_mm[1]]

    # 🎨 **Plot Film Dose**
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(plan_dose, origin='lower', cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dose (cGy)")
    ax.set_title(f'Calculated Film Dose (cGy) ({type.capitalize()} Channel)', fontsize=14)
    plt.show()

    # 🔥 **Save Data**
    save_files = {
        "red": (film_python_output_file_rc, film_matlab_output_file_rc),
        "green": (film_python_output_file_gc, film_matlab_output_file_gc),
        "blue": (film_python_output_file_bc, film_matlab_output_file_bc),
    }
    
    if type in save_files:
        np.savez(save_files[type][0], dose_cgy=plan_dose, origin_mm=origin, pixel_size_mm=[plan_image_handle.reso_mm, plan_image_handle.reso_mm])
        savemat(save_files[type][1], {'dose_cgy': plan_dose, 'origin_mm': origin, 'pixel_size_mm': [plan_image_handle.reso_mm, plan_image_handle.reso_mm]})




def extract_dicom(dicom_file_path):
    sns.set_style("white")  # Use clean white background style

    # Load DICOM dose data
    rtp = FilmLabDoseFile(dicom_file_path)
    isocentre = rtp.plan_dicom.isocenter

    # Find the nearest voxel index to the isocenter
    iso_voxel = rtp.dose_array_cgy.coord_tuple_to_nearest_index(isocentre)
    print(f"Isocentre Voxel: {iso_voxel}")

    # Extract the dose slice at the isocenter level
    isocentre_slice_cGy = rtp.dose_array_cgy.slice_3D_to_2D(1, iso_voxel[1])

    # 🎨 **Plot Dose Map with Colorbar**
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(isocentre_slice_cGy, cmap="viridis", **isocentre_slice_cGy.PLOT_KWARGS)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dose (cGy)")

    # Improve Title & Labels
    ax.set_title("AAA Dose from Eclipse (cGy)", fontsize=14)
    ax.set_xlabel("X-Axis (mm)")
    ax.set_ylabel("Z-Axis (mm)")

    plt.tight_layout()
    plt.show()

    # 🔥 **Save in Python Format**
    np.savez(python_dose_output_file, 
             dose_cgy=isocentre_slice_cGy, 
             origin_mm=isocentre_slice_cGy.origin, 
             pixel_size_mm=isocentre_slice_cGy.voxdims)

    # 🔥 **Save in MATLAB Format**
    savemat(matlab_dose_output_file, {
        'dose_cgy': isocentre_slice_cGy, 
        'origin_mm': isocentre_slice_cGy.origin, 
        'pixel_size_mm': isocentre_slice_cGy.voxdims
    })
    
def combine_dose_files(channel_1, channel_2):
    suffix_1 = channel_suffix_map.get(channel_1)
    suffix_2 = channel_suffix_map.get(channel_2)
    
    if not suffix_1 or not suffix_2:
        raise ValueError("Invalid channel name. Use 'red', 'green', or 'blue'.")

    # Construct the file paths dynamically
    film_file_1 = OUTDIR / f"sample_Film_Dose_{suffix_1}.npz"
    film_file_2 = OUTDIR / f"sample_Film_Dose_{suffix_2}.npz"
    
    film_data_1 = np.load(film_file_1)
    film_dose_1 = NDVoxelArray(film_data_1['dose_cgy'], origin=film_data_1['origin_mm'], voxdims=film_data_1['pixel_size_mm'])

    film_data_2 = np.load(film_file_2)
    film_dose_2 = NDVoxelArray(film_data_2['dose_cgy'], origin=film_data_2['origin_mm'], voxdims=film_data_2['pixel_size_mm'])

        # Ensure the arrays have the same shape before averaging
    if film_dose_1.shape != film_dose_2.shape:
        raise ValueError("Mismatched dose array shapes. Resampling may be needed.")

    # Compute the average dose
    avg_dose_cgy = (film_dose_1 + film_dose_2) / 2

    # Create a new NDVoxelArray with the averaged dose
    avg_film_dose = NDVoxelArray(
        avg_dose_cgy, 
        origin=film_dose_1.origin,  # Assuming both have the same origin
        voxdims=film_dose_1.voxdims  # Assuming both have the same voxel dimensions
    )

    return avg_film_dose



def register_ndvoxelarray_ecc(film_dose: NDVoxelArray, aaa_dose: NDVoxelArray):
    # Extract the dose arrays
    img1 = film_dose.astype(np.float32)  # Convert NDVoxelArray to float32
    img2 = aaa_dose.astype(np.float32)

    # 🔹 Store original min/max values for later rescaling
    min_dose, max_dose = aaa_dose.min(), aaa_dose.max()

    # Normalize images to range [0, 255] for ECC
    img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ECC requires an initial affine transform matrix (2x3)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(img1_norm, img2_norm, warp_matrix, cv2.MOTION_AFFINE, criteria)
        print(f"ECC Convergence Score: {cc}")

        # Apply the transformation
        height, width = img1.shape
        aligned_img_norm = cv2.warpAffine(img2_norm, warp_matrix, (width, height), 
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # 🔹 Rescale back to original dose values
        aligned_img = (aligned_img_norm / 255.0) * (max_dose - min_dose) + min_dose

        # Convert back to NDVoxelArray with original dose range
        aligned_ndvoxel = NDVoxelArray(aligned_img, origin=film_dose.origin, voxdims=film_dose.voxdims)
        return aligned_ndvoxel, warp_matrix

    except cv2.error:
        raise RuntimeError("ECC failed to converge.")


def compare_dose_file(channel_1, channel_2, dose_file):
    sns.set_style("white")  
    sns.set_context("talk")  

    # Load Film & AAA Data
    film_dose = combine_dose_files(channel_1, channel_2)
    aaa_data = np.load(dose_file)
    aaa_dose = NDVoxelArray(aaa_data['dose_cgy'], origin=aaa_data['origin_mm'], voxdims=aaa_data['pixel_size_mm'])

    # Crop & Resample
    aaa_dose = aaa_dose.crop_to_NDVoxel(film_dose)
    film_dose = film_dose.interp_to_NDVoxel(aaa_dose)

    # Register images
    aligned_aaa_dose, transform_matrix = register_ndvoxelarray_ecc(film_dose, aaa_dose)

    # Gamma Analysis
    gamma_percent_pass = 2
    gamma_mm_pass = 2
    gamma_options = {
        'dose_percent_threshold': gamma_percent_pass,
        'distance_mm_threshold': gamma_mm_pass,
        'lower_percent_dose_cutoff': 10,
        'interp_fraction': 20,  
        'max_gamma': 2,
        'random_subset': None,
        'local_gamma': False,
        'ram_available': 2 ** 32
    }

    gimg = gamma(
        (film_dose.coordinate_mesh[0], film_dose.coordinate_mesh[1]), film_dose,
        (aligned_aaa_dose.coordinate_mesh[0], aligned_aaa_dose.coordinate_mesh[1]), aligned_aaa_dose, **gamma_options
    )

    fail = np.where(gimg > 1)
    gimg_fail = np.zeros_like(gimg)
    gimg_fail[fail] = 1
    valid_gamma = gimg[~np.isnan(gimg)]
    passing = 100 * np.round(np.sum(valid_gamma <= 1) / len(valid_gamma), 4)

    # 🎯 **Dose Difference Map**
    dose_diff = film_dose - aligned_aaa_dose
    diff_min, diff_max = np.percentile(dose_diff, [1, 99])

    # 🎨 **Colormap Choices**
    diff_cmap = "bwr"
    min_dose = min(film_dose.min(), aligned_aaa_dose.min())
    max_dose = max(film_dose.max(), aligned_aaa_dose.max())

    # 📊 **Updated Figure Layout (2x3)**
    fig, ax = plt.subplots(2, 3, figsize=(15, 12))

    # 🎭 **Dose Images (Shared Color Scale)**
    im0 = ax[0, 0].imshow(film_dose, vmin=min_dose, vmax=max_dose, cmap='viridis', **film_dose.PLOT_KWARGS)
    im1 = ax[0, 1].imshow(aligned_aaa_dose, vmin=min_dose, vmax=max_dose, cmap='viridis', **aligned_aaa_dose.PLOT_KWARGS)

    # 🎭 **Gamma Map & Failures**
    im2 = ax[1, 0].imshow(gimg, cmap='coolwarm', **film_dose.PLOT_KWARGS)
    im3 = ax[1, 1].imshow(gimg_fail, cmap="Reds", **film_dose.PLOT_KWARGS)

    # 🎭 **Dose Difference Map**
    im4 = ax[1, 2].imshow(dose_diff, cmap=diff_cmap, vmin=diff_min, vmax=diff_max, **film_dose.PLOT_KWARGS)

    # 🔥 **Add Colorbars**
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04).set_label("Dose (cGy)")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04).set_label("Gamma Index")
    fig.colorbar(im4, ax=ax[1, 2], fraction=0.046, pad=0.04).set_label("Dose Difference (cGy)")

    # 🏷 **Titles**
    ax[0, 0].set_title("Film Dose", fontsize=14)
    ax[0, 1].set_title("AAA Dose", fontsize=14)
    ax[1, 0].set_title(f"Gamma {gamma_percent_pass}%/{gamma_mm_pass}mm", fontsize=14)
    ax[1, 1].set_title(f"Gamma Fails | Passing rate = {round(passing, 2)}%", fontsize=14)
    ax[1, 2].set_title("Dose Difference (Film - AAA)", fontsize=14)

    # 🏷 **Hide Empty Axis**
    ax[0, 2].axis("off")  

    plt.tight_layout()
    plt.show()

    # 📊 **Extract X & Y Dose Profiles**
    center_x = film_dose.shape[1] // 2  # Middle column
    center_y = film_dose.shape[0] // 2  # Middle row

    film_x_profile = film_dose[center_y, :]
    film_y_profile = film_dose[:, center_x]

    aaa_x_profile = aligned_aaa_dose[center_y, :]
    aaa_y_profile = aligned_aaa_dose[:, center_x]

    x_axis = np.linspace(film_dose.origin[1], film_dose.origin[1] + film_dose.shape[1] * film_dose.voxdims[1], film_dose.shape[1])
    y_axis = np.linspace(film_dose.origin[0], film_dose.origin[0] + film_dose.shape[0] * film_dose.voxdims[0], film_dose.shape[0])

    # 🎯 **X and Y Profile Comparisons**
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 🔹 **X Profile (Horizontal)**
    ax[0].plot(x_axis, film_x_profile, label="Film Dose", color='blue', linewidth=2)
    ax[0].plot(x_axis, aaa_x_profile, label="AAA Dose", color='orange', linestyle='dashed', linewidth=2)
    ax[0].set_title("X Profile (Middle Row)", fontsize=14)
    ax[0].set_xlabel("X Position (mm)")
    ax[0].set_ylabel("Dose (cGy)")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.6)

    # 🔹 **Y Profile (Vertical)**
    ax[1].plot(y_axis, film_y_profile, label="Film Dose", color='blue', linewidth=2)
    ax[1].plot(y_axis, aaa_y_profile, label="AAA Dose", color='orange', linestyle='dashed', linewidth=2)
    ax[1].set_title("Y Profile (Middle Column)", fontsize=14)
    ax[1].set_xlabel("Y Position (mm)")
    ax[1].set_ylabel("Dose (cGy)")
    ax[1].legend()
    ax[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # 📊 **Dose & Gamma Histograms**
    film_dose_values = film_dose.flatten()
    aaa_dose_values = aligned_aaa_dose.flatten()

    # 🎯 **Create Figure with Two Subplots**
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)  # Share y-axis for better comparison

    # 🎭 **1. Film Dose Histogram**
    sns.histplot(film_dose_values, bins=30, kde=True, ax=ax[0], color='blue', alpha=0.6)
    ax[0].set_title("Film Dose Distribution", fontsize=14)
    ax[0].set_xlabel("Dose (cGy)")
    ax[0].set_ylabel("Number of Pixels")
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # 🎭 **2. AAA Dose Histogram**
    sns.histplot(aaa_dose_values, bins=30, kde=True, ax=ax[1], color='orange', alpha=0.6)
    ax[1].set_title("AAA Dose Distribution", fontsize=14)
    ax[1].set_xlabel("Dose (cGy)")
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # 🔥 **Adjust Layout**
    plt.tight_layout()
    plt.show()

    # 🎯 **2. Cumulative Dose Histogram**
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(film_dose_values, label="Film Dose", color='blue')
    sns.ecdfplot(aaa_dose_values, label="AAA Dose", color='orange')
    plt.title("Cumulative Dose Distribution", fontsize=14)
    plt.xlabel("Dose (cGy)")
    plt.ylabel("Cumulative Probability")
    plt.axhline(0.5, color='gray', linestyle="dashed", alpha=0.5, label="50% Threshold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == '__main__':

    main()