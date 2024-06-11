import os
import json
import shutil
import glob
import re
import nibabel as nib
import numpy as np
import argparse


def load_case_mapping(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_directory_structure(base_dir, case):
    paths = [
        os.path.join(base_dir, case, 'imagesTr'),
        os.path.join(base_dir, case, 'labelsTr')
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def copy_files(case_mapping, src_images_dir, src_labels_dir, dest_base_dir):
    for bifdet_case, atm_case in case_mapping.items():
        print(f"We are processing case {bifdet_case}!")
        create_directory_structure(dest_base_dir, bifdet_case)
        
        src_image_file = os.path.join(src_images_dir, f"{atm_case}_0000.nii.gz")
        dest_image_file = os.path.join(dest_base_dir, bifdet_case, 'imagesTr', f"{atm_case}_0000.nii.gz")
        shutil.copy(src_image_file, dest_image_file)
        print(dest_image_file)
        
        src_label_file = os.path.join(src_labels_dir, f"{atm_case}_0000.nii.gz")
        dest_label_file = os.path.join(dest_base_dir, bifdet_case, 'labelsTr', f"{atm_case}_0000.nii.gz")
        shutil.copy(src_label_file, dest_label_file)
        print(dest_label_file)
        print("="*30)

        
def load_bboxes(path, lbl_air_path, case_mapping, min_dim_sizes=[1,1,1], lbl_tag=0):
    """Loads bounding boxes from JSON files and filters based on size."""
    list_of_dict = []
    for case_path in glob.glob(os.path.join(path, "*")):
        gen_dict = {}
        case_name = os.path.basename(case_path)
        gen_dict["case_name"] = f"BifDet_{case_name.split('_')[-1]}"

        # Load label data and its affine matrix
        lbl, lbl_name = load_lbl(lbl_air_path, case_name, case_mapping)

        # Load and filter points
        all_points = glob.glob(os.path.join(case_path, "boxes", "*.json"))
        num_whole = len(all_points)
        all_points = [item for item in all_points if not re.search("trachea", item)]
        
        json_list = []
        num_filtered = 0  # Count the number of filtered boxes

        for point_file in all_points:
            with open(point_file) as f:
                bbox = load_points(json.load(f), lbl, min_dim_sizes)
                if bbox is not None:  # Only append if box is valid
                    json_list.append(bbox)
                else:
                    num_filtered += 1  

        print(f"{case_name} --> {num_filtered} out {num_whole} of boxes filtered out due to the minimum size of {min_dim_sizes}. Trachea: {num_whole-len(all_points)}. Remained: {num_whole-1-num_filtered}")  # Output filter count

        gen_dict["case_name"] = gen_dict["case_name"]
        gen_dict["image"] = lbl_name
        gen_dict["lung"] = lbl_name
        gen_dict["awlabel"] = lbl_name
        gen_dict["boxes"] = json_list
        gen_dict["label"] = [lbl_tag] * len(json_list)
        list_of_dict.append(gen_dict)

    return list_of_dict

def load_lbl(lbl_path, c_name, case_mapping):
    """Loads the label NIfTI file based on the case name."""
    original_name = case_mapping.get(f"BifDet_{c_name.split('_')[-1]}", None)
    if not original_name:
        raise ValueError(f"Case name {c_name} not found in the case mapping dictionary.")
    pth = os.path.join(lbl_path, f"{original_name}_0000.nii.gz")
    nifti_f_src = nib.load(pth)
    return nifti_f_src, f"{original_name}_0000.nii.gz"

def load_points(data_point, lbl, MIN_SIZES=[1, 1, 1]):
    """Converts annotation center and size to RAS voxel coordinates and checks size thresholds."""
    
    roi = data_point["markups"][0]
    center_lps = np.array(roi["center"])
    size_lps = np.array(roi["size"])

    # Convert center from LPS to RAS (voxel) coordinates
    center_ras = nib.affines.apply_affine(lbl.affine, center_lps)
    center_vox = np.round(center_ras).astype(int)

    # Calculate bounding box corners in RAS (voxel) coordinates
    half_size = size_lps / 2
    corners_ras = np.array([center_ras - half_size, center_ras + half_size])

    # Convert corners to voxel indices
    corners_vox = np.round(nib.affines.apply_affine(np.linalg.inv(lbl.affine), corners_ras)).astype(int)

    # Calculate bounding box size in voxels
    size_vox = np.abs(corners_vox[1] - corners_vox[0])
    
    voxel_sizes = np.abs(np.diag(lbl.affine)[:3]) 
    if any((size_vox * voxel_sizes)[i]  - size_lps[i] for i in range(3)) > 1:
        print("Warning: Inaccurate coordinate conversion detected.")
        return None
    
    # Check if each dimension exceeds its minimum
    if any(size_vox[i] < MIN_SIZES[i] for i in range(3)):
        return None
    else:
        return [*center_lps, *size_lps]  # Return center and size in original LPS units

def main():
    parser = argparse.ArgumentParser(description="Setup BifDet dataset from ATM22 dataset.")
    parser.add_argument('--atm22_path', type=str, required=True, help='Path to the base directory of the ATM22 dataset')
    parser.add_argument('--destination_base_dir', type=str, required=True, help='Path to the base directory where BifDet dataset will be created')
    args = parser.parse_args()

    src_base_dir = args.atm22_path
    src_images_dir = os.path.join(src_base_dir, "imagesTr")
    src_labels_dir = os.path.join(src_base_dir, "labelsTr")
    base_dir = args.destination_base_dir
    dest_base_dir = os.path.join(base_dir, 'BifDet_Dataset')
    case_mapping_file = os.path.join(base_dir, 'case_mapping.json')

    case_mapping = load_case_mapping(case_mapping_file)
    copy_files(case_mapping, src_images_dir, src_labels_dir, dest_base_dir)
    
    min_ss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lbl_tags = [1]
    for lbl_tag in lbl_tags:
        for min_s in min_ss:
            MIN_SIZES = [min_s, min_s, min_s]  # Minimum sizes for x, y, and z dimensions respectively
            d = load_bboxes(path=dest_base_dir, lbl_air_path=src_labels_dir, case_mapping=case_mapping, min_dim_sizes=MIN_SIZES, lbl_tag=lbl_tag)
            par_d = {
                "training": d
            }
            json_save_path = os.path.join(base_dir, f'BifDet_lbl{lbl_tag}_min_{min_s}.json')
            with open(json_save_path, 'w') as outfile:
                json.dump(par_d, outfile)
            print("="*20)
