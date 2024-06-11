## Preparing the Dataset

1. **Download the BifDet Dataset**:
   Download the BifDet dataset from the following link:
   [BifDet Dataset](https://...)

2. **Download the ATM22 Dataset**:
   Download the ATM22 dataset from the following link:
   [ATM22 Challenge website](https://...)
   Ensure it is extracted to an accessible directory. The structure should look like this:

   ```plaintext
   ATM22/
   └── TrainBatch1/
       ├── imagesTr/
       ├── labelsTr/


Run the Data Setup Script:
The data setup script data_setup.py will copy the necessary files and prepare the dataset for training. It will also filter bounding boxes based on size and create JSON files with the processed data.

Run the script using the following command:

```
python data_setup.py --src_base_dir "path_to_ATM22/TrainBatch1/" --base_dir "path_to_BifDet_Dataset/"
```

Replace "path_to_ATM22/TrainBatch1/" and "path_to_BifDet_Dataset/" with the actual paths to your datasets. For example:

python data_setup.py --src_base_dir "./ATM22/TrainBatch1/" --base_dir "./bifdet2024/exp/"

**Script Explanation**

The data_setup.py script performs the following tasks:

Load Case Mapping: Reads the case mapping from case_mapping.json.

Create Directory Structure: Creates the necessary directories for each BifDet case.

Copy Files: Copies the CT scan nifti files and airway segmentation ground-truth nifti files from the ATM22 dataset to the BifDet dataset.

Load and Filter Bounding Boxes: Loads bounding boxes from JSON files, filters them based on size, and converts coordinates.

Save Processed Data: Saves the processed data into JSON files for training.

**Output**

The script generates JSON files in the specified base_dir with the following naming convention:

```BifDet_lbl{lbl_tag}_min_{min_s}.json```

(lbl_tag is always 1 and min_s is the minimum cubic size for the bounding boxes)

Each JSON file contains the processed data for training.

Notes
Ensure that the paths provided to the script are correct and the directories exist.
The script includes print statements to provide feedback on the progress of the file processing and bounding box filtering.
By following these steps, you will have the BifDet dataset ready for training.
