# BifDet: A 3D Bifurcation Detection Dataset for Airway-Tree Modeling

This repository contains the BifDet dataset, the first publicly available dataset specialized for 3D airway bifurcation detection in thoracic Computed Tomography (CT) scans. Bifurcations, the points where airways diverge, are crucial for understanding lung physiology and disease mechanisms.

## Key Features
- **3D Bifurcation Bounding Boxes**: Carefully annotated CT scans with precise bifurcation bounding boxes covering the parent and daughter nodes.
- **Tailored Detection Task**: A standardized framework for 3D airway bifurcation detection.
- **Comprehensive Pipeline**: Detailed methodological pipeline, including preprocessing steps and code implementations using the MONAI framework.
- **Baseline Models**: Benchmark models across various object detection categories (RetinaNet, Deformable DETR) to facilitate future research.

## Table of Contents
- [BifDet: A 3D Bifurcation Detection Dataset for Airway-Tree Modeling](#bifdet-a-3d-bifurcation-detection-dataset-for-airway-tree-modeling)
  - [Key Features](#key-features)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
    - [Data Acquisition](#data-acquisition)
    - [Bifurcation Annotation](#bifurcation-annotation)
  - [Methodology](#methodology)
    - [Feature Extraction](#feature-extraction)
    - [Detection Methods](#detection-methods)
      - [RetinaNet for 3D Bifurcation Detection](#retinanet-for-3d-bifurcation-detection)
  - [Usage](#usage)
    - [Installation](#installation)
  - [Preparing the Dataset](#preparing-the-dataset)
  - [Loading BifDet Data into 3D Slicer](#loading-bifdet-data-into-3d-slicer)
  - [Training RetinaNet on BifDet](#training-retinanet-on-bifdet)
  - [Training DefDETR on BifDET](#training-defdetr-on-bifdet)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## Introduction

BifDet presents a novel approach to 3D airway bifurcation detection, providing a dataset, problem formulation, and a methodological pipeline tailored for this task. The dataset includes 22 cases from the publicly available ATM22 dataset, annotated with bounding boxes around bifurcations to capture their full morphological context.

## Dataset

### Data Acquisition

The BifDet dataset is derived from the ATM22 dataset, specifically utilizing 22 cases from the 300 publicly available CT scans. Each case includes segmentation masks for the trachea, main bronchi, lobar bronchi, and distal segmental bronchi.

![Annotation Pipeline](/img/annot_pipeline.png)

### Bifurcation Annotation

The annotation process was meticulously carried out by a lung CT expert, under the supervision of a medical imaging expert and a clinician specializing in pulmonary and respiratory health. Bounding boxes were annotated to tightly encompass the bifurcations, covering the parent and both daughter branches to ensure comprehensive coverage.

## Methodology

### Feature Extraction

We compare deep learning-based 3D detection methods using the ResNet50 backbone architecture for feature extraction. A Feature Pyramid Network (FPN) is attached to enrich the feature representation, enabling detection of bifurcations at various scales.

### Detection Methods

#### RetinaNet for 3D Bifurcation Detection

As a baseline, we use RetinaNet, a one-stage detector based on FPN for feature extraction and two sub-networks for candidate box classification and regression. Anchors are generated at each location within the feature maps, representing potential locations and sizes of airway bifurcations.

## Usage

### Installation

To set up the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your_username/BifDet.git
cd BifDet
pip install -r requirements.txt
```
You can also install the packages within a new conda environment directly with:

```bash
conda env create -f environment.yml
```

## Preparing the Dataset

1. **Download the BifDet Dataset**:
   Download the BifDet dataset from the following link:
   [BifDet Dataset](https://...)
   Ensure it is extracted to an accessible directory. The structure should look like this:

   ```
   release
   ├── BifDet_Dataset/
      ├── BifDet_001/
      │   ├── imagesTr/
      │   ├── labelsTr/
      │   ├── lungsTr/
      │   ├── render/
      │   └── boxes/
      ├── BifDet_002/
      │   ├── imagesTr/
      │   ├── labelsTr/
      │   ├── lungsTr/
      │   ├── render/
      │   └── boxes/
      └── ...
   ├── training
      │   ├── imagesTr/
      │   ├── labelsTr/
      │   ├── lungsTr/
      │   ├── annotation JSON files for different setups 

3. **Download the ATM22 Dataset**:
   Download the ATM22 dataset from the following link:
   [ATM22 Challenge website](https://...)
   Ensure it is extracted to an accessible directory. The structure should look like this:

   ```
   ATM22/
   └── TrainBatch1/
       ├── imagesTr/
       │   ├── ATM_001_0000.nii.gz
       │   ├── ATM_002_0000.nii.gz
       │   └── ...
       ├── labelsTr/
       │   ├── ATM_001_0000.nii.gz
       │   ├── ATM_002_0000.nii.gz
       │   └── ...


Run the Data Setup Script:
The data setup script data_setup.py will copy the necessary files and prepare the dataset for training. It will also filter bounding boxes based on size and create JSON files with the processed data.

Run the script using the following command:

```
python data_setup.py --atm22_path "path_to_ATM22/TrainBatch1/" --destination_base_dir "path_to_BifDet_Dataset/"
```

Replace "path_to_ATM22/TrainBatch1/" and "path_to_BifDet_Dataset/" with the actual paths to your datasets. For example:

python data_setup.py --atm22_path "./ATM22/TrainBatch1/" --destination_base_dir "./bifdet2024/exp/"

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



## Loading BifDet Data into 3D Slicer

Prerequisites:

3D Slicer: Make sure you have 3D Slicer installed. You can download it from https://download.slicer.org/.
BifDet & ATM22 Datasets: Ensure you have downloaded and prepared both the BifDet and ATM22 datasets as described in the "Preparing the Dataset" section of your repository.
Steps:

Load CT Scan (imagesTr):

In 3D Slicer, go to "File" -> "Add Data."
Navigate to the imagesTr folder of the desired BifDet case (e.g., BifDet_Dataset/BifDet_001/imagesTr).
Select the .nii.gz file containing the CT scan data.
Click "Add."

Load Airway Segmentation Ground Truth (labelsTr):

Repeat the same steps as above, but select the .nii.gz file from the labelsTr folder of the same case and choose "Segmentation" as the description when loading the file. This will load the ground truth segmentation.

Load 3D Rendering (render):

Go to "File" -> "Add Data."
Navigate to the render folder and select the 3D model .vtk file.
Click "Add."

Load Bounding Boxes (boxes):
Same procedure and load the json files from "boxes" directory

## Training RetinaNet on BifDet

1. Update the paths in `os_retinanet/config.env` :
   - **Output path** `OUTPUT_PATH`: Indicate where checkpoints and logs will be saved. Should be set to a specific folder in your working directory.
   - **Data path** `DATA_SRC`: Indicate where the BifDet dataset is located. 
   - **Annotaton path** `ANNOT_FNAME`: Indicate the *name* of the json file containing all the annotations. This json file should be located in the folder of the BifDet dataset. The annotations will be loaded as `DATA_SRC/ANNOT_FNAME`.
2. Run the training script `os_retinanet/run.sh`.
   - Default hyperparameters used for experiments in the paper are already set as arguments in the training script.
3. Evaluation can be done using the same script with the flags `--eval` and `--resume PATH_TO_CKPT`

## Training DefDETR on BifDET

1. Compile CUDA operations for the deformable attention module.
   - Run `./def_detr/models/transoar/ops/make.sh` to install, then `./def_detr/models/transoar/ops/test.py` to verify installation.
   - Setup and test files can be found in `./def_detr/models/transoar/ops/setup.py` and `./def_detr/models/transoar/ops/test.py`
   - Refer to [transoar github](https://github.com/bwittmann/transoar) or [original Deformable DETR github](https://github.com/fundamentalvision/Deformable-DETR) for more informations.
2. Update the paths in `def_detr/config.env`
3. Run the training script `def_detr/training.py`.
4. Evaluation can be done using the same script with the flags `--eval` and `--resume PATH_TO_CKPT`

## Acknowledgments
This research was funded by the Doctoral School of IP Paris and Hi!Paris, and utilized HPC resources from GENCI-IDRIS (Grant 2023-AD011013999). We acknowledge the support of the National Institutes of Health (NIH) under grant R01-HL155816 and the ATM22 organizers for their invaluable efforts.

## License
This dataset is released under the Creative Commons BY-NC-SA 4.0 license.


For more detailed documentation and information, please refer to the full paper.
