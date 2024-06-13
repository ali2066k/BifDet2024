# BifDet: A 3D Bifurcation Detection Dataset for Airway-Tree Modeling

This repository contains the BifDet dataset, the first publicly available dataset specialized for 3D airway bifurcation detection in thoracic Computed Tomography (CT) scans. Bifurcations, the points where airways diverge, are crucial for understanding lung physiology and disease mechanisms.

## Key Features
- **3D Bifurcation Bounding Boxes**: Carefully annotated CT scans with precise bifurcation bounding boxes covering the parent and daughter nodes.
- **Tailored Detection Task**: A standardized framework for 3D airway bifurcation detection.
- **Comprehensive Pipeline**: Detailed methodological pipeline, including preprocessing steps and code implementations using the MONAI framework.
- **Baseline Models**: Benchmark models across various object detection categories (RetinaNet, Deformable DETR) to facilitate future research.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Data Acquisition](#data-acquisition)
  - [Bifurcation Annotation](#bifurcation-annotation)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction)
  - [Detection Methods](#detection-methods)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Code](#running-the-code)
  - [Preparing the Dataset](#preparing-the-dataset)
  - [Run the pipeline](#run-pipeline)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

BifDet presents a novel approach to 3D airway bifurcation detection, providing a dataset, problem formulation, and a methodological pipeline tailored for this task. The dataset includes 22 cases from the publicly available ATM22 dataset, annotated with bounding boxes around bifurcations to capture their full morphological context.

## Dataset

### Data Acquisition

The BifDet dataset is derived from the ATM22 dataset, specifically utilizing 22 cases from the 300 publicly available CT scans. Each case includes segmentation masks for the trachea, main bronchi, lobar bronchi, and distal segmental bronchi.

![Annotation Pipeline](/img/annot_pipeline.pdf)

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

## Run the pipeline

## Acknowledgments
This research was funded by the Doctoral School of IP Paris and Hi!Paris, and utilized HPC resources from GENCI-IDRIS (Grant 2023-AD011013999). We acknowledge the support of the National Institutes of Health (NIH) under grant R01-HL155816 and the ATM22 organizers for their invaluable efforts.

## License
This dataset is released under the Creative Commons BY-NC-SA 4.0 license.


For more detailed documentation and information, please refer to the full paper.
