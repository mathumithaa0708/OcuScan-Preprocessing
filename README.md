# OcuScan – Member 1: Image Preprocessing & Enhancement

This repository contains the preprocessing module of the OcuScan project.
It prepares retinal fundus images for vessel segmentation.

## Dataset
DRIVE retinal image dataset (not included due to size and license).

## Preprocessing Steps
1. Image resizing and standardization
2. Green channel extraction
3. Illumination correction
4. CLAHE contrast enhancement
5. Noise reduction (Gaussian filtering)
6. Image fusion (CLAHE + Top-hat)

## Output
Final enhanced images are saved in:
output/fused/

These images are used as input for the segmentation module (Member 2).

## Tools Used
- Python 3.8
- OpenCV
- NumPy

## Author
Member 1 – Image Preprocessing Lead
