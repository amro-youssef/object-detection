# Object Detection with Classical Computer Vision

This project implements an object detection system using classical computer vision techniques such as SIFT feature matching, RANSAC, homography estimation, and bounding box prediction.

The system detects and classifies icons in images and evaluates detection accuracy using Intersection over Union (IoU).

## Features

- SIFT keypoint detection and matching
- RANSAC-based homography estimation
- Bounding box prediction and visualization
- Evaluation metrics: Accuracy, TPR, FPR, FNR, and Average IoU

## Example Output

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img width="256" height="256" alt="output1" src="https://github.com/user-attachments/assets/9f324a8d-1109-4668-883c-b0700b199a6b" />
  <img width="256" height="256" alt="output2" src="https://github.com/user-attachments/assets/47870714-31cd-470f-9a55-4823e50d273a" />
  <img width="256" height="256" alt="output3" src="https://github.com/user-attachments/assets/7a04fce2-c848-4874-8cc4-7729ca2577cb" />
  <img width="256" height="256" alt="output4" src="https://github.com/user-attachments/assets/c0c70548-61fa-4b10-80d3-1b4c39f51392" />
</div>

## Getting started

To install all the necassary packages run the following command in the terminal
```
pip install -r requirements.txt
```

## Running the programme

To run the programme run the following command in the terminal. This command assumes that the dataset path contains the iconDir (IconDataset) and testDir (Task3Dataset) folders.
```

python .\main.py --Task3Dataset "Task3Dataset" --IconDataset "IconDataset"
```
