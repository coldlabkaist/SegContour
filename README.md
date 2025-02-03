# SegContour
Automatic Contour Masking Tool for color-segmented mouse videos
- for research/acadmic use only!!

This program automatically generates contours around mice in grayscale 
videos with a bright background. The contours are created by detecting 
edges around the mice.

Before running this program, ensure that the input videos have been 
color-segmented properly.

To use this program:
1. Specify the input video files.
2. Specify the output directory for the processed videos.
3. Click the 'Process' button to apply contours to the videos.

**Note**: The program's performance may vary depending on recording 
conditions, environment, and the quality of the color segmentation.

## Features
- You can import single or multiple videos using the upper 'Browse' button.
- You can select the output video directory using the lower 'Browse' button.
- You can mask the contour onto the input video and export the preprocessed videos by clicking the 'Process' button.
- You can monitor the progress through the progress bar.
- You can quit the program using the 'Close' button.

## Release Notes
### 25.01.16
SegContour 1.0.0 released

## Requirements
- opencv-python>=4.10.0.84
- numpy>=1.26.4
- scipy>=1.13.1

--

Developed by PSW

Copyright (c) 2025, coldlabkaist
