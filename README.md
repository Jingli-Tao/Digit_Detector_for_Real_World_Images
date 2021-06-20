## Project Intro/Objective
The purpose of this project is to develope a multi-class classifier for digit detection and recognition in real-world images. The classifier should be able to perform two tasks: 1) detect digit in real-world images; 2) if a digit is detected, recognize it from 0 to 9. The classifier needs to be invariant to conditions including scale, location, font, pose, lighting, and noise and robust to complex scene background.

### Methods Used
* Maximally Stable Extremal Regions (MSERs)
* Connected component splitting
* Convolutional Neural Network (CNN)

### Libraries
* PyTorch 
* OpenCV
* Numpy

## Set up
```sh
conda env create -f digit_detector.yml
```
## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Donwload raw images (train.tar.gz, test.tar.gz, train_32x32.mat, and test_32x32.mat) from http://ufldl.stanford.edu/housenumbers/. 
3. Process and transform images.
```sh
python data_create.py
```
4. Train model and tune hyperparameters.
```sh
python experiment.py
```
5. Evaluate each model over validation set.
```sh
python evaluate.py
```
6. Apply the best model on real-world images.
```sh
python run.py
```

## Featured Notebooks/Analysis/Deliverables
* [Digit Detector for Real World Images](https://github.com/Jingli-Tao/Digit_Detector_for_Real_World_Images/blob/main/Digit%20Detector%20for%20Real%20World%20Images.ipynb)