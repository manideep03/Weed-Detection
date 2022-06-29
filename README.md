# Weed-Detection


## What is this project about ?


Weeds are the unwanted crops in agriculture lands. They disturb the growing plants around them. Weeds are dependent plants which use nutrients, water, etc resources. So we need to remove them as early as possible. So this project will help to detect the weed in between plants further we can add a mechanical arm which will automatically pick weed and remove them. 


## What will this give you ?

We can use this code to detect weeds in agriculture lands.


## How i did it ?

I used **YOLO(You Only Look Once)** Real time object detection algorithm to detect weeds. Dataset i used is taken from [kaggle](https://www.kaggle.com/) and dataset name is *crop and weed detection data with bounding boxes* ([Link to dataset](https://www.kaggle.com/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)) which contains around *1300 images*. Traning is done in [Goole Colab](https://colab.research.google.com/).

## How to use ?
Structure of the repository

```
├───testing
│   ├───images
│   └───results
└───traning
    ├───backup
    └───test
```

- Traning folder consists of files used for traning
- Testing folder consists of testing file

Clone the repo and Upload Traning file in google drive and open ipynb file in colab and then you can change the parameters and you can play with the code.

For testing and for using the project download weights from the [link](https://mega.nz/file/LIcFWZhb#XZ9YACBuAz2jeiklyqiDN1AGyDbfvztacRIlar9wP7k) and paste it in testing folder.

In testing folder we have two files use ***detect_image.py*** to detect in image and use ***detect_video.py*** for detecting in video and with webcame access.


## Results

- Program detected crop in image


![detect_crop01](https://github.com/manideep03/Weed-Detection/blob/main/testing/results/detect_crop01.png)

- Program detected weed in image

![detect_weed01](https://github.com/manideep03/Weed-Detection/blob/main/testing/results/detect_weed01.png)


- Program detecting weeds and crops in video

![gif](https://github.com/manideep03/Weed-Detection/blob/main/testing/results/video_detection.gif)
