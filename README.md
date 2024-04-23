# Background Subtraction


The problem of separating backgrounds and moving objects is an important part of modern photo and video processing. As digital technologies are rapidly evolving, this improves the qual- ity of photo and video materials, and thus increases the amount of memory required to store and process the matrices that repre- sent the corresponding images or videos. Thus, in this study, the authors try to solve the problem of background subtraction in a computationally efficient way. For background estimation, low- rank matrix approximations are used, which are obtained by the computationally efficient Randomized SVD algorithm. This re- port provides an explanation of the Randomized SVD algorithm and the results of background approximation. In conclusion, the authors argue that SVD is a powerful tool in general, but it is best used for photo and video preprocessing.

## Data Preparation

Before application of any algorithm, video should be correctly preprocessed. To obtain a matrix that represents a video, the team decided to extract each frame of the video according to the video’s per second frame rate. Then each frame was converted into grayscale and flattened into a column. Each flattened vector then forms a video matrix, columns of which represent a flattened frames in correct chronological order.

## Example of video sample matrix:

![matrix_visualisation](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/5dc97995-9abd-4c29-8dd7-0ace193b84f3)

As one can see, there are some black lines along the horizontal axes. Such diffrence represents the moving objects that are present on the particular video frame.

## Idea Behind Background Approximation

To obtaine the best background approximation model it is appropriate to constuct a rank-1 approximation of the matrix. Since singular values are in descending order, then the product of first singular value and its corresponding singular vectors forms a matrix each column of which represents dominant pattern for each particular frame. Taking into account that background is static, then the dominant pattern formed by the product of the first singular value and its corresponding singular vectors represents models background. Such low-rank approximation may be obtained by Randomized SVD

## Example of Background Model

![low_rank_representation](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/6f0e43b1-4d4f-4a2e-815d-14f9758173fb)

It is clearly visible that black lines representing the movements are excluded.

## Results
The dataset including video samples was constructed by the team. Here are the axample of background subtraction.
![ts_outdoor_woman_25_10_50_480_270_frame](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/ff663141-ab34-4131-8d2c-726a7a80921d)
![ts_outdoor_woman_25_10_50_480_270_background](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/fd6dd192-b8f9-4a66-8554-64d3fbe4d198)
![ts_outdoor_woman_50_10_50_960_540_foreground](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/ca4d5349-3201-490d-b4a0-e2c6e2b9df86)

Composed Video:
https://github.com/sharaievPn/background-foreground-separation/assets/116552240/96d6369f-2228-4d2b-a662-93f1c1089e4f

## Requirments
The algorithms use different packages which may be installed through command line. 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages. Run the commands in command line in directory where the Separator.py is located.

```bash
pip install numpy
pip install moviepy
pip install opencv-python
pip install matplotlib
pip install ffmpeg-python
```

Also ensure there are such directories like 
```bash
video
matrices_data
svd_results
video_matrix
video_without_background
```

## Usage
The main class is Separator which should be imported.

```python
from Separator import Separator
```

The object of the class may be created parametrized or unparametrized, where parameter is the name of the video sample.
```python
separator = Separator('Video.MOV')
```

```python
separator = Separator()
```

Are both correct ways to initialize an onbject of the class.

There is also a feature to load data from json, but it will work only when the video was once processed.

```python
separator.load_json("file.json")
```

## First Time Initialization
If the initialization is performed for the first, meaning the video is loaded for the first time, then you should specify characteristics of the video (note: it is not required while loading data through json, but it is required using any other methods).
The characteristics are: 
1) Start second (for subclip)
2) End second (for subclip)
3) scale - factor to make the subclip of smaller size
Important: scale factor of 100 represents the full size of the video
4) Tag - represents environemnt where the video has been shot
5) Place - specific place like room name, location name (for example, stairs)
6) Feature - what is the featuring element of the video

All other characteristics will be calculated by the program

Input example:

<img width="272" alt="image" src="https://github.com/sharaievPn/background-foreground-separation/assets/116552240/4de09fec-c687-4d67-9fec-bf7793a9a50e">

## Functionality

1) Display background
```python
separator.display_background()
```
Displays background of the video sample

2) Display foreground
```python
separator.display_foreground(second)
```
To display foreground provide the specific second you would like to display the frame of. As a result will be displayed a specific frame and separated foreground

3) Create a video without background
```python
separator.create_video_without_background()
```

## More Results
![k2_stairs_freshman_25_5_26_270_480_second](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/783b0394-2c09-49a6-9c35-ac7904f7cfc3)
![k2_stairs_freshman_25_5_26_270_480_background](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/81319f44-bc6c-4d32-ae9a-1e2a3a3ae0dc)
![k2_stairs_freshman_25_5_26_270_480_foreground](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/f12781f3-ef4e-476c-8317-0c5c2f14147c)

![ts_stairs_man_25_0_18_270_480_second](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/38002f7f-6737-4353-b970-6d1506d75198)
![ts_stairs_man_25_0_18_270_480_background](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/c69bc2b6-25be-4d8f-a852-183c1067bd89)
![ts_stairs_man_25_0_18_270_480_foreground](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/7f978e4f-d57d-4a9f-bc6a-24d3a55fb1d6)

Video: https://github.com/sharaievPn/background-foreground-separation/assets/116552240/222b6029-b7a9-4d70-8046-212114e5bb3e

![ts_second-floor_elevator_25_60_110_480_270_second](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/c57e51fa-0281-4285-a896-c454e3467334)
![ts_second-floor_elevator_25_60_110_480_270_background](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/6dff66b4-c781-4f63-812d-9f3088635dda)
![ts_second-floor_elevator_25_60_110_480_270_foreground](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/ce102fda-9374-4622-9fde-9f1b81a74c36)

Video:  https://github.com/sharaievPn/background-foreground-separation/assets/116552240/45813347-6fd6-4fb9-aa6a-9c84b22a17f0

![ts_stairs_wall_25_30_90_480_270_second](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/4f89e413-ac52-45f0-a3c0-3d515fbb284d)
![ts_stairs_wall_25_30_90_480_270_background](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/d7fdba51-010a-487d-aa65-c70676db1a90)
![ts_stairs_wall_25_30_90_480_270_foreground](https://github.com/sharaievPn/background-foreground-separation/assets/116552240/68015af5-feb8-48aa-bbc1-6c6601b7d330)

Video: https://github.com/sharaievPn/background-foreground-separation/assets/116552240/bec49ec0-3b58-4ce7-a3ae-df0bb64da656




