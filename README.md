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


## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

[MIT](https://choosealicense.com/licenses/mit/)
