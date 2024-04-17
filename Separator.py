import numpy as np
import os
from moviepy.editor import VideoFileClip, VideoClip
import cv2 as cv
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage


class IncorrectName(Exception):
    """
    Raised the file or directory does not exist
    """
    def __init__(self, message):
        super().__init__(message)


class IncorrectDuration(Exception):
    """
    Raised when the problems connected to the video duration occur
    """
    def __init__(self, message):
        super().__init__(message)


class IncorrectFrame(Exception):
    """
    Raised when specific issues with video frames occur
    """
    def __init__(self, message):
        super().__init__(message)


class Separator:
    """
    Class which provides functionality on background and foreground separation
    :param default_dims: Default dimensions of the output video. Used when dimensions are unknown
    """
    default_dims = (1080, 1920)

    def __init__(self, video_name: str = None, matrix_name: str = None) -> None:
        """

        :param video_name:
        :param matrix_name:
        """
        self.video_name = video_name
        self.matrix_name = matrix_name
        self.clip = None
        self.fps = None
        self.start_second = None
        self.end_second = None
        self.scale = None
        self.tag = None
        self.place = None
        self.feature = None
        self.matrix = None
        self.U = None
        self.s = None
        self.V = None
        self.dims = None
        self.low_rank = None

    def __grab_video_details(self):
        """
        Gather essential characteristics of the video
        :return:
        """
        if self.video_name is None:
            raise IncorrectName("The video was not specified")

        print('Specify video characteristics: ')
        start_second = float(input('Start second: '))
        end_second = float(input('End second: '))

        if start_second < 0 or end_second < 0:
            raise IncorrectDuration("Time must be positive")

        if start_second < end_second:
            raise IncorrectDuration("Starting second must be lower than ending second")
        video = VideoFileClip(self.video_name)
        if end_second > video.duration:
            end_second = video.duration

        self.start_second = start_second
        self.end_second = end_second
        self.scale = float(input('Scale'))
        self.tag = input('Tag: ')
        self.place = input('Place: ')
        self.feature = input('Feature')

        self.fps = self.clip.fps
        self.dims = self.clip.size[0] * self.scale / 100, self.clip.size[1] * self.scale / 100

    def open(self, matrix_name: str = None, video_name: str = None):
        """
        Visualises background according to the provided video or matrix name. Checks whether such video or matrix exists
        :param matrix_name: name of the matrix
        :param video_name: name of the video
        :return:
        """
        if matrix_name is None and video_name is None:
            raise IncorrectName("There aren't any materials provided")

        if matrix_name is not None:
            if self.__check_matrix(matrix_name):
                self.matrix = np.load(matrix_name)
                self.__perform_svd(True, 1, 1, 5)
                self.display_background()
                return

        if video_name is not None:
            if self.__check_video(video_name):
                self.video_name = video_name
                self.__grab_video_details()
                self.matrix_name = f'{self.tag}_{self.place}_{self.feature}_{self.scale}_{self.start_second}_{self.end_second}.npy'

                if self.__check_matrix(matrix_name):
                    self.matrix = np.load(matrix_name)
                else:
                    self.__construct_matrix_from_video()

                self.__perform_svd(True, 1, 1, 5)
                self.display_background()
                return

        raise IncorrectName("There aren't any materials found")

    def __check_matrix(self, matrix_name: str) -> bool:
        """
        Checks whether such matrix already exists
        :param matrix_name:
        :return:
        """
        if '.npy' not in matrix_name:
            matrix_name = matrix_name + '.npy'

        matrices = os.listdir('./video_matrix')

        if matrix_name not in matrices:
            return False

        self.matrix_name = matrix_name
        return True

    @staticmethod
    def __rgb2gray(rgb):
        """
        Converts each frame into grayscale
        :param rgb: particular frame of the video
        :return:
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def __construct_matrix_from_video(self):
        """
        Creates matrix out of the video. Takes each from and flattens it according to the output dimensions
        :return:
        """
        frames = []
        for i in range(self.fps * int(self.clip.duration)):
            frame = self.clip.get_frame(self.fps)

            gray_frame = self.__rgb2gray(frame).astype(np.uint8)

            if self.dims is None:
                self.dims = (Separator.default_dims[0] * 0.25, Separator.default_dims[1] * 0.25)

            resized_frame = cv.resize(gray_frame, self.dims)

            frames.append(resized_frame.flatten())

            self.matrix = np.vstack(frames).T
            np.save(f'./video_matrix/{self.matrix_name}', self.matrix)

    def __check_video(self, video_name: str) -> bool:
        """
        Checks whether such video exists
        :param video_name:
        :return:
        """
        videos = os.listdir('./video')
        if video_name not in videos:
            return False
        return True

    def __perform_svd(self, rsvd: bool = False, r: int = None, q: int = None, p: int = None):
        """
        Performs SVD. Decides whether to perform Full SVD or Randomized SVD
        :param rsvd: responsible for decision on Randomized of Full SVD
        :param r: desirable rank for Randomized SVD
        :param q: number of power iterations for Randomized SVD
        :param p: oversampling parameter for Randomized SVD
        :return:
        """
        svd_results = os.listdir('./svd_results')
        if f'U_{self.matrix_name}' in svd_results and f'V_{self.matrix_name}' in svd_results and f's_{self.matrix_name}' in svd_results:
            self.U = np.load(f'./svd_results/U_{self.matrix_name}')
            self.s = np.load(f'./svd_results/s_{self.matrix_name}')
            self.V = np.load(f'./svd_results/V_{self.matrix_name}')

        if rsvd:
            self.U, self.s, self.V = self.__randomized_svd(r, q, p)
        else:
            self.U, self.s, self.V = np.linalg.svd(self.matrix, full_matrices=False)

        np.save(f'./svd_results/U_{self.matrix_name}', self.U)
        np.save(f'./svd_results/s_{self.matrix_name}', self.s)
        np.save(f'./svd_results/V_{self.matrix_name}', self.V)
        self.__low_rank_approx(self, 0)

    def __low_rank_approx(self, level):
        """
        Provides low-rank approximation of the matrix
        :param level: Responsible for the desired rank of the matrix approximation
        :return:
        """
        if level > self.U.shape[0]:
            level = self.U.shape[0]
        self.low_rank = (self.U[:, :level + 1].reshape(self.U.shape[0], level + 1) @
                         np.diag(self.s[:level + 1]) @
                         self.V[:level + 1, :].reshape(level + 1, self.V.shape[0]))

    def display_background(self):
        """
        Displays the background of the video which is a static part
        :return:
        """
        plt.imshow(self.low_rank[:, 0].reshape(self.dims), cmap='gray')
        plt.show()

    def display_foreground(self, frame):
        """
        Displays foreground of particular frame of the video
        :param frame: particular frame of the video
        :return:
        """
        if frame > self.matrix.shape[1] or frame < self.matrix.shape[1]:
            raise IncorrectFrame('There is no such frame')
        plt.imshow(np.reshape(self.matrix[:, frame] - self.low_rank[:, 0], self.dims), cmap='gray')
        plt.show()

    def create_video_without_background(self):
        """
        Creates video without background. Also checks whether such video already exists
        :return:
        """
        if self.__check_video_nobg():
            return
        mat_reshaped = np.reshape(self.matrix, (self.dims[0], self.dims[1], -1))

        fig, ax = plt.subplots()

        def make_frame(t):
            ax.clear()
            ax.imshow(mat_reshaped[..., int(t * self.fps)])
            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=int(self.clip.duration))
        animation.write_videofile(f'./video_without_background/{self.matrix_name}.mp4', fps=self.fps)

    def __check_video_nobg(self):
        """
        Checks if the without background appropriate to specific matrix has ever been created
        :return: The result as boolean type
        """
        video_no_bg = os.listdir('./video_without_background')
        if self.matrix_name not in video_no_bg:
            return False
        return True

    def __randomized_svd(self, r, q, p):
        """
        Compute the randomized SVD of the video matrix
        :param r: desirable rank
        :param q: number of power iterations
        :param p: oversampling parameter
        :return: matrices obtained by the randomized SVD
        """

        ny = self.matrix.shape[1]
        P = np.random.randn(ny, r + p)
        Z = self.matrix @ P
        for k in range(q):
            Z = self.matrix @ (self.matrix.T @ Z)

        Q, R = np.linalg.qr(Z, mode='reduced')

        Y = Q.T @ self.matrix
        UY, S, V = np.linalg.svd(Y, full_matrices=False)
        U = Q @ UY

        return U, S, V
