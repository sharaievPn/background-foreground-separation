import numpy as np
import os
from moviepy.editor import VideoFileClip, VideoClip
import cv2 as cv
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import json


class NameErrorException(Exception):
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
    """

    def __init__(self, video_name: str = None) -> None:
        """

        :param video_name:
        :param matrix_name:
        """
        self.__video_name = video_name
        self.__matrix_name = None
        self.__clip = None
        self.__fps = None
        self.__start_second = None
        self.__end_second = None
        self.__scale = None
        self.__tag = None
        self.__place = None
        self.__feature = None
        self.__matrix = None
        self.__U = None
        self.__s = None
        self.__V = None
        self.__low_rank = None
        self.__height = None
        self.__width = None
        self.__duration = None

    def __grab_video_details(self):
        """
        Gather essential characteristics of the video
        :return:
        """
        if self.__video_name is None:
            raise NameErrorException("The video was not specified")

        print('Specify video characteristics: ')
        start_second = int(input('Start second: '))
        end_second = int(input('End second: '))

        if start_second < 0 or end_second < 0:
            raise IncorrectDuration("Time must be positive")

        if end_second < start_second:
            raise IncorrectDuration("Starting second must be lower than ending second")

        video = VideoFileClip(f'./video/{self.__video_name}')
        vid = cv.VideoCapture(f'./video/{self.__video_name}')
        self.__clip = video
        if end_second > video.duration:
            end_second = int(video.duration)

        self.__clip = self.__clip.subclip(start_second, end_second)
        self.__duration = self.__clip.duration

        self.__start_second = start_second
        self.__end_second = end_second
        self.__scale = int(input('Scale: '))
        self.__tag = input('Tag: ')
        self.__place = input('Place: ')
        self.__feature = input('Feature: ')

        self.__fps = int(self.__clip.fps)
        self.__width = int(int(vid.get(cv.CAP_PROP_FRAME_WIDTH)) * self.__scale / 100)
        self.__height = int(int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)) * self.__scale / 100)
        self.__matrix_name = f'{self.__tag}_{self.__place}_{self.__feature}_{self.__scale}_{self.__start_second}_{self.__end_second}_{self.__width}_{self.__height}.npy'
        data = dict()
        data['video_name'] = self.__video_name
        data['matrix_name'] = self.__matrix_name
        data['start_second'] = self.__start_second
        data['end_second'] = self.__end_second
        data['scale'] = self.__scale
        data['tag'] = self.__tag
        data['place'] = self.__place
        data['feature'] = self.__feature
        data['fps'] = self.__fps
        data['width'] = self.__width
        data['height'] = self.__height
        data['duration'] = self.__duration
        with open(f'./matrices_data/{self.__matrix_name[:-4]}.json', 'w') as file:
            json.dump(data, file, indent=4)
        print('Video details grabbed...')

    def load_json(self, json_file: str):
        if '.json' not in json_file:
            json_file = json_file + '.json'
        data = json.load(open(f'./matrices_data/{json_file}', 'r'))
        self.__video_name = data['video_name']
        self.__matrix_name = data['matrix_name']
        self.__start_second = data['start_second']
        self.__end_second = data['end_second']
        self.__scale = data['scale']
        self.__tag = data['tag']
        self.__place = data['place']
        self.__feature = data['feature']
        self.__fps = data['fps']
        self.__width = data['width']
        self.__height = data['height']
        self.__duration = data['duration']

    def open(self, video_name: str = None):
        """
        Visualises background according to the provided video or matrix name. Checks whether such video or matrix exists
        :param video_name: name of the video
        :return:
        """
        if self.__matrix_name is None and video_name is None:
            raise NameErrorException("There aren't any materials provided")

        if self.__matrix_name is not None:
            if self.__check_matrix(self.__matrix_name):
                self.__matrix = np.load('./video_matrix/' + self.__matrix_name)
                self.__perform_svd(True, 1, 2, 10)
                return

        if video_name is not None:
            if self.__check_video(video_name):
                self.__video_name = video_name
                self.__grab_video_details()

                if self.__check_matrix(self.__matrix_name):
                    self.__matrix = np.load('./video_matrix/' + self.__matrix_name)
                else:
                    self.__construct_matrix_from_video()

                self.__perform_svd(True, 1, 2, 10)
                return

        raise NameErrorException("There aren't any materials found")

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

        self.__matrix_name = matrix_name
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
        Creates matrix out of the video. Takes each frame and flattens it according to the output dimensions
        :return:
        """
        print('Started matrix construction...')
        frames = []
        duration = int(self.__clip.duration)
        for i in range(self.__fps * duration):
            frame = self.__clip.get_frame(i / float(self.__fps))
            frame = self.__rgb2gray(frame)
            frame = cv.resize(frame, (self.__width, self.__height))
            frame = frame.astype(np.uint8)
            frames.append(frame.flatten())
        self.__matrix = np.vstack(frames).T
        np.save(f'./video_matrix/{self.__matrix_name[:-4]}', self.__matrix)
        print('Matrix constructed...')

    @staticmethod
    def __check_video(video_name: str) -> bool:
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
        print('Started SVD application...')
        svd_results = os.listdir('./svd_results')
        if rsvd:
            if f'U_rsvd_{self.__matrix_name}' in svd_results and f'V_rsvd_{self.__matrix_name}' in svd_results and f's_rsvd_{self.__matrix_name}' in svd_results:
                self.__U = np.load(f'./svd_results/U_rsvd_{self.__matrix_name}')
                self.__s = np.load(f'./svd_results/s_rsvd_{self.__matrix_name}')
                self.__V = np.load(f'./svd_results/V_rsvd_{self.__matrix_name}')
            else:
                self.__U, self.__s, self.__V = self.__randomized_svd(r, q, p)
                np.save(f'./svd_results/U_rsvd_{self.__matrix_name}', self.__U)
                np.save(f'./svd_results/s_rsvd_{self.__matrix_name}', self.__s)
                np.save(f'./svd_results/V_rsvd_{self.__matrix_name}', self.__V)
        else:
            if f'U_fsvd_{self.__matrix_name}' in svd_results and f'V_fsvd_{self.__matrix_name}' in svd_results and f's_fsvd_{self.__matrix_name}' in svd_results:
                self.__U = np.load(f'./svd_results/U_fsvd_{self.__matrix_name}')
                self.__s = np.load(f'./svd_results/s_fsvd_{self.__matrix_name}')
                self.__V = np.load(f'./svd_results/V_fsvd_{self.__matrix_name}')
            else:
                self.__U, self.__s, self.__V = np.linalg.svd(self.__matrix, full_matrices=False)
                np.save(f'./svd_results/U_fsvd_{self.__matrix_name}', self.__U)
                np.save(f'./svd_results/s_fsvd_{self.__matrix_name}', self.__s)
                np.save(f'./svd_results/V_fsvd_{self.__matrix_name}', self.__V)

        self.__low_rank_approx(0)
        print('SVD performed...')

    def display_background(self):
        """
        Displays the background of the video which is a static part
        :return:
        """
        if self.__video_name is None and self.__matrix_name is None:
            raise NameErrorException('Please provide materials')

        if self.__video_name is None and self.__matrix_name is not None or self.__matrix_name is not None and self.__video_name is not None:
            self.open()

        if self.__matrix_name is None and self.__video_name is not None:
            self.open(video_name=self.__video_name)

        plt.imshow(self.__low_rank.mean(1).reshape(self.__height, self.__width), cmap='gray')
        plt.axis('off')
        plt.gca().set_facecolor('none')
        plt.gcf().patch.set_facecolor('none')
        plt.gcf().patch.set_alpha(0)
        plt.savefig(f'{self.__matrix_name[:-4]}_background.png')
        plt.show()

    def display_foreground(self, second: int):
        """
        Displays foreground of particular frame of the video
        :param frame: particular frame of the video
        :return:
        """
        if self.__video_name is None and self.__matrix_name is None:
            raise NameErrorException('Please provide materials')

        if self.__video_name is None and self.__matrix_name is not None or self.__matrix_name is not None and self.__video_name is not None:
            self.open()

        if self.__matrix_name is None and self.__video_name is not None:
            self.open(video_name=self.__video_name)

        if second > self.__duration:
            second = self.__duration
        if second < 0:
            second = 0

        frame = second * self.__fps
        if frame > self.__matrix.shape[1]:
            frame = self.__matrix.shape[1] - 1
        if frame < 0:
            frame = 0

        plt.imshow(np.reshape(self.__matrix[:, frame], (self.__height, self.__width)), cmap='gray')
        plt.axis('off')
        plt.gca().set_facecolor('none')
        plt.gcf().patch.set_facecolor('none')
        plt.gcf().patch.set_alpha(0)
        plt.savefig(f'{self.__matrix_name[:-4]}_second.png')
        plt.show()

        plt.imshow(np.reshape(self.__matrix[:, frame] - self.__low_rank[:, frame], (self.__height, self.__width)),
                   cmap='gray')
        plt.axis('off')
        plt.gca().set_facecolor('none')
        plt.gcf().patch.set_facecolor('none')
        plt.gcf().patch.set_alpha(0)
        plt.savefig(f'{self.__matrix_name[:-4]}_foreground.png')
        plt.show()

    def create_video_without_background(self):
        """
        Creates video without background. Also checks whether such video already exists
        :return:
        """
        if self.__video_name is None and self.__matrix_name is None:
            raise NameErrorException('Please provide materials')

        if self.__video_name is None and self.__matrix_name is not None or self.__matrix_name is not None and self.__video_name is not None:
            self.open()

        if self.__matrix_name is None and self.__video_name is not None:
            self.open(video_name=self.__video_name)

        if self.__check_video_nobg():
            return

        if self.__clip is None:
            self.__clip = VideoFileClip(f'./video/{self.__video_name}').subclip(self.__start_second, self.__end_second)

        mat_reshaped = np.reshape(self.__matrix - self.__low_rank, (self.__height, self.__width, -1))

        fig, ax = plt.subplots()

        def make_frame(t):
            ax.clear()
            ax.imshow(mat_reshaped[..., int(t * self.__fps)], cmap='gray')
            ax.axis('off')
            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=int(self.__clip.duration))
        animation.write_videofile(f'./video_without_background/{self.__matrix_name[:-4]}' + '.mp4', fps=self.__fps)

    def __check_video_nobg(self):
        """
        Checks if the without background appropriate to specific matrix has ever been created
        :return: The result as boolean type
        """
        video_no_bg = os.listdir('./video_without_background')
        if f'{self.__matrix_name[:-4]}.mp4' not in video_no_bg:
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

        ny = self.__matrix.shape[1]
        P = np.random.randn(ny, r + p)

        for i in range(len(P[0, :])):
            P[:, i] = P[:, i] / np.linalg.norm(P[:, i])

        Y = self.__matrix @ P
        for k in range(q):
            Y = self.__matrix @ (self.__matrix.T @ Y)

        Q, R = np.linalg.qr(Y, mode='reduced')

        B = Q.T @ self.__matrix
        UY, S, V = np.linalg.svd(B, full_matrices=False)
        U = Q @ UY

        return U, S, V

    def __low_rank_approx(self, level: int = 0):
        """
        Provides low-rank approximation of the matrix
        :param level: Responsible for the desired rank of the matrix approximation
        :return:
        """
        if level > self.__U.shape[0]:
            level = self.__U.shape[0]
        self.__low_rank = (self.__U[:, :level + 1].reshape(self.__U.shape[0], level + 1) @
                           np.diag(self.__s[:level + 1]) @
                           self.__V[:level + 1, :].reshape(level + 1, self.__V.shape[1]))

    @property
    def video_name(self):
        return self.__video_name

    @property
    def matrix_name(self):
        return self.__matrix_name

    @property
    def clip(self):
        return self.__clip

    @property
    def fps(self):
        return self.__fps

    @property
    def start_second(self):
        return self.__start_second

    @property
    def end_second(self):
        return self.__end_second

    @property
    def scale(self):
        return self.__scale

    @property
    def tag(self):
        return self.__tag

    @property
    def place(self):
        return self.__place

    @property
    def feature(self):
        return self.__feature

    @property
    def matrix(self):
        return self.__matrix

    @property
    def U(self):
        return self.__U

    @property
    def s(self):
        return self.__s

    @property
    def V(self):
        return self.__V

    @property
    def low_rank(self):
        return self.__low_rank

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width
