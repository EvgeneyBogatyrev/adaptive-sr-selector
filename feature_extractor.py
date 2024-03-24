import os
import cv2
import numpy as np
import skimage
import subprocess
import face_recognition
from tqdm import tqdm
import torchvision.transforms as transforms
from pathlib import Path
from utils import get_bitrate, convert_video_to_frames, remove_tmp_folder, split_video_to_scenes

from erqa_modified import ERQA
from modified_lpips import LPIPS
#from PerceptualSimilarity.lpips.lpips import LPIPS

loss_fn_alex = None
erqa_metric = None

class FeatureExtractor:
    '''
    Extracts features from given images or videos
    '''
    def __init__(self, features="all", frame_step=1, tmp_dir="/tmp", split_to_scenes=True):
        self.name2feature = {
            "gabor" : gabor,
            "sobel" : sobel,
            "lbp" : lbp,
            "haff" : haff,
            "fft" : fft,
            "laplac" : laplac,
            "colorfulness" : colorfulness,
            "SI" : SI,
            "TI" : TI(),
            "erqa" : erqa,
            "lpips0" : get_lpips(0),
            "lpips1" : get_lpips(1),
            "lpips2" : get_lpips(2),
            "lpips3" : get_lpips(3),
            "lpips4" : get_lpips(4),
            "face_count" : face_count,
        }

        if features == "all":
            self.features = list(self.name2feature.keys())
        else:
            self.features = features
    
        self.tmp_path = tmp_dir
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)

        self.frame_step = frame_step
        self.split_to_scenes = split_to_scenes


    def __call__(self, video_path):
        bitrate = get_bitrate(video_path, self.tmp_path)
        if self.split_to_scenes:
            scene_ends = split_video_to_scenes(video_path)
        else:
            scene_ends = None
        image_lists = convert_video_to_frames(video_path, tmp_path=self.tmp_path, frame_step=self.frame_step, scene_ends=scene_ends)
        overall_result = []

        for images in image_lists:
            result = self.process_image_list(images)
            result["bitrate"] = bitrate
            overall_result.append(result)

        remove_tmp_folder(self.tmp_path)
        return overall_result


    def process_image_list(self, image_list):
        result = []
        for img in tqdm(image_list):
            result.append(self.run_on_frame(img))
        
        return self.aggregate(result)


    def run_on_frame(self, img):
        values = {}
        for feature in self.features:
            values[feature] = self.name2feature[feature](img)
        return values
    

    def reinit(self):
        self.name2feature["TI"] = TI()


    def transform(self, feat_dict):
        new_dict = {}
        for key in feat_dict:
            value = feat_dict[key]
            if type(value) is list:
                for i, elem in enumerate(value):
                    new_dict[key + "_" + str(i)] = elem
            else:
                new_dict[key] = value
        return new_dict
    

    def aggregate(self, feat_list):
        result = {}
        aggr = {}

        for feat_frame in feat_list:
            feat_frame = self.transform(feat_frame)
            for key in feat_frame:
                if key not in aggr:
                    aggr[key] = []
                aggr[key].append(feat_frame[key])
        
        for key in aggr:
            result[key] = {
                "mean" : np.mean(aggr[key]),
                "min" : min(aggr[key]),
                "max" : max(aggr[key])
            }
        return result


def get_lpips(layer_idx):
    assert layer_idx < 5
    def lpips_on_layer(img1):
        global loss_fn_alex
        if loss_fn_alex is None:
            loss_fn_alex = LPIPS(net='alex',verbose=False)
        transform = transforms.ToTensor()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = transform(img1)
        result = loss_fn_alex(img1)#.detach().numpy()
        layer = result[layer_idx].detach().numpy()
        return np.linalg.norm(layer)
    
    return lpips_on_layer

def gabor(image):
    frequency = 0.15
    sigma = 3.5
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 128))
    real, _ = skimage.filters.gabor(
        image, frequency=frequency, theta=np.pi / 3, 
        sigma_x=sigma, sigma_y=sigma, mode="wrap"
    )
    return np.linalg.norm(np.array(cv2.meanStdDev(real)))


def sobel(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)

    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)

    return np.linalg.norm(grad)


def sobel_filter(image):
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)

    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)

    return grad


def lbp(image):
    edges = np.rint(sobel_filter(image)).astype(np.uint8)
    grayscale = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    patterns = skimage.feature.local_binary_pattern(grayscale, P=4, R=8, method='uniform')

    return np.linalg.norm(patterns)


def haff(img):
    edges = cv2.Canny(img, 150, 255)
    lines = cv2.HoughLinesP(edges, 200, np.pi / 3, 150, None, 0, 0)
    image = np.zeros_like(img)
    if lines is not None:
        for line_tuple in lines:
            line = line_tuple[0]
            cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), thickness=5)
    return np.linalg.norm(image)


def fft(image):
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 128))
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    size = 35
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = np.log(np.abs(recon))
    return np.linalg.norm(magnitude)


def laplac(image):
    image = cv2.resize(image, (128, 128))
    return np.linalg.norm(cv2.Laplacian(image, cv2.CV_64F, ksize=3))


def colorfulness(im): 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    (B, G, R) = cv2.split(im.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)

    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    return stdRoot + (0.3 * meanRoot)


def SI(frame): 
    grad_x = cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    grad_y = cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)
    value = np.hypot(grad_x, grad_y).std()
    return value


class TI: 
    def __init__(self):
        self._previous_frame = None

    def __call__(self, frame):
        value = 0
        if self._previous_frame is not None:
            value = (frame - self._previous_frame).std()
        self._previous_frame = frame
        return value


def erqa(im):
    global erqa_metric
    if erqa_metric is None:
        erqa_metric = ERQA()
    try:
        return erqa_metric(im)
    except:
        return np.nan
    

def face_count(image):
    face_locations = face_recognition.face_locations(image)
    return len(face_locations)
