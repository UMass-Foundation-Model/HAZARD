import pdb

import cv2
import mmcv
import torch

from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from mmdet.apis.inference import DetDataSample

import os
PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while not os.path.exists(os.path.join(PATH, "../envs")):
    PATH = os.path.dirname(PATH)


# checkpoint can be downloaded from https://drive.google.com/file/d/15KP4EWoQ_8EsuWFpOGNJLeqOFlEu1hFT/view?usp=sharing
class Detector:
    def __init__(self,
                 category_file=os.path.join(PATH, '../data', 'meta_data', 'categories.txt'),
                 config_file=os.path.join(PATH, '', 'vision', 'mask-rcnn_r50-caffe_fpn_ms-1x_tdw.py'),
                 checkpoint_file=os.path.join(PATH, '../data', 'tdw_rcnn.pth'),
                 device=None,
                 **kwargs):
        if device == None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device = str(device)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.categories = []
        with open(category_file, 'r') as f:
            self.categories = eval(f.read())
        self.num_categories = len(self.categories)

    def data_to_array(self, data: DetDataSample):
        labels = data.pred_instances.labels
        masks = data.pred_instances.masks
        scores = data.pred_instances.scores
        
        sem = torch.zeros((self.num_categories, ) + masks.shape[1:], device=masks.device)
        for i, label in enumerate(labels):
            sem[label] = torch.max(sem[label], masks[i] * scores[i])
        return sem

    def inference(self, img):
        # img = mmcv.imread(img)
        img = img[:, :, ::-1]
        result = inference_detector(self.model, img)
        return result

if __name__ == "__main__":
    d = {"a": 1, "b": 2}
    detector = Detector(**d)
    img = 'outputs/flood_LLM4/0/rgb_43.png'
    img = mmcv.imread(img)
    result = detector.inference(img)
    print(result.pred_instances)
