import pdb

import cv2
import mmcv
import torch
from groundingdino.util.inference import load_model, predict, annotate, Model
import groundingdino.datasets.transforms as T
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from mmdet.apis.inference import DetDataSample
import numpy as np
from PIL import Image

import os
PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while not os.path.exists(os.path.join(PATH, "../envs")):
    PATH = os.path.dirname(PATH)

CONFIG_PATH = "/data/share/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "/data/share/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
DEVICE = "cuda"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

def load_image(image_array):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_array = np.asarray(image_array[:, :, ::-1])
    image = Image.fromarray(np.uint8(image_array))
    image_transformed, _ = transform(image, None)
    return image_array, image_transformed

class PredInstances:
    def __init__(self, labels, masks, scores, bboxes):
        self.labels = labels
        self.masks = masks
        self.scores = scores
        self.bboxes = bboxes

class SegmentReturn:
    def __init__(self, pred_instances: PredInstances):
        self.pred_instances = pred_instances

class DetectorSAM:
    def __init__(self,
                 category_file=os.path.join(PATH, '../data', 'meta_data', 'categories_new.txt'),
                 **kwargs):
        self.model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
        self.categories = []
        with open(category_file, 'r') as f:
            self.categories = eval(f.read())
        self.num_categories = len(self.categories)
        self.targets = []

    def set_targets(self, targets):
        self.targets = targets

    def phrase_to_id(self, phrase):
        if phrase in self.categories:
            return self.categories.index(phrase)
        for i, cate in enumerate(self.categories):
            if phrase in cate:
                return i
        return 0

    def inference(self, img):
        image_source, image = load_image(img)
        TEXT_PROMPT = " ".join([f"{target}." for target in self.targets])

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE,
        )

        labels = []
        masks = []
        scores = []
        bboxes = []
        img_size = img.shape[0]
        for box, phrase, logit in zip(boxes, phrases, logits):
            x_min = box[0] - box[2] / 2
            x_max = box[0] + box[2] / 2
            y_min = box[1] - box[3] / 2
            y_max = box[1] - box[3] / 2
            labels.append(self.phrase_to_id(phrase))
            mask = torch.zeros((img_size, img_size), dtype=torch.bool)
            mask[int(x_min*img_size):int(x_max*img_size), int(y_min*img_size):int(y_max*img_size)] = True
            masks.append(mask)
            scores.append(logit)
            bboxes.append([x_min*img_size, y_min*img_size, x_max*img_size, y_max*img_size])
        if len(boxes) == 0:
            return None
        pred_instance = PredInstances(labels=torch.LongTensor(labels),
                                      masks=torch.stack(masks),
                                      scores=torch.FloatTensor(scores),
                                      bboxes=torch.FloatTensor(bboxes))
        return SegmentReturn(pred_instances=pred_instance)

if __name__ == "__main__":
    d = {"a": 1, "b": 2}
    detector = DetectorSAM(**d)
    img = '~/segment/GroundingDINO/test.png'
    img = mmcv.imread(img)
    result = detector.inference(img)
    print(result.pred_instances)
