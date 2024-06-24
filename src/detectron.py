from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import numpy as np
import cv2
import os
from pathlib import Path
import glob
import re
from tqdm import tqdm

class detectron_predictor():
    def __init__(self,
                 config_file='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                 weights_path='model_weights/detectron_branch_segmentation.pth',
                 num_classes=1, 
                 score_threshold=0.5):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.WEIGHTS = weights_path 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold 
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img, filter=False):        
        outputs = self.predictor(img)
        masks = outputs['instances'].get('pred_masks')
        masks = masks.to('cpu').numpy()
        scores = outputs['instances'].get('scores')
        scores = scores.to('cpu').numpy()
        if filter:
            for i in range(2): # Repeat just in case
                masks, scores = self.filter_overlapping_masks(masks, scores)
        v = Visualizer(img[:, :, ::-1], scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = v.get_image()[:, :, ::-1]
        return masks, scores, img
    
    def segment_from_folder(self, data_path):
        out_path = Path(os.path.join(data_path, 'output', 'MASKS'))
        # Check if point clouds are already in the output folder
        if out_path.exists():
            num_files = len([file for file in out_path.iterdir() if file.is_file() if file.suffix == '.npy'])
            expected_num_files = len(glob.glob(os.path.join(data_path, 'output', 'LEFT_RECT', '*')))
            if num_files == expected_num_files:
                print("Segmentations already exist in output folder")
                return
        out_path.mkdir(parents=True, exist_ok=True)

        left_rect_paths = sorted(glob.glob(os.path.join(data_path, 'output', 'LEFT_RECT', '*'), recursive=True))
        for left_rect_path in tqdm(left_rect_paths):
            left_rect = cv2.imread(left_rect_path)
            masks, scores, img = self.predict(left_rect)
            numbers = [int(s) for s in re.findall(r'-?\d+?\d*', left_rect_path)]
            cv2.imwrite(out_path / "mask_{:04d}.png".format(numbers[-1]), img)
            mask_score = {}
            mask_score['masks'] = masks
            mask_score['scores'] = scores
            np.save(out_path / "mask_score_{:04d}.npy".format(numbers[-1]), mask_score)


    @staticmethod
    def filter_overlapping_masks(masks, scores):
        new_masks = []
        new_scores = []
        n_masks = len(masks)
        joined_masks = []
        for i in range(n_masks):
            for j in range(i+1, n_masks):
                if j in joined_masks:
                    break
                num_mask_i = np.sum(masks[i])
                num_mask_j = np.sum(masks[j])
                num_mask_ij = np.sum(np.logical_and(masks[i], masks[j]))
                if num_mask_ij/num_mask_i>0.5 or num_mask_ij/num_mask_j>0.5:
                    new_mask = np.logical_or(masks[i], masks[j])
                    new_masks.append(new_mask)
                    joined_masks.append(i)
                    joined_masks.append(j)
                    new_scores.append((scores[i]+scores[j])/2)
                    continue
            if i in joined_masks:
                continue
            new_masks.append(masks[i])
            joined_masks.append(i)
            new_scores.append(scores[i])
        new_masks = np.array(new_masks)
        new_scores = np.array(new_scores)
        return new_masks, new_scores



    

