import json
import base64
import io
from PIL import Image

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import convert_PIL_to_numpy

import numpy as np
from pycocotools import mask
from skimage import measure

import cv2

def init_context(context):
    context.logger.info("Init context...  0%")

    cfg = get_cfg()
    cfg.merge_from_file("config_20.yaml")
    cfg.MODEL.WEIGHTS = "model_final_20.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATALOADER.NUM_WORKERS = 0 # Wyłączenie gpu
    cfg.MODEL.DEVICE = 'cpu' # Wyłączenie gpu
    predictor = DefaultPredictor(cfg)

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run detectro-maskrcnn-R101 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.7))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    predictions = context.user_data.model_handler(image)

    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    pred_mask = instances.pred_masks
    scores = instances.scores
    pred_classes = instances.pred_classes
    results = []
    
    binary_masks = pred_mask.numpy().astype(np.uint8)
    
    polygon = []

    def reduce_contour(contour, epsilon):
        contour = np.array(contour).astype(np.int32)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        approx_contour = approx_contour.squeeze().tolist()

        return approx_contour

    for binary_mask in binary_masks:
        fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)

        contours, hierarchy = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            simplified_contour = reduce_contour(contour, epsilon=2.0)
            polygon.append(simplified_contour)
            
    new_arr = []

    for sublist in polygon:
        if len(sublist) > 5:
            for cords in sublist:
                new_arr.append(cords[0])
                new_arr.append(cords[1])
                
    print(new_arr)
    
    for mask_points, score, label in zip(new_arr, scores, pred_classes):
            results.append({
                "confidence": str(float(score)),
                "label": "tulip",
                "points": mask_points,
                "type": "polygon",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
