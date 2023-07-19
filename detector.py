import os 
os.chdir(r"C:\Users\amb\Downloads\FasterRcnn\water-body-json\tree-json\data\10-img-test\detecrton\detectron2")

import numpy as np
import json
import random
import matplotlib.pyplot as plt
import torch
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, Metadata
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
device = "cuda" if torch.cuda.is_available() else "cpu"
device
import glob

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib.patches import Rectangle
import cv2  

import torch
assert torch.__version__.startswith("1.8") 
import torchvision
import cv2
     
torch.__version__





cfg = get_cfg() 
cfg = get_cfg() 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")) 
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001 
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.STEPS = [500] 
cfg.TEST.EVAL_PERIOD = 200 

cfg.SOLVER.MAX_ITER = 2000 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.58 
predictor = DefaultPredictor(cfg).to("cpu")

for imageName in glob.glob('a.jpg'):
    im = cv2.imread('a.jpg')
    print(im)


for imageName in glob.glob('a.jpg'):
  im = cv2.imread(imageName)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8
                 )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2.imshow(out.get_image()[:, :, ::-1])
  
  key = cv2.waitKey(0)
  if key == ord('q'):
     break
  cap.release()
cv2.destroyAllWindows()
  
  
#torch.cuda.is_available()  
