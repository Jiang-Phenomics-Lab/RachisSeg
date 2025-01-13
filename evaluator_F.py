from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset
)
from detectron2.modeling import build_model

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)
#register the datasets
register_coco_instances('AA508_train', {}, '/home/ubuntu/picture/AA508_TRAIN/new_dataset/splited_dataset_pad/annotations/train_annotations.json', '/home/ubuntu/picture/AA508_TRAIN/new_dataset/splited_dataset_pad/train')
register_coco_instances('AA508_val', {}, '/home/ubuntu/picture/AA508_TRAIN/new_dataset/splited_dataset_pad/annotations/val_annotations.json', '/home/ubuntu/picture/AA508_TRAIN/new_dataset/splited_dataset_pad/val')
register_coco_instances('RIL_test', {}, '/home/ubuntu/picture/AA508_TRAIN/new_dataset/splited_dataset_pad/annotations/test_annotations.json', '/home/ubuntu/picture/AA508_TRAIN/new_dataset/splited_dataset_pad/test')
 
cfg = get_cfg()
base_cfg_path = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
cfg.merge_from_file(model_zoo.get_config_file(base_cfg_path))
out_dir='/home/ubuntu/picture/AA508_TRAIN/evaluator'
cfg.MODEL.WEIGHTS = '/home/ubuntu/lrx/Rachis/DL_2/getPanincleTraits/img_in_0314/natural_pop/models_new_2/model_final.pth
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MIN_SIZE_TEST: ()     
cfg.INPUT.MAX_SIZE_TEST: 99999
cfg.OUTPUT_DIR = out_dir
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 
# cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.1 
predictor = DefaultPredictor(cfg)
# model = build_model(cfg)

train_loader = build_detection_test_loader(cfg, 'AA508_train')
val_loader=build_detection_test_loader(cfg, 'AA508_val')
test_loader=build_detection_test_loader(cfg, 'RIL_test')

evaluator_train = COCOEvaluator('AA508_train', cfg, True, output_dir=out_dir)
evaluator_val = COCOEvaluator('AA508_val', cfg, True, output_dir=out_dir)
evaluator_test= COCOEvaluator('RIL_test', cfg, True, output_dir=out_dir)

results_1 = inference_on_dataset(predictor.model, train_loader, evaluator_train)
results_2 = inference_on_dataset(predictor.model, val_loader, evaluator_val)
results_3 = inference_on_dataset(predictor.model, test_loader, evaluator_test)

