from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.solver import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from detectron2 import model_zoo
import numpy as np
import os
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.transforms import Augmentation,Transform
import cv2
import numpy as np
logger = logging.getLogger("detectron2")

class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2).astype(np.float32)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)
    
class MyCustomColorTransform(Augmentation):
    def __init__(self, hue_factor_range=(-0.5, 0.5), **kwargs):
        super().__init__(**kwargs)
        self.hue_factor_range = hue_factor_range

    def get_transform(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue_factor = np.random.uniform(low=self.hue_factor_range[0], high=self.hue_factor_range[1])
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_factor * 360) % 180
        transformed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return MyTransform(transformed_image)
    
class MyTransform(Transform):
    def __init__(self, image):
        self.image = image

    def apply_image(self, img):
        return self.image

    def apply_coords(self, coords):
        return coords

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
global max_mAP
max_mAP=0

def do_test(cfg, model):
    global max_mAP
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
        if max_mAP<results_i['bbox']['AP50']:
            
            max_mAP=results_i['bbox']['AP50']
            # torch.save(model,out_dir+'/best.pth')
            torch.save(model.state_dict(),out_dir+'/best_weights.pth')
            print(results_i['bbox']['AP50'])
            print('saved_models')

    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=True):
    model.train()
    
    optimizer = build_optimizer(cfg, model)
    
    # factor=0.1
    # step_size=20000
    scheduler = build_lr_scheduler(cfg, optimizer)
    # scheduler = LambdaLR(optimizer, lambda iteration: 1 - (iteration / max_iter) \
    #                      if iteration <= step_size else factor * (iteration - step_size) / (max_iter - step_size))
    
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    #data_loader = build_detection_train_loader(cfg)
    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
        #T.ResizeFractionWidth(3,1024),
        # T.Resize((512, 1024)),
        # T.ResizeShortestEdge(128,1024),
        # T.ResizeFixedWidth,
        # T.ResizeOneThirdWidth(2048),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
        T.RandomLighting(random.random() + 0.5),
        MyCustomColorTransform(),
        T.RandomRotation((-45,45)),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    ]))
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                
                do_test(cfg, model)

                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            # if (
            # iteration != max_iter - 1
            # ):
            #     do_test(cfg, model)
            #     # Compared to "train_net.py", the test results are not dumped to EventStorage
            #     comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

out_dir = '/public/home/rxlu/DL/model/train_output'
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    base_cfg_path = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    cfg.merge_from_file(model_zoo.get_config_file(base_cfg_path))
    cfg.DATASETS.TRAIN = ("rachis_train")
    cfg.DATASETS.TEST = ["rachis_val"]
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_cfg_path)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 30000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = out_dir
    cfg.TEST.EVAL_PERIOD=300
    cfg.INPUT.MIN_SIZE_TRAIN: ()      
    cfg.INPUT.MAX_SIZE_TRAIN: 99999
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[16],[32],[64],[128],[256]]
    cfg.SOLVER.OPTIMIZER='ADAM'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    '''
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(Z
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    '''
    return cfg


def main(args):
    register_coco_instances('rachis_train', {}, '/public/home/rxlu/DL/rachis_train_1220_coco.json', '/public/home/rxlu/DL//rachis_train_1212')
    register_coco_instances('rachis_val', {}, '/public/home/rxlu/DL/rachis_val_1220_coco.json', '/public/home/rxlu/DL//rachis_val')
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        ) 
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return# do_test(cfg, model)

if __name__ == "__main__":
    #main()
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    import tensorboard
