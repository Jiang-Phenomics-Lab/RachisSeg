import os
import json
import random
import shutil
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def split_coco_annotation(coco_json_path, image_root, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Split the COCO annotation file into training, validation, and test sets based on specified ratios,
    and generate corresponding image folders.
    Additionally, renumber the IDs in the split JSON files to ensure they are consecutive.
    
    Args:
        coco_json_path (str): Full path to the COCO annotation file.
        image_root (str): Path to the original image folder.
        output_dir (str): Directory to save the split annotation files and images.
        split_ratio (tuple): Ratios for training, validation, and test sets.
        seed (int): Random seed to ensure reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    images = coco_data['images']
    annotations = coco_data['annotations']
    image_ids = [img['id'] for img in images]
    random.seed(seed)
    random.shuffle(image_ids)
    num_images = len(image_ids)
    train_end = int(num_images * split_ratio[0])
    val_end = train_end + int(num_images * split_ratio[1])
    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])
    test_ids = set(image_ids[val_end:])
    

    def split_data(ids_set):
        imgs = [img for img in images if img['id'] in ids_set]
        annos = [anno for anno in annotations if anno['image_id'] in ids_set]
        return imgs, annos
    
    train_images, train_annotations = split_data(train_ids)
    val_images, val_annotations = split_data(val_ids)
    test_images, test_annotations = split_data(test_ids)
    
    def reindex_dataset(images, annotations):
        old_id_to_new_id = {}
        for new_id, img in enumerate(images):
            old_id = img['id']
            old_id_to_new_id[old_id] = new_id + 1 
            img['id'] = new_id + 1  
        for new_ann_id, ann in enumerate(annotations):
            ann['image_id'] = old_id_to_new_id[ann['image_id']]
            ann['id'] = new_ann_id + 1  
        return images, annotations

    train_images, train_annotations = reindex_dataset(train_images, train_annotations)
    val_images, val_annotations = reindex_dataset(val_images, val_annotations)
    test_images, test_annotations = reindex_dataset(test_images, test_annotations)

    datasets = {
        'train': {
            'images': train_images,
            'annotations': train_annotations,
            'categories': coco_data['categories']
        },
        'val': {
            'images': val_images,
            'annotations': val_annotations,
            'categories': coco_data['categories']
        },
        'test': {
            'images': test_images,
            'annotations': test_annotations,
            'categories': coco_data['categories']
        }
    }
    
    for split in ['train', 'val', 'test']:
        output_json_path = os.path.join(output_dir, f'{split}.json')
        with open(output_json_path, 'w') as f:
            json.dump(datasets[split], f)
        split_image_dir = os.path.join(output_dir, split)
        os.makedirs(split_image_dir, exist_ok=True)
        
        for img_info in datasets[split]['images']:
            src_img_path = os.path.join(image_root, img_info['file_name'])
            dst_img_path = os.path.join(split_image_dir, img_info['file_name'])
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            shutil.copy2(src_img_path, dst_img_path)
    return datasets

def register_datasets(output_dir):
    """
    Register the dataset to Detectron2.

    Args:
        output_dir (str): Directory to save the split annotation files and images.
    """

    for split in ['train', 'val', 'test']:
        dataset_name = f"my_dataset_{split}"
        json_file = os.path.join(output_dir, f'{split}.json')
        image_root = os.path.join(output_dir, split)
        register_coco_instances(dataset_name, {}, json_file, image_root)
        print(f"registered dataset：{dataset_name}")

def visualize_dataset(dataset_name, output_dir, num_samples=None):
    """
    Visualize samples from the dataset.

    Args:
        dataset_name (str): Name of the registered dataset.
        output_dir (str): Directory to save the visualization images.
        num_samples (int or None): Number of samples to visualize. If None, visualize all images.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
# If num_samples is not specified or exceeds the dataset size, visualize all images
    if num_samples is None or num_samples > len(dataset_dicts):
        num_samples = len(dataset_dicts)
        samples = dataset_dicts
    else:
        samples = random.sample(dataset_dicts, num_samples)
    
    for d in samples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = visualizer.draw_dataset_dict(d)
        img_filename = os.path.basename(d["file_name"])
        output_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
    print(f"save {num_samples} images to {output_dir}")

def main():
    coco_json_path = '//home/ubuntu/picture//coco/rachis_coco.json' 
    image_root = '/home/ubuntu/picture/cropRachis'  
    output_dir = '/home/ubuntu/picture/dataset_spilit' 
    vis_output_dir = os.path.join(output_dir, 'visualizations')
    

    split_coco_annotation(coco_json_path, image_root, output_dir)
    

    register_datasets(output_dir)

    for split in ['train', 'val', 'test']:
        dataset_name = f"my_dataset_{split}"
        split_vis_output_dir = os.path.join(vis_output_dir, split)
        visualize_dataset(dataset_name, split_vis_output_dir)
    
if __name__ == "__main__":
    main()
