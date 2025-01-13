import json

def modify_coco_annotations(annotations_file, output_file, dpi_threshold=1232):
    try:
        # 打开COCO格式标注文件
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # 遍历标注文件中的图像条目
        for image_info in coco_data['images']:
            # 提取图像编号
            img_id=image_info["id"]
            img_name = int(image_info['file_name'][2:6])

            # 判断图像编号是否大于1232
            if img_name > dpi_threshold:
                # 更新图像信息（宽高缩小一半）
                image_info['width'] //= 2
                image_info['height'] //= 2

                # 遍历标注文件中的注释条目
                for annotation in coco_data['annotations']:
                    if annotation['image_id'] == img_id:
                        # 更新bounding box
                        bbox = annotation['bbox']
                        bbox[0] //= 2  # x坐标
                        bbox[1] //= 2  # y坐标
                        bbox[2] //= 2  # 宽度
                        bbox[3] //= 2  # 高度

                        # 更新segmentation
                        segmentation = annotation['segmentation'][0]
                        for i in range(len(segmentation)):
                            segmentation[i] //= 2

                        # 更新area
                        annotation['area'] //= 4  # 宽度和高度都缩小一半，面积缩小四分之一

        # 保存修改后的COCO格式标注文件
        with open(output_file, 'w',indent=None) as f:
            json.dump(coco_data, f)

        print("COCO格式标注文件修改成功！")
    except Exception as e:
        print(f"发生错误：{e}")

# 使用示例
annotations_file_path = "/home/ubuntu/picture/AA508_TRAIN/val_coco/AA508_val_1220_coco.json"
output_file_path = "/home/ubuntu/picture/AA508_TRAIN/val_coco/modified_AA508_val.json"

annotations_file_path = "/home/ubuntu/picture/AA508_TRAIN/train_coco/AA508_train_1220_coco.json"
output_file_path = "/home/ubuntu/picture/AA508_TRAIN/train_coco/modified_AA508_train.json"

modify_coco_annotations(annotations_file_path, output_file_path)