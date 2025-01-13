import os
import cv2
import copy
import numpy as np
from glob import glob
import math
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from numpy import ma, uint8
from numpy.core.numeric import ones_like
from skimage import filters
from skimage import color
from skimage.draw import rectangle
from skimage.measure import label, regionprops
import skfmm

def main():   

    in_dir='/home/ubuntu/picture/rachis_crop'
    out_dir = '/home/ubuntu/picture/rachis_crop_out'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    csv_file = out_dir + "/rachis_crop.csv"
    cfg = get_cfg()
    base_cfg_path = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    cfg.merge_from_file(model_zoo.get_config_file(base_cfg_path))
    cfg.MODEL.WEIGHTS = '/public/home/rxlu/DL/model/train_output/best_model.pth' 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.INPUT.MIN_SIZE_TEST: ()       #设置为空即不会对图像的大小再处理
    cfg.INPUT.MAX_SIZE_TEST: 99999
    predictor = DefaultPredictor(cfg)

    csv_content='Img_name,SNS, RL, PD, SD, AVB_SNS, AVB_IL, SDI, IL, IW\n'
    file = open(csv_file, 'a+')
    with open(csv_file,'w') as f:
        f.write(csv_content)

    flist = sorted(glob(in_dir +'/*.png'))
    for id in range(len(flist)):
        img = cv2.imread(flist[id])
        filename = os.path.basename(flist[id])
        gray_img = color.rgb2gray(img)
        blurred_img = filters.gaussian(gray_img, sigma=1.0)
        thre = filters.threshold_otsu(blurred_img)
        bw_img = blurred_img < thre
        bw_img= keepLargestComponent(bw_img)
        label_img = label(bw_img,connectivity=1)
        panicles = regionprops(label_img, blurred_img)
        i = 0

        for region in panicles:
            if region.area > 9000 and region.major_axis_length > 550:
                out_file_path = os.path.join(out_dir, '{}_{}'.format(filename[:-4], i))
                crop_img = img[max(region.bbox[0]-50,0):min(region.bbox[2]+50, img.shape[0]), \
                    max(region.bbox[1]-50,0):min(region.bbox[3]+50, img.shape[1])]
                crop_bw = bw_img[max(region.bbox[0]-50,0):min(region.bbox[2]+50, img.shape[0]), \
                    max(region.bbox[1]-50,0):min(region.bbox[3]+50, img.shape[1])]
                i = i+1
                bs=50 if region.bbox[0]>50 else region.bbox[0]
                outputs = predictor(crop_img)
                traits= save_test_image(filename,crop_img, crop_bw, bs,outputs, out_file_path)
                out=[]
                out.append([filename[0:-4]])
                out.append(traits)
                s = [y for x in out for y in x]
                s = str(s).replace('[','').replace(']','')
                s = s.replace("'",'') +'\n'
                file.write(s)
    return

def nms(bounding_boxes, confidence_scores, threshold):
    if len(bounding_boxes) == 0:
        return [], []
    start_x = bounding_boxes[ :,0]
    start_y = bounding_boxes[:,1]
    end_x = bounding_boxes[:,2]
    end_y = bounding_boxes[:,3]

    picked_bboxes = []
    picked_scores = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    order = np.argsort(confidence_scores)

    while order.size > 0:
        index = order[-1]

        picked_bboxes.append(bounding_boxes[index])
        picked_scores.append(confidence_scores[index])

        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        # ratio = intersection / areas[index] 
        left = np.where(ratio < threshold)
        order = order[left]
    picked_bboxes=np.array(picked_bboxes)
    picked_scores=np.array(picked_scores)

    return picked_bboxes, picked_scores

def keepLargestComponent(bw):
    labels = label(bw,connectivity=1)
    largestComponent = labels == np.argmax(np.bincount(labels.flat, weights = bw.flat))
    return largestComponent

def auc_calculation(Branch_length):
    #Branch_length is a list of branch_length or branch_width
    Add_IL=[]
    x=np.arange(1,len(Branch_length)+1)
    y=np.array(Branch_length)
    threshold=3.0
    mean_y=np.mean(y)
    std_y=np.std(y)
    outliers=np.abs(y-mean_y)>threshold*std_y
    x_filtered, y_filtered = x[~outliers], y[~outliers]

    for n in range(len(y_filtered)+1):
        add_il=np.sum(y_filtered[:n])
        Add_IL.append(add_il)

    y1=np.array(Add_IL).copy()
    y1_norm=y1/np.max(y1)
    x1=np.arange(0,len(y1_norm))
    x1_norm=x1/np.max(x1)
    auc= (np.trapz(y1_norm, x1_norm))/0.5 
    return auc


def IL(node_gdMax,j):
    IL=(node_gdMax[j]-node_gdMax[j-1])/(node_gdMax[j+1]-node_gdMax[j])
    return IL

def restore_order(bounding_boxes, scores, picked_bboxes, picked_scores):
    mask = np.zeros(len(bounding_boxes), dtype=bool)
    mask[np.isin(bounding_boxes, picked_bboxes).all(axis=1)] = True
    original_order = np.argsort(scores)[::-1]
    order = original_order[mask[original_order]]

    restored_bboxes = picked_bboxes[np.argsort(order)]
    restored_scores = picked_scores[np.argsort(order)]

    return restored_bboxes, restored_scores

def rotatecordiate(angle,rect,height,width):
    angle=angle*math.pi/180
    n=width
    m=height
    def onepoint(x,y):
        X = x * math.cos(angle) - y * math.sin(angle) - 0.5 * n * math.cos(angle) + 0.5 * m * math.sin(angle) + 0.5 * n
        Y = y * math.cos(angle) + x * math.sin(angle) - 0.5 * n * math.sin(angle) - 0.5 * m * math.cos(angle) + 0.5 * m
        return [int(X),int(Y)]
    newrect=[]
    newrect=onepoint(rect[0],rect[1])
    return newrect

def save_test_image(filename,image, bw, bs, outputs, out_file_path):
    bw_copy = copy.deepcopy(bw)
    predictions = outputs["instances"].to("cpu")
    bboxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    height=image.shape[0]
    width=image.shape[1]
    bw = keepLargestComponent(bw)

    gdt = ones_like(bw)
    gdt = ma.masked_array(gdt, bw==0)
    gdt[bs, bw[bs, :] > 0] = 0
    gd = skfmm.distance(gdt)

    node_gdMax = []
    node_gdMax1 = []
    temp_bboxes=[]
    temp_scores=[]
    final_bboxes=[]
    final_scores=[]
    sorted_bboxes=[]
    sorted_scores=[]

    bboxes = bboxes.tensor.to('cpu').numpy()
    scores = np.array(scores)
    bboxes,scores=nms(bboxes,scores,0.1)

    for bbox,cls,sco in zip(bboxes,classes,scores):
        if sco < 0.4:
            continue
        start_point = (round(bbox[0].item()), round(bbox[1].item()))
        end_point = (round(bbox[2].item()), round(bbox[3].item()))
        r, c = rectangle(np.flip(start_point), np.flip(end_point), shape = bw.shape)
        mask = np.zeros(bw.shape, dtype=bool)
        mask[r, c] = True
        flag = np.logical_and(bw, mask)
        if np.any(flag):
            node_gdMax=np.append(node_gdMax,np.max(gd[r, c]))
            temp_bboxes.append(bbox)
            temp_scores.append(sco)

    sort_idx=np.argsort(node_gdMax)
    node_gdMax = np.sort(node_gdMax)

    labeled_bw = np.zeros_like(bw_copy)
    selected_region=(gd >= 0) & (gd <=node_gdMax[0])
    labeled_bw[selected_region == 1] = 1
    labeled_region=label(labeled_bw)
    region_properties=regionprops(labeled_region)
    pd_width=region_properties[0].axis_minor_length

    for  idx in sort_idx:
        t_bboxes=temp_bboxes[idx]
        t_scores=temp_scores[idx]
        np.array(sorted_bboxes)
        np.array(sorted_scores)
        sorted_bboxes=np.append(sorted_bboxes,t_bboxes)
        sorted_scores=np.append(sorted_scores,t_scores)
    sorted_bboxes=np.reshape(sorted_bboxes,(len(sort_idx),4))
  
    top_bboxes = np.take(sorted_bboxes, np.arange(len(node_gdMax)//2), axis=0)
    top_scores = np.take(sorted_scores, np.arange(len(node_gdMax)//2), axis=0)

    bottom_bboxes = np.take(sorted_bboxes, np.arange(len(node_gdMax)//2, len(node_gdMax)), axis=0)
    bottom_scores = np.take(sorted_scores, np.arange(len(node_gdMax)//2, len(node_gdMax)), axis=0)

    top_bboxes_picked,top_scores_picked=nms(top_bboxes,top_scores,0.01)
    top_bboxes_sorted,top_scores_sorted=restore_order(top_bboxes, top_scores,  top_bboxes_picked, top_scores_picked)

    final_top_boxes,final_top_scores=filter_bboxes(top_bboxes_sorted,top_scores_sorted,pd_width,0.2,1)
    final_bottom_bboxes,final_bottom_scores=filter_bboxes_b(bottom_bboxes,bottom_scores,pd_width,0.9,0.1)

    final_bboxes= np.concatenate(( final_top_boxes,final_bottom_bboxes),axis=0)
    final_scores=np.concatenate((final_top_scores,final_bottom_scores),axis=0)

    for bbox,sco in zip(final_bboxes,final_scores):
        start_point = (round(bbox[0].item()), round(bbox[1].item()))
        end_point = (round(bbox[2].item()), round(bbox[3].item()))
        r, c = rectangle(np.flip(start_point), np.flip(end_point), shape = bw.shape)
        mask = np.zeros(bw.shape, dtype=bool)
        mask[r, c] = True
        flag = np.logical_and(bw, mask)
        if np.any(flag):
            node_gdMax1=np.append(node_gdMax1,np.max(gd[r, c]))


    r_image=np.rot90(image,k=2)       
    for f_bbox,f_sco in zip(final_bboxes,final_scores):
    # for f_bbox,f_sco in zip(sorted_bboxes,sorted_scores):
        f_start_point = (round(f_bbox[0].item()), round(f_bbox[1].item()))
        f_end_point = (round(f_bbox[2].item()), round(f_bbox[3].item()))
        f_start_point=rotatecordiate(180,f_start_point, height, width)
        f_end_point=rotatecordiate(180,f_end_point, height, width)
        rt_point=( f_start_point[0],  f_end_point[1])
        r_image = cv2.rectangle(r_image.copy(), f_start_point, f_end_point, (0,255,0), 2)  
        r_image = cv2.putText(r_image.copy(), "{}".format(round(f_sco.item(), 2)),  rt_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA) 
    cv2.imwrite(out_file_path+"_rect.png", r_image)
    SNS=len(node_gdMax1)
    ped_length=node_gdMax1[0]
    RL_g=(node_gdMax1[-1]-node_gdMax1[0])

    Branch_length,f_stage,s_stage,t_stage,length_f,length_s,length_t,AVB_IL=[],[],[],[],[],[],[],[]
    for i in range(1,len(node_gdMax1)):
        l=(node_gdMax1[i]-node_gdMax1[i-1])/118.11
        Branch_length.append(l)
    SDI=auc_calculation(Branch_length)
    x=np.arange(1,len(Branch_length)+1)
    y=np.array(Branch_length)
    threshold=3.0
    mean_y=np.mean(y)
    std_y=np.std(y)
    outliers=np.abs(y-mean_y)>threshold*std_y
    x_filtered, y_filtered = x[~outliers], y[~outliers]

    l=RL_g/3
    for num in range(0,len(node_gdMax1)):
        if node_gdMax1[num]<ped_length+l:
            f_stage.append(node_gdMax1[num])
        elif node_gdMax1[num]<ped_length+l*2:
            s_stage.append(node_gdMax1[num])
        else:
            t_stage.append(node_gdMax1[num])
    fst_num=[len(f_stage),len(s_stage),len(t_stage)]

    length = y_filtered.copy()
    for i in range(len(y_filtered)):
        if x_filtered[i]<fst_num[0]+1:
            length_f.append(length[i])
        elif x_filtered[i]<fst_num[0]+fst_num[1]+1:
            length_s.append(length[i])
        else:
            length_t.append(length[i])

    if len(length_f)!=0 and len(length_t)!=0:
        IL_f=np.mean(np.array(length_f))
        IL_t=np.mean(np.array(length_t))
        AVB_IL=IL_t/IL_f
    AVB_SNS=fst_num[2]/fst_num[0]

    bw[gd>=node_gdMax1[-1]]=0
    bw_indice = np.nonzero(bw)
    seg_im = np.zeros(bw.shape, dtype = uint8)
    gd_list = gd[bw_indice]
    seg_im[bw_indice] = np.searchsorted(node_gdMax1, gd_list) + 1  
    seg_img=seg_im.copy()
    seg_im = color.label2rgb(seg_im, bg_label = 0, bg_color = 'black')
    seg_im=np.rot90(seg_im,k=2)

    result_image=cv2.addWeighted(r_image,1,(seg_im*255).astype(np.uint8),0.4,0.1)
    seg_im=seg_im*255
    seg_im=seg_im.astype(np.uint8)
    imsave(out_file_path+"_seg.png", seg_im)
    imsave(out_file_path+"_combine.png", result_image)
    traits = []
    props = regionprops(seg_img)
    RL=RL_g/118.11
    PD=props[0].minor_axis_length/118.11
    SD=SNS/RL
    IL=np.mean(Branch_length[1:len(Branch_length)])
    width = np.array([prop.minor_axis_length for prop in props])[1:]/118.11
    IW=np.mean(width[1:len(width)])
    traits.extend([SNS, RL, PD, SD, AVB_SNS, AVB_IL, SDI, IL, IW])

    return traits

def calculate_distance(bbox1, bbox2,overlap_ratio):

    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
    top1, bottom1 = bbox1[1], bbox1[3]
    top2, bottom2 = bbox2[1], bbox2[3]
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    if overlap_ratio==1:
        shortest_ydistance=0
    elif overlap_ratio==0:
        shortest_ydistance=max(top1,top2)-min(bottom1,bottom2)
    else:
        shortest_ydistance = np.abs(center1[1]-center2[1])
    distance=distance+shortest_ydistance
    return distance

def calculate_overlap_ratio(bbox_i,bbox_j):
    overlap_y = max(0, min(bbox_i[3], bbox_j[3]) - max(bbox_i[1], bbox_j[1]))
    min_length = min(bbox_i[3] - bbox_i[1], bbox_j[3] - bbox_j[1])
    overlap_ratio = overlap_y / min_length
    return overlap_ratio

def filter_bboxes(bboxes, scores, width,threshold, dw_threshold):
    new_bboxes = []
    new_scores = []
    bbox1=bboxes[0]
    bbox2=bboxes[1]
    score1=scores[0]
    score2=scores[1]
    overlap_ratio_initial=calculate_overlap_ratio(bbox1,bbox2)
    if overlap_ratio_initial > threshold:
        if score1>score2:
            new_bboxes.append(bbox1)
            new_scores.append(score1)
        else:
            new_bboxes.append(bbox2)
            new_scores.append(score2)
    else:
        new_bboxes.append(bbox1)
        new_scores.append(score1)
        new_bboxes.append(bbox2)
        new_scores.append(score2)

    for i in range(0,len(bboxes)):
        bbox_i = bboxes[i]
        score_i = scores[i]
        keep_bbox = True
        for j in range(i + 1, len(bboxes)):
            if j==1:
                continue
            bbox_j = bboxes[j]
            overlap_ratio=calculate_overlap_ratio(bbox_i,bbox_j)
            distance = calculate_distance(bbox_i, bbox_j,overlap_ratio)
            dw=distance/width
            if dw < dw_threshold and overlap_ratio > threshold:
                keep_bbox = False
                break

        if keep_bbox:
            if i==0 or i==1:
                continue
            new_bboxes.append(bbox_i)
            new_scores.append(score_i)

    return np.array(new_bboxes), np.array(new_scores)

def filter_bboxes_b(bboxes, scores, width,threshold, dw_threshold):
    new_bboxes = []
    new_scores = []


    for i in range(0,len(bboxes)):
        bbox_i = bboxes[i]
        score_i = scores[i]
        keep_bbox = True
        for j in range(i + 1, len(bboxes)):
            # if j==1:
            #     continue
            bbox_j = bboxes[j]
            overlap_ratio=calculate_overlap_ratio(bbox_i,bbox_j)
            distance = calculate_distance(bbox_i, bbox_j,overlap_ratio)
            dw=distance/width
            if dw < dw_threshold and overlap_ratio > threshold:
                keep_bbox = False
                break

        if keep_bbox:
            # if i==0 or i==1:
            #     continue
            new_bboxes.append(bbox_i)
            new_scores.append(score_i)

    return np.array(new_bboxes), np.array(new_scores)

if __name__ == "__main__": 
    main()