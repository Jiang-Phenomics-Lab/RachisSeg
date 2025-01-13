# Merge multiple annotated coco-formated json files
import json
import os

# json file path
path = '/home/ubuntu/picture/coco_jsons/'
entries = os.listdir(path)
entries.sort()

main = open(path + entries[0])
main = json.load(main)

main_image_number = len(main['images'])
main_annotation_number = len(main['annotations'])

# acquire the image info and ann from json file
for entry in entries[1:]:
    file = open(path + entry)
    file = json.load(file)

    for i in file['images']:
        main['images'].append(i)

    for i in file['annotations']:
        main['annotations'].append(i)
# modify the image id of image info
for i in range(len(main['images'])):
    main['images'][i]['id'] = i+1
# modify the image id of ann
for i in range(len(main['annotations'])):
    main['annotations'][i]['id'] = i+1
    if main['annotations'][i]['id'] > main_annotation_number:
        main['annotations'][i]['image_id'] = main['annotations'][i]['image_id'] + main_image_number
# file save path
with open('/home/ubuntu/picture/merge_coco.json', 'w') as outfile:
    json.dump(main, outfile)
