import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(voc_dir, yolo_dir):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for xml_file in os.listdir(voc_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(voc_dir, xml_file))
            root = tree.getroot()
            
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            yolo_file = os.path.join(yolo_dir, xml_file.replace('.xml', '.txt'))

            with open(yolo_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = get_class_id(class_name) 
                    if class_id == -1:
                        print(f"Warning: Class '{class_name}' not found in mapping. Skipping.")
                        continue
                    
                    bbox = obj.find('bndbox')
                    x_min = int(bbox.find('xmin').text)
                    y_min = int(bbox.find('ymin').text)
                    x_max = int(bbox.find('xmax').text)
                    y_max = int(bbox.find('ymax').text)

                    # Convert to YOLO format
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # Write to YOLO format file
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def get_class_id(class_name):
    # Define your class mapping here
    class_mapping = {
        'person': 0,
        'hard-hat': 1,
        'gloves': 2,
        'mask': 3,
        'glasses': 4,
        'boots': 5,
        'vest': 6,
        'ppe-suit': 7,
        'ear-protector': 8,
        'safety-harness': 9
    }
    return class_mapping.get(class_name, -1)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Pascal VOC annotations to YOLO format.')
    parser.add_argument('--voc_dir', type=str, required=True, help='Path to the directory containing Pascal VOC annotations')
    parser.add_argument('--yolo_dir', type=str, required=True, help='Path to the directory where YOLO annotations will be saved')
    args = parser.parse_args()

    convert_voc_to_yolo(args.voc_dir, args.yolo_dir)

    #script to run this:
    #python pascalVOC_to_yolo.py --voc_dir "datasets/labels" --yolo_dir "datasets/labels_for_yolo"



    #python pascalVOC_to_yolo.py --voc_dir "/path/to/pascalvoc/annotations" --yolo_dir "/path/to/yolov8/annotations"