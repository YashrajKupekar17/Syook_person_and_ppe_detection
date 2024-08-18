import os
import cv2
from ultralytics import YOLO
import argparse

def draw_label(img, text, position, color):
    # Draw a filled rectangle behind the text for better visibility
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (position[0], position[1] - text_height - baseline), 
                  (position[0] + text_width, position[1]), color, -1)  # Filled rectangle
    cv2.putText(img, text, (position[0], position[1] - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # White text

def perform_inference(image_dir, output_dir, person_model, ppe_model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)

            # Perform person detection using the person model
            person_results = person_model.predict(img)[0]
            person_boxes = person_results.boxes

            # List to hold cropped person images and their corresponding bounding boxes
            cropped_persons = []

            # Crop person images and store their bounding boxes
            for box in person_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_img = img[y1:y2, x1:x2]
                cropped_persons.append((person_img, (x1, y1, x2, y2), box.conf[0]))

                # Draw person bounding boxes on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for person
                draw_label(img, f"Person: {box.conf[0]:.2f}", (x1, y1), (255, 0, 0))

            # Perform PPE detection on each cropped person image
            for person_img, (x1, y1, x2, y2), person_conf in cropped_persons:
                ppe_results = ppe_model.predict(person_img)[0]
                ppe_boxes = ppe_results.boxes

                # Draw PPE bounding boxes on the original image
                for ppe_box in ppe_boxes:
                    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, ppe_box.xyxy[0])
                    
                    # Convert PPE box coordinates back to the full image coordinates
                    ppe_x1 += x1
                    ppe_y1 += y1
                    ppe_x2 += x1
                    ppe_y2 += y1

                    # Draw the bounding box on the original image
                    class_name = ppe_results.names[int(ppe_box.cls[0])]
                    confidence = ppe_box.conf[0]
                    
                    if class_name == 'hard-hat':
                        color = (0, 255, 0)  # Green for hard-hat
                    elif class_name == 'gloves':
                        color = (0, 255, 255)  # Yellow for gloves
                    elif class_name == 'mask':
                        color = (255, 0, 255)  # Purple for mask
                    elif class_name == 'glasses':
                        color = (0, 128, 128)  # Teal for glasses
                    elif class_name == 'boots':
                        color = (0, 165, 255)  # Orange for boots
                    elif class_name == 'vest':
                        color = (128, 0, 128)  # Dark purple for vest
                    elif class_name == 'ppe-suit':
                        color = (0, 128, 0)  # Dark green for PPE suit
                    elif class_name == 'ear-protector':
                        color = (255, 165, 0)  # Brown for ear protector
                    elif class_name == 'safety-harness':
                        color = (128, 0, 0)  # Maroon for safety harness
                    else:
                        color = (0, 0, 255)  # Red for unknown PPE items
                    
                    cv2.rectangle(img, (ppe_x1, ppe_y1), (ppe_x2, ppe_y2), color, 1)  # Decreased thickness
                    draw_label(img, f"{class_name}: {confidence:.2f}", (ppe_x1, ppe_y1), color)

            # Save the processed image with bounding boxes
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference for person and PPE detection.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory path for images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path for processed images')
    parser.add_argument('--person_det_model', type=str, required=True, help='Path to the person detection model')
    parser.add_argument('--ppe_detection_model', type=str, required=True, help='Path to the PPE detection model')
    args = parser.parse_args()

    person_model = YOLO(args.person_det_model)
    ppe_model = YOLO(args.ppe_detection_model)
    perform_inference(args.input_dir, args.output_dir, person_model, ppe_model)


    #script:  python inference.py --input_dir "datasets/images" --output_dir "datasets/output" --person_det_model "weights/person_detection.pt" --ppe_detection_model "weights/ppe_detection.pt"