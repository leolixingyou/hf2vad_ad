import os
import cv2
import numpy as np
from tqdm import tqdm
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Initialize the Roboflow API Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="api_Key"
)
custom_configuration = InferenceConfiguration(confidence_threshold=0.3)  

CLIENT.configure(custom_configuration)

def visualize_bounding_boxes(image, predictions):
    for pred in predictions:
        # Convert center (x, y) to top-left corner (x_min, y_min)
        x_min = int(pred['x'] - (pred['width'] / 2))
        y_min = int(pred['y'] - (pred['height'] / 2))
        
        # Convert dimensions to bottom-right corner (x_max, y_max)
        x_max = int(pred['x'] + (pred['width'] / 2))
        y_max = int(pred['y'] + (pred['height'] / 2))
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_with_roboflow(file_path, model_id, visualize=False):
    image = cv2.imread(file_path)
    response = CLIENT.infer(file_path, model_id=model_id)
    # class for vehicles
    predictions_class_9 = [pred for pred in response['predictions'] if pred['class_id'] == 9]
    
    if visualize:
        print(file_path)
        visualize_bounding_boxes(image, predictions_class_9)
    
    bboxes = [
        [pred['x'] - (pred['width'] / 2),
         pred['y'] - (pred['height'] / 2),
         pred['x'] + (pred['width'] / 2),
         pred['y'] + (pred['height'] / 2)]
        for pred in predictions_class_9
    ]
    return np.array(bboxes)

def process_directory_with_roboflow(directory, model_id):
    image_bboxes_list = []
    for filename in tqdm(sorted(os.listdir(directory)), desc="Processing images", leave=False):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            bboxes = process_image_with_roboflow(file_path, model_id, visualize=False)
            image_bboxes_list.append(bboxes)  
    return image_bboxes_list

root_dir = '/workspace/data/carla_local/testing/frames/'

model_id = "carla-f4l16/1"

# npy_output_path = 'carla_test_bboxes.npy'
npy_output_path = 'carla_train_bboxes.npy'

all_bboxes = []
for scenario in tqdm(sorted(os.listdir(root_dir)), desc="Processing scenarios", leave=False):
    scenario_path = os.path.join(root_dir, scenario, 'RGB_IMG')
    print(scenario_path)
    if os.path.isdir(scenario_path):
        image_bboxes_list = process_directory_with_roboflow(scenario_path, model_id)
        all_bboxes.extend(image_bboxes_list)

all_bboxes_array = np.array(all_bboxes, dtype=object)

np.save(npy_output_path, all_bboxes_array, allow_pickle=True)