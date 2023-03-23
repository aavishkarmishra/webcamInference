import os
import time
import tensorflow as tf
import cv2
import numpy as np

from object_detection.utils import label_map_util
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from base64 import b64encode
# from detr_tf.training_config import TrainingConfig, training_config_parser
# from detr_tf.networks.detr import get_detr_model
# from detr_tf.data import processing
# from detr_tf.data.coco import COCO_CLASS_NAME
# from detr_tf.inference import get_model_inference, numpy_bbox_to_image


# Path to saved model

PATH_TO_SAVED_MODEL = "inference/saved_model"

# Load label map and obtain class names and ids
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index = label_map_util.create_category_index_from_labelmap(
    "label_map.pbtxt", use_display_name=True)


def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin,
                        ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def run_webcam_inference():
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #fps = int(video_capture.get(5))
    size = (frame_width, frame_height)
    print(size)
    while (True):
        ret, frame = cap.read()

        if not ret:
            break

        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
#      # The model expects a batch of images, so also add an axis with `tf.newaxis`.
        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

#      # Pass frame through detector
        detections = detect_fn(input_tensor)

#      # Set detection parameters

        score_thresh = 0.2   # Minimum threshold for object detection
        max_detections = 20

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
        scores = detections['detection_scores'][0, :max_detections].numpy()
        bboxes = detections['detection_boxes'][0, :max_detections].numpy()
        labels = detections['detection_classes'][0,
                                                 :max_detections].numpy().astype(np.int64)
        labels = [category_index[n]['name'] for n in labels]

      # Display detections
        visualise_on_image(frame, bboxes, labels, scores, score_thresh)

        end_time = time.time()
        fps = int(1/(end_time - start_time))
        start_time = end_time
        cv2.putText(frame, f"FPS: {fps}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # config = TrainingConfig()
    # args = training_config_parser().parse_args()
    # config.update_from_args(args)

    # Load the model with the new layers to finetune
    # detr = get_detr_model(config, include_top=True, weights="detr")
    # config.background_class = 91

    # Run webcam inference
    run_webcam_inference()
