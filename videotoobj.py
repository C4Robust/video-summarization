import cv2
import tensorflow as tf
import numpy as np

# Load pre-trained object detection model
model_dir = 'D:/Video to Object/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_dir)

# Function to perform object detection on a frame
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    detected_classes = detections['detection_classes'][0].numpy().astype(int)
    detected_scores = detections['detection_scores'][0].numpy()

    return detected_classes, detected_scores

# Function to summarize video based on detected objects with frame interval
def summarize_video(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return ""

    summary = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame only if frame_count is a multiple of interval
        if frame_count % interval == 0:
            detected_classes, detected_scores = detect_objects(frame)

            # Get top detected object labels
            top_indices = np.argsort(detected_scores)[::-1][:3]
            detected_labels = [get_label_from_index(class_index) for class_index in detected_classes[top_indices]]

            # Append unique detected labels to summary
            unique_labels = set(detected_labels)  # Remove duplicates
            frame_summary = f"Frame {frame_count}: {' '.join(unique_labels)}"
            summary.append(frame_summary)

        frame_count += 1

    cap.release()
    return '\n'.join(summary)

def get_label_from_index(class_index):
    # Replace this with a dictionary-based label mapping
    label_map = {
        1: 'person', 
        2: 'bicycle', 
        3: 'car', 
        # Add more class mappings as needed...
    }

    if class_index in label_map:
        return label_map[class_index]
    else:
        return 'unknown'  # Handle unknown class indices gracefully

# Example usage:
video_path = 'D:/Video to Object/input/click/test1.mp4'
frame_interval = 1000  # Process every 5th frame
video_summary = summarize_video(video_path, interval=frame_interval)
print("Video Summary:")
print(video_summary)