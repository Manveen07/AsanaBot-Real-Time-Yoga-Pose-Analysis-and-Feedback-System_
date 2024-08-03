import warnings
warnings.filterwarnings("ignore")

from torchvision import transforms
from transformers import AutoModelForImageClassification
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
import mediapipe as mp
import cv2
import time
import keyboard as kb
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model_dir = './resultss'
model = AutoModelForImageClassification.from_pretrained(model_dir)

def classify_image(image, model, transform):
    image = transform(image)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    probabilities = F.softmax(outputs.logits, dim=1).squeeze()
    predicted_class = torch.argmax(probabilities).item()
    predicted_probability = probabilities[predicted_class].item()
    return probabilities, predicted_class, predicted_probability

id2label = {
    0: 'Bridge-Pose', 1: 'Child-Pose', 2: 'Cobra-Pose', 3: 'Downward-Dog-Pose',
    4: 'Pigeon-Pose', 5: 'Standing-Mountain-Pose', 6: 'Tree-Pose', 7: 'Triangle-Pose', 8: 'Warrior-Pose'
}

id2image = {
    0: "yoga-pose-classification-dataset/Bridge-Pose/Fotolia_51829528_M.jpg",
    1: "yoga-pose-classification-dataset/Child-Pose/Child_9671.jpg",
    2: "yoga-pose-classification-dataset/Cobra-Pose/bigyogaaa.jpg",
    3: "yoga-pose-classification-dataset/Downward-Dog-Pose/Blog-Header-Image_65123d9f-895e-4aad-ada6-335f1910c7d2.jpg",
    4: "yoga-pose-classification-dataset/Pigeon-Pose/Blog-Header-Image_e6d90723-c4cd-49be-b396-c0981e42234f.jpg",
    5: "yoga-pose-classification-dataset/Standing-Mountain-Pose/Blog-Mountain-Pose.jpg",
    6: "yoga-pose-classification-dataset/Tree-Pose/3a5eebbb7cd2bb091d82520d5d103ec1.png",
    7: "yoga-pose-classification-dataset/Triangle-Pose/ed7c8f8a5200ca132c00e44aefff88a3.jpg",
    8: "yoga-pose-classification-dataset/Warrior-Pose/AdobeStock_360772900.jpeg"
}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define drawing specs
gray_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3, color=(128, 128, 128))  # Gray color
green_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3, color=(0, 255, 0))  # Green color

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

current_pose = 3
best_times = {pose: 0 for pose in id2label.keys()}  # Initialize best times to 0
pose_start_time = None
pose_display_time = 0
pose_detected = False
user_changed_pose = False

try:
    while True:
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1) as pose:
            ret, image = cap.read()
            if not ret:
                print("Failed to capture image")
                continue

            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pose_landmarks = results.pose_landmarks

            output_increase = 1.5
            image_height, image_width, _ = image.shape
            enlarge_height = int(output_increase * image_height)
            enlarge_width = int(output_increase * image_width)
            large_image = (enlarge_width, enlarge_height)

            target_pose_image_location = id2image.get(current_pose, "")
            if target_pose_image_location and os.path.exists(target_pose_image_location):
                img = cv2.imread(target_pose_image_location)
                example_width = 150
                example_height = 150
                img = cv2.resize(img, (example_width, example_height), interpolation=cv2.INTER_AREA)
                x_offset = image.shape[1] - example_width
                y_offset = image.shape[0] - example_height
                image[y_offset:y_offset + example_height, x_offset:x_offset + example_width] = img

            if pose_landmarks is not None:
                annotated_image = image.copy()
                # Choose drawing spec based on whether the timer has started
                drawing_spec = green_drawing_spec if pose_detected else gray_drawing_spec
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                probabilities, predicted_class, predicted_probability = classify_image(pil_image, model, transform)

                pose_class = id2label[predicted_class]

                if predicted_class == current_pose:
                    if not pose_detected:
                        pose_start_time = time.time()
                        pose_detected = True
                        user_changed_pose = False  # Reset user_changed_pose flag when pose is detected
                    pose_display_time = time.time() - pose_start_time
                    # Update best time if current display time is greater
                    best_times[current_pose] = max(best_times[current_pose], pose_display_time)
                else:
                    if pose_detected and not user_changed_pose:
                        # Update best time if current display time is greater
                        best_times[current_pose] = max(best_times[current_pose], pose_display_time)
                    pose_detected = False
                    pose_start_time = None
                    pose_display_time = 0

                cv2.rectangle(annotated_image, (0, 0), (image_width, 50), (0, 0, 0), -1)
                cv2.putText(annotated_image, f"Pose: {pose_class} {predicted_probability * 100:.2f}%", (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)

                if pose_detected:
                    cv2.putText(annotated_image, f"Time: {int(pose_display_time)}s", (5, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)

                best_time = best_times[current_pose]
                cv2.putText(annotated_image, f"Best Time: {int(best_time)}s", (5, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)

                cv2.imshow('Pose', annotated_image)
            else:
                pose_detected = False
                pose_start_time = None
                pose_display_time = 0
                image = cv2.resize(image, large_image, interpolation=cv2.INTER_AREA)
                cv2.imshow('Pose', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if kb.is_pressed("right"):
                if pose_detected and not user_changed_pose:
                    best_times[current_pose] = max(best_times[current_pose], pose_display_time)
                current_pose = (current_pose + 1) % len(id2label)
                pose_start_time = None
                pose_detected = False
                pose_display_time = 0
                user_changed_pose = True
                time.sleep(0.2)  # Add a small delay to avoid multiple rapid key presses

            if kb.is_pressed("left"):
                if pose_detected and not user_changed_pose:
                    best_times[current_pose] = max(best_times[current_pose], pose_display_time)
                current_pose = (current_pose - 1) % len(id2label)
                pose_start_time = None
                pose_detected = False
                pose_display_time = 0
                user_changed_pose = True
                time.sleep(0.2)  # Add a small delay to avoid multiple rapid key presses

finally:
    cap.release()
    cv2.destroyAllWindows()

