# Importing necessary libraries and modules
import warnings  # Import the 'warnings' module for handling warnings
warnings.filterwarnings("ignore")  # Ignore warnings during execution


from datasets import  Image




from torchvision import transforms

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the input size expected by the model
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])
from transformers import AutoModelForImageClassification

# Path to the directory where the model is saved
model_dir = './resultss'

# Load the model for image classification
model = AutoModelForImageClassification.from_pretrained(model_dir)

from PIL import Image
import torch
import torch.nn.functional as F


# Function to classify an image and get probabilities
def classify_image(image_path, model, transform):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs.logits, dim=1).squeeze()

    # Get the class with the highest probability
    predicted_class = torch.argmax(probabilities).item()
    predicted_probability = probabilities[predicted_class].item()

    return probabilities, predicted_class, predicted_probability

# Example usage: Classify an image
id2label={0: 'Bridge-Pose', 1: 'Child-Pose', 2: 'Cobra-Pose', 3: 'Downward-Dog-Pose', 4: 'Pigeon-Pose', 5: 'Standing-Mountain-Pose', 6: 'Tree-Pose', 7: 'Triangle-Pose', 8: 'Warrior-Pose'}

image_path1=r"C:\Users\Manveen\PycharmProjects\AsanaBot\yoga-pose-classification-dataset\Tree-Pose\images153.jpg"
image_path = r"C:\Users\Manveen\PycharmProjects\AsanaBot\yoga-pose-classification-dataset\Bridge-Pose\image16.jpeg"
probabilities, predicted_class, predicted_probability = classify_image(image_path, model, transform)
for i, prob in enumerate(probabilities):
    print(f"Class {id2label[i]}: {prob*100:.2f}%")


print(f"Predicted class: {id2label[predicted_class]} with probability {predicted_probability:.2%}")