import os
import cv2
import numpy as np
import json
import requests
from io import BytesIO
from PIL import Image
import pytesseract
import ollama
import tempfile
from retinaface import RetinaFace
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

DEFAULT_IMAGE_PATH = "44.jpeg"

def detect_and_blur_faces_with_retinaface(image_path, output_path=None):
    """
    Detect faces in an image using RetinaFace and blur them.
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the output image
    
    Returns:
        str: Path to the saved image with blurred faces
        int: Number of faces detected
    """
    # Read the image
    if isinstance(image_path, str):
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_path)
    else:
        # If image is already a numpy array
        img = image_path
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Determine output path
    if output_path is None:
        if isinstance(image_path, str):
            base_name = os.path.basename(image_path)
            output_path = f"blurred_{base_name}"
        else:
            output_path = "blurred_image.jpg"
    
    # Make a copy of the image
    img_copy = img.copy()
    
    try:
        print("Detecting faces using RetinaFace...")
        # Convert BGR to RGB for RetinaFace
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(rgb_img)
        
        num_faces = 0
        
        # If faces were detected
        if isinstance(faces, dict):
            num_faces = len(faces)
            
            for key in faces:
                face = faces[key]
                # Get facial area
                x1, y1, x2, y2 = [int(coord) for coord in face['facial_area']]
                
                # Ensure coordinates are within image boundaries
                h, w = img_copy.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Create a mask for the face area
                mask = np.zeros(img_copy.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Apply Gaussian blur to the face region
                blurred_img = cv2.GaussianBlur(img_copy, (99, 99), 30)
                
                # Replace the face region with the blurred version
                img_copy = np.where(mask[:, :, np.newaxis] == 255, blurred_img, img_copy)
        
        # Save the result
        cv2.imwrite(output_path, img_copy)
        
        print(f"Detected and blurred {num_faces} faces using RetinaFace")
        return output_path, num_faces
        
    except Exception as e:
        print(f"Error using RetinaFace: {e}")
        # Save the original image if face detection fails
        cv2.imwrite(output_path, img)
        return output_path, 0

def extract_text_from_image(image_path):
    """
    Extract text from the image using pytesseract OCR.
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        str: Extracted text
    """
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
        
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def analyze_image_with_ollama(image_path, model_name="moondream"):
    """
    Use Ollama's vision model to analyze the image content.
    
    Args:
        image_path (str): Path to the image
        model_name (str): Name of the model in Ollama
    
    Returns:
        str: Summary of the image
    """
    try:
        print(f"Analyzing image with Ollama model '{model_name}'...")
        
        # Get response from Ollama
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Summarize this image fully and properly.',
                    'images': [image_path],
                }
            ]
        )
        
        # Extract content from response
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        elif isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "Failed to get a valid response from Ollama."
            
    except Exception as e:
        print(f"Error analyzing image with Ollama: {e}")
        return f"Failed to analyze image: {str(e)}"

def detect_image_authenticity(image_path):
    """
    Detect if the image is AI-generated, real, or a deepfake.
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        str: Classification result (AI-generated, Real, or Deepfake)
        float: Confidence score
    """
    try:
        print("Detecting image authenticity...")
        
        # Load image
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        
        # Load the model and processor
        model = ViTForImageClassification.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real")
        processor = ViTImageProcessor.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real")
        
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Map class index to label
        label = model.config.id2label[predicted_class]
        confidence = probabilities[predicted_class].item() * 100
        
        print(f"Image authenticity classification: {label} (Confidence: {confidence:.2f}%)")
        return label, confidence
        
    except Exception as e:
        print(f"Error detecting image authenticity: {e}")
        return "Unknown", 0.0

def process_image(image_path, output_path=None, model_name="moondream"):
    """
    Main function to process an image:
    1. Detect and blur faces using RetinaFace
    2. Extract text using OCR
    3. Analyze the blurred image with Ollama
    4. Detect if the image is AI-generated, real, or deepfake
    5. Return results in JSON format
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the processed image
        model_name (str): Name of the Ollama model to use
    
    Returns:
        dict: JSON-compatible dictionary with results
    """
    # Step 1: Detect if the image is AI-generated, real, or deepfake (on unblurred image)
    authenticity_label, confidence = detect_image_authenticity(image_path)
    
    # Step 2: Detect and blur faces
    blurred_image_path, num_faces = detect_and_blur_faces_with_retinaface(image_path, output_path)
    
    # Step 3: Extract text from the original image
    text = extract_text_from_image(image_path)
    
    # Step 4: Analyze the blurred image with Ollama
    summary = analyze_image_with_ollama(blurred_image_path, model_name)
    
    # Step 5: Prepare the results
    results = {
        "image_content": summary,
        "image_authenticity": {
            "classification": authenticity_label,
            "confidence": f"{confidence:.2f}%"
        }
    }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images by detecting authenticity, blurring faces, and analyzing with Ollama')
    parser.add_argument('--image_path', '-i', default=DEFAULT_IMAGE_PATH, help='Path to the input image or URL')
    parser.add_argument('--output', '-o', help='Path to save the processed image')
    parser.add_argument('--json_output', '-j', help='Path to save the JSON results')
    parser.add_argument('--model', '-m', default="moondream", help='Name of the Ollama model to use (e.g., moondream, llama3.2-vision)')
    
    args = parser.parse_args()
    
    # Process the image
    results = process_image(args.image_path, args.output, args.model)
    
    # Print results to console
    print(json.dumps(results, indent=2))
    
    # Save JSON results if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()