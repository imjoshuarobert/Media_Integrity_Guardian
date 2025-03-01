import os
import cv2
import numpy as np
import json
import time
import torch
import argparse
import subprocess
from collections import Counter
from PIL import Image
from faster_whisper import WhisperModel
from retinaface import RetinaFace
import ollama

from transformers import ViTForImageClassification, ViTImageProcessor, pipeline

# Configuration variables
DEFAULT_VIDEO_PATH = "1.mp4"  # Default video path
FRAMES_PER_SECOND = 1  # Number of frames to extract per second
OUTPUT_DIR = "output_frames"  # Directory to save processed frames
TEMP_AUDIO_PATH = "temp_audio.wav"  # Temporary path to save extracted audio

def ensure_output_dir(directory):
    """Ensure output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def extract_frames(video_path, frames_per_second):
    """
    Extract frames from a video at specified frames per second
    
    Args:
        video_path (str): Path to video file
        frames_per_second (int): Number of frames to extract per second
    
    Returns:
        list: List of (frame, timestamp) tuples
    """
    frames = []
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video properties: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")
    
    # Calculate frame interval
    interval = int(fps / frames_per_second)
    if interval < 1:
        interval = 1
    
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Process every Nth frame based on the interval
        if frame_count % interval == 0:
            timestamp = frame_count / fps
            frames.append((frame, timestamp))
            
        frame_count += 1
    
    video.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames

def extract_audio(video_path, output_audio_path):
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path (str): Path to video file
        output_audio_path (str): Path to save extracted audio
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use ffmpeg to extract audio
        command = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-y",  # Overwrite if file exists
            output_audio_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully extracted audio to {output_audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error extracting audio: {e}")
        return False

def detect_and_blur_faces(frame):
    """
    Detect faces in a frame using RetinaFace and blur them
    
    Args:
        frame: Input frame (numpy array)
    
    Returns:
        numpy.ndarray: Frame with blurred faces
        int: Number of faces detected
    """
    # Make a copy of the frame
    img = frame.copy()
    h, w = img.shape[:2]
    
    try:
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
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Create a mask for the face area
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Apply Gaussian blur to the face region
                blurred_img = cv2.GaussianBlur(img, (99, 99), 30)
                
                # Replace the face region with the blurred version
                img = np.where(mask[:, :, np.newaxis] == 255, blurred_img, img)
        
        return img, num_faces
        
    except Exception as e:
        print(f"Error using RetinaFace: {e}")
        # Return original image if face detection fails
        return img, 0

def analyze_with_ollama(image_path, model_name="moondream"):
    """
    Use Ollama's vision model to analyze the image content.
    
    Args:
        image_path (str): Path to the image
        model_name (str): Name of the model in Ollama
    
    Returns:
        str: Description of the image
    """
    try:
        print(f"Analyzing image with Ollama model '{model_name}'...")
        
        # Get response from Ollama
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Briefly describe what you see in this image.',
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

def analyze_image_ai_vs_real(image_path):
    """
    Analyze image to determine if it's AI-generated, real, or deepfake
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Classification result
        float: Confidence score
    """
    try:
        # Load the model and processor
        model = ViTForImageClassification.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real")
        processor = ViTImageProcessor.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real")
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probs[predicted_class].item()
        
        # Map class index to label
        label = model.config.id2label[predicted_class]
        return label, confidence
    
    except Exception as e:
        print(f"Error analyzing image for AI vs real: {e}")
        return "Unknown", 0.0

def analyze_audio_ai_vs_real(audio_path):
    """
    Analyze audio to determine if it's AI-generated or real
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        str: Classification result
        float: Confidence score
    """
    try:
        # Use the pipeline for audio classification
        pipe = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2")
        
        # Get classification result
        results = pipe(audio_path)
        
        # Return the top result and its score
        top_result = results[0]
        return top_result["label"], top_result["score"]
    
    except Exception as e:
        print(f"Error analyzing audio for AI vs real: {e}")
        return "Unknown", 0.0

def transcribe_audio_with_whisper(audio_path, model_size="small"):
    """
    Transcribe audio using faster_whisper
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
    
    Returns:
        dict: Transcription result with full text and segments
    """
    try:
        print(f"Loading Whisper model ({model_size})...")
        # Initialize the whisper model
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        print(f"Transcribing audio file: {audio_path}")
        # Transcribe the audio file
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        # Collect all segments with timestamps
        full_transcript = ""
        timestamped_transcript = []
        
        for segment in segments:
            # Add segment text to full transcript
            full_transcript += segment.text + " "
            # Save segment with timestamps
            timestamped_transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            
        result = {
            "full_text": full_transcript.strip(),
            "segments": timestamped_transcript,
            "detected_language": info.language,
            "language_probability": info.language_probability
        }
        
        return result
    
    except Exception as e:
        print(f"Error transcribing audio with faster_whisper: {e}")
        return {"full_text": f"Failed to transcribe audio: {str(e)}", "segments": []}

def get_best_response(error_message):
    """
    Parse error message from Google AI API and suggest solutions
    
    Args:
        error_message (str): Error message from Google AI API
    
    Returns:
        str: Suggestion for fixing the issue
    """
    if "API key" in error_message:
        return "There seems to be an issue with your Google AI API key. Make sure it's valid and properly set."
    elif "rate limit" in error_message.lower():
        return "You've hit Google's rate limit. Please wait a bit before trying again."
    elif "permission" in error_message.lower():
        return "You might not have permission to use this model. Check your Google AI API access permissions."
    else:
        return f"An error occurred: {error_message}. Try using a different model or check your API access."

def get_final_summary_from_ollama(frame_results, model_name="phi4:latest"):
    """
    Send frame descriptions to Ollama using Phi4 for a final summary based only on visual content
    
    Args:
        frame_results (list): List of frame results with descriptions
        model_name (str): Name of the Ollama model to use
    
    Returns:
        str: Final summary of the video based on visual content only
    """
    try:
        print(f"Generating summary with Ollama model '{model_name}'...")
        
        # Construct the prompt with all frame descriptions - focus only on visual content
        prompt = "I'm going to provide descriptions of frames from a video. Based ONLY on these visual descriptions, please give a comprehensive summary of what the video appears to show. Ignore any mention of audio.\n\n"
        
        for frame in frame_results:
            prompt += f"Frame {frame['frame_number']} (Timestamp: {frame['timestamp']}):\n"
            prompt += f"Description: {frame['description']}\n"
            prompt += f"AI/Real/Deepfake analysis: {frame['ai_real_classification']} (confidence: {frame['ai_real_confidence']:.2f})\n\n"
        
        prompt += "Based on these visual frame descriptions ONLY, what is happening in this video? Please provide a concise but comprehensive summary of the visual content only."
        
        # Get response from Ollama
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        # Extract content from response
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        elif isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "Failed to get a valid summary from Ollama."
            
    except Exception as e:
        print(f"Error generating summary with Ollama: {e}")
        return f"Failed to generate video summary: {str(e)}"

def process_video(video_path, output_dir, frames_per_second, 
                  ollama_model="moondream", summary_model="phi4:latest", whisper_model_size="small"):
    """
    Process a video by extracting frames and audio, analyzing both
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save processed frames
        frames_per_second (int): Number of frames to extract per second
        ollama_model (str): Name of the Ollama model to use
        groq_api_key (str): Groq API key for final summary
        whisper_model_size (str): Size of the Whisper model to use
    
    Returns:
        dict: Results including frame summaries and statistics
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)
    
    # Extract frames from video
    frames = extract_frames(video_path, frames_per_second)
    
    # Extract audio from video
    audio_path = os.path.join(output_dir, TEMP_AUDIO_PATH)
    audio_extracted = extract_audio(video_path, audio_path)
    
    # Process each frame
    frame_results = []
    ai_real_classifications = []
    
    for i, (frame, timestamp) in enumerate(frames):
        print(f"\nProcessing frame {i+1}/{len(frames)} (timestamp: {timestamp:.2f}s)")
        
        # Define output path for this frame
        frame_filename = f"frame_{i:04d}_{timestamp:.2f}s.jpg"
        output_path = os.path.join(output_dir, frame_filename)
        
        # Detect and blur faces
        blurred_frame, num_faces = detect_and_blur_faces(frame)
        
        # Save the blurred frame
        cv2.imwrite(output_path, blurred_frame)
        print(f"  Saved blurred frame to {output_path}")
        print(f"  Detected {num_faces} faces")
        
        # Analyze frame to determine if it's AI-generated, real, or deepfake
        ai_real_classification, ai_real_confidence = analyze_image_ai_vs_real(output_path)
        ai_real_classifications.append(ai_real_classification)
        print(f"  AI/Real/Deepfake analysis: {ai_real_classification} (confidence: {ai_real_confidence:.2f})")
        
        # Analyze the frame with Ollama
        description = analyze_with_ollama(output_path, ollama_model)
        print(f"  Frame description: {description}")
        
        # Store results for this frame
        frame_results.append({
            "frame_number": i+1,
            "timestamp": f"{timestamp:.2f}s",
            "filename": frame_filename,
            "faces_detected": num_faces,
            "description": description,
            "ai_real_classification": ai_real_classification,
            "ai_real_confidence": ai_real_confidence
        })
    
    # Determine overall AI/Real/Deepfake classification based on majority vote
    if ai_real_classifications:
        most_common = Counter(ai_real_classifications).most_common(1)[0]
        overall_classification = most_common[0]
        classification_count = most_common[1]
        classification_percentage = (classification_count / len(ai_real_classifications)) * 100
    else:
        overall_classification = "Unknown"
        classification_percentage = 0
    
    # Process audio if available
    audio_classification = None
    audio_classification_confidence = 0.0
    transcript_result = None
    
    if audio_extracted:
        # Analyze audio for AI vs Real
        audio_classification, audio_classification_confidence = analyze_audio_ai_vs_real(audio_path)
        print(f"\nAudio classification: {audio_classification} (confidence: {audio_classification_confidence:.2f})")
        
        # Transcribe audio with faster_whisper
        print("\nTranscribing audio with faster_whisper...")
        transcript_result = transcribe_audio_with_whisper(audio_path, whisper_model_size)
        print(f"Audio transcript: {transcript_result['full_text']}")
        
        # Print some timestamped segments as a sample
        if transcript_result["segments"]:
            print("\nSample of timestamped segments:")
            for segment in transcript_result["segments"][:3]:  # Show first 3 segments
                print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
            if len(transcript_result["segments"]) > 3:
                print("...")
    
    # Generate final video summary from Ollama based on visual content only
    final_visual_summary = None
    print("\nGenerating final video summary using Ollama with Phi4...")
    final_visual_summary = get_final_summary_from_ollama(frame_results, summary_model)
    
    # Create results
    summary = {
        "video_path": video_path,
        "total_frames_processed": len(frames),
        "total_faces_detected": sum(result["faces_detected"] for result in frame_results),
        "video_classification": {
            "result": overall_classification,
            "confidence_percentage": classification_percentage
        },
        "audio_classification": {
            "result": audio_classification,
            "confidence": audio_classification_confidence
        },
        "audio_transcript": transcript_result,
        "frame_results": frame_results,
        "final_visual_summary": final_visual_summary
    }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Process a video by extracting frames, detecting faces, blurring them, analyzing content, and generating summaries')
    parser.add_argument('--video', '-v', default=DEFAULT_VIDEO_PATH, help='Path to the input video')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='Directory to save processed frames')
    parser.add_argument('--fps', '-f', type=int, default=FRAMES_PER_SECOND, help='Frames per second to extract')
    parser.add_argument('--ollama-model', '-m', default="moondream", help='Name of the Ollama model to use (default: moondream)')
    parser.add_argument('--json', '-j', help='Path to save the JSON results')
    parser.add_argument('--summary-model', '-s', default="phi4:latest", help='Ollama model to use for final video summary (default: phi4:latest)')
    parser.add_argument('--whisper-model', '-w', default="small", choices=["tiny", "base", "small", "medium", "large"], 
                       help='Size of the Whisper model to use for audio transcription')
    
    args = parser.parse_args()
    
    # Process the video
    start_time = time.time()
    results = process_video(
        args.video, 
        args.output, 
        args.fps, 
        args.ollama_model, 
        args.summary_model,
        args.whisper_model
    )
    processing_time = time.time() - start_time
    
    # Add processing time to results
    results["processing_time_seconds"] = processing_time
    
    # Print summary
    print("\n===== Processing Summary =====")
    print(f"Video: {args.video}")
    print(f"Frames processed: {results['total_frames_processed']}")
    print(f"Total faces detected: {results['total_faces_detected']}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Print classifications
    print("\n===== Content Classification =====")
    print(f"Video content classification: {results['video_classification']['result']} " +
          f"({results['video_classification']['confidence_percentage']:.2f}%)")
    
    if results['audio_classification']['result']:
        print(f"Audio classification: {results['audio_classification']['result']} " +
              f"({results['audio_classification']['confidence']:.2f})")
    
    # Display audio transcript if available
    if results.get('audio_transcript') and results['audio_transcript'].get('full_text'):
        print("\n===== Audio Transcript =====")
        print(results['audio_transcript']['full_text'])
        
        # Print language detection info
        if results['audio_transcript'].get('detected_language'):
            print(f"\nDetected language: {results['audio_transcript']['detected_language']} " +
                  f"(confidence: {results['audio_transcript']['language_probability']:.2f})")
    
    # Display final visual summary if available
    if results.get('final_visual_summary'):
        print("\n===== Final Video Content Summary (Phi4 via Ollama) =====")
        print(results['final_visual_summary'])
    
    # Print the four key outputs as requested
    print("\n===== FINAL RESULTS =====")
    print(f"1. Video classification: {results['video_classification']['result']}")
    print(f"2. Video content: {results.get('final_visual_summary', 'No visual summary available')}")
    print(f"3. Audio context: {results.get('audio_transcript', {}).get('full_text', 'No transcript available')}")
    print(f"4. Audio classification: {results['audio_classification']['result'] or 'Unknown'}")
    
    # Save JSON results if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.json}")

if __name__ == "__main__":
    main()