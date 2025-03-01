import os
import json
from dotenv import load_dotenv

# Import specialized modules
from video_main import process_video
from image_main import process_image
from audio_main import analyze_audio
from text_main import detect_ai_text

# Import agent verification system
from agent_main import verification_system

# Load environment variables
load_dotenv(override=True)

# Default output directory
DEFAULT_OUTPUT_DIR = "output"

def ensure_directory(directory):
    """Ensure output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def read_text_file(file_path):
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None

def process_video_content(video_path, output_dir):
    """Process video content and return results"""
    print("\n====== PROCESSING VIDEO ======")
    try:
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Process video using video_main.py
        results = process_video(
            video_path=video_path,
            output_dir=output_dir,
            frames_per_second=1,  # Default from video_main.py
            ollama_model="moondream",  # Default model
            summary_model="phi4:latest",  # Default model
            whisper_model_size="small"  # Default model size
        )
        
        # Format content for agent verification
        content_for_verification = {
            "content_type": "video",
            "video_content": results.get('final_visual_summary', ''),
            "audio_content": results.get('audio_transcript', {}).get('full_text', ''),
            "video_classification": results.get('video_classification', {}).get('result', 'Unknown'),
            "audio_classification": results.get('audio_classification', {}).get('result', 'Unknown'),
            "raw_results": results
        }
        
        return content_for_verification
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            "content_type": "video",
            "error": f"Failed to process video: {str(e)}"
        }

def process_image_content(image_path, output_dir):
    """Process image content and return results"""
    print("\n====== PROCESSING IMAGE ======")
    try:
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Generate output path for processed image
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{base_name}")
        
        # Process image using image_main.py
        results = process_image(
            image_path=image_path,
            output_path=output_path,
            model_name="moondream"  # Default model
        )
        
        # Format content for agent verification
        content_for_verification = {
            "content_type": "image",
            "image_content": results.get('image_content', ''),
            "image_classification": results.get('image_authenticity', {}).get('classification', 'Unknown'),
            "raw_results": results
        }
        
        return content_for_verification
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            "content_type": "image",
            "error": f"Failed to process image: {str(e)}"
        }

def process_audio_content(audio_path, output_dir):
    """Process audio content and return results"""
    print("\n====== PROCESSING AUDIO ======")
    try:
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Process audio using audio_main.py
        # The analyze_audio function prints its results directly
        analyze_audio(audio_path)
        
        # Since analyze_audio doesn't return a value but prints results,
        # we'll need to capture the transcript and classification separately
        # using the individual functions
        
        from audio_main import transcribe_audio, detect_deepfake
        
        # Get transcript
        transcript = transcribe_audio(audio_path)
        
        # Detect if AI-generated
        detection_results = detect_deepfake(audio_path)
        
        # Determine if real or fake
        is_real = True
        for result in detection_results:
            if result.get("label") == "fake" and result.get("score", 0) > 0.5:
                is_real = False
                break
                
        # Format content for agent verification
        content_for_verification = {
            "content_type": "audio",
            "audio_content": transcript,
            "audio_classification": "REAL HUMAN AUDIO" if is_real else "AI-GENERATED AUDIO",
            "raw_results": {
                "transcript": transcript,
                "detection_results": detection_results
            }
        }
        
        return content_for_verification
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return {
            "content_type": "audio",
            "error": f"Failed to process audio: {str(e)}"
        }

def process_text_content(text_content):
    """Process text content and return results"""
    print("\n====== PROCESSING TEXT ======")
    try:
        # Process text using text_main.py
        ai_probability = detect_ai_text(text_content)
        
        # Format content for agent verification
        content_for_verification = {
            "content_type": "text",
            "text_content": text_content,
            "text_classification": "AI-GENERATED" if ai_probability > 70 else "HUMAN-WRITTEN",
            "ai_probability": ai_probability,
            "raw_results": {
                "ai_probability": ai_probability
            }
        }
        
        return content_for_verification
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "content_type": "text",
            "error": f"Failed to process text: {str(e)}"
        }

def verify_content_with_agent(content):
    """Verify content using the agent verification system"""
    print("\n====== VERIFYING CONTENT AUTHENTICITY ======")
    try:
        # Format content for verification system based on content type
        verification_input = ""
        
        if content.get("error"):
            return {
                "verification_result": "ERROR",
                "details": content.get("error")
            }
        
        content_type = content.get("content_type")
        
        if content_type == "video":
            verification_input = f"""
Video Content: {content.get('video_content', '')}
Audio Content: {content.get('audio_content', '')}
Video Classification: {content.get('video_classification', 'Unknown')}
Audio Classification: {content.get('audio_classification', 'Unknown')}
"""
        elif content_type == "image":
            verification_input = f"""
Image Content: {content.get('image_content', '')}
Image Classification: {content.get('image_classification', 'Unknown')}
"""
        elif content_type == "audio":
            verification_input = f"""
Audio Content: {content.get('audio_content', '')}
Audio Classification: {content.get('audio_classification', 'Unknown')}
"""
        elif content_type == "text":
            verification_input = f"""
Text Content: {content.get('text_content', '')}
AI Probability: {content.get('ai_probability', 0)}%
Text Classification: {content.get('text_classification', 'Unknown')}
"""
        
        # Use the verification_system to verify content
        # Capture the response - using a special method to get response as an object
        # Note: We're adapting this from the verification_system usage in agent_main.py
        verification_result = verification_system.run(f"Verify the following content: {verification_input}")
        
        return {
            "verification_result": verification_result,
            "input_content": verification_input
        }
        
    except Exception as e:
        print(f"Error verifying content: {e}")
        return {
            "verification_result": "ERROR",
            "details": f"Failed to verify content: {str(e)}"
        }

def get_valid_file_path(prompt, file_must_exist=True):
    """Get a valid file path from the user"""
    while True:
        file_path = input(prompt).strip()
        
        # Check if file exists if required
        if file_must_exist and not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist. Please enter a valid path.")
            continue
            
        # Check if it's actually a file (not a directory)
        if file_must_exist and os.path.isdir(file_path):
            print(f"Error: '{file_path}' is a directory. Please enter a file path.")
            continue
            
        return file_path

def interactive_main():
    """Interactive main function to guide user through content processing"""
    print("\n======== CONTENT VERIFICATION SYSTEM ========")
    print("This system can process and verify different types of content.")
    
    # Create temp directory for all outputs
    output_dir = "temp"
    ensure_directory(output_dir)
    
    # First ask for content type
    print("\nWhat type of content would you like to analyze?")
    print("1. Video")
    print("2. Image")
    print("3. Audio")
    print("4. Text")
    
    while True:
        content_type_choice = input("Enter your choice (1-4): ").strip()
        if content_type_choice in ["1", "2", "3", "4"]:
            break
        print("Invalid choice. Please enter a number from 1 to 4.")
    
    # Process based on content type
    content_results = None
    
    if content_type_choice == "1":  # Video
        video_path = get_valid_file_path("Enter the path to the video file: ")
        content_results = process_video_content(video_path, output_dir)
        
    elif content_type_choice == "2":  # Image
        image_path = get_valid_file_path("Enter the path to the image file: ")
        content_results = process_image_content(image_path, output_dir)
        
    elif content_type_choice == "3":  # Audio
        audio_path = get_valid_file_path("Enter the path to the audio file: ")
        content_results = process_audio_content(audio_path, output_dir)
        
    elif content_type_choice == "4":  # Text
        print("\nEnter or paste the text you want to analyze:")
        text_content = input(">> ").strip()
        
        # If input is very short, ask if they want to enter multiline text
        if len(text_content) < 50:
            multiline = input("Would you like to enter longer text (multiple lines)? (y/n): ").strip().lower()
            if multiline.startswith('y'):
                print("Enter your text below (type 'END' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                text_content = "\n".join(lines)
        
        content_results = process_text_content(text_content)
    
    # Print content processing results
    print("\n===== CONTENT PROCESSING RESULTS =====")
    for key, value in content_results.items():
        if key != "raw_results":  # Skip the raw results to keep output cleaner
            print(f"{key}: {value}")
    
    # Ask if user wants to verify content
    verify_choice = input("\nWould you like to verify the content authenticity? (y/n): ").strip().lower()
    
    verification_results = None
    if verify_choice.startswith('y'):
        verification_results = verify_content_with_agent(content_results)
        
        # Print verification results
        print("\n===== VERIFICATION RESULTS =====")
        print(verification_results.get("verification_result", "Verification failed"))
    
    # Combine results
    final_results = {
        "content_processing": content_results,
        "verification": verification_results
    }
    
    # Ask if user wants to save results to JSON
    save_choice = input("\nWould you like to save the detailed results to a JSON file? (y/n): ").strip().lower()
    
    if save_choice.startswith('y'):
        json_path = input("Enter the path to save the JSON file: ").strip()
        if not json_path:
            json_path = os.path.join(output_dir, "results.json")
        
        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nDetailed results saved to {json_path}")
    
    print("\nThank you for using the Content Verification System.")

if __name__ == "__main__":
    interactive_main()