import argparse
import os
from faster_whisper import WhisperModel
from transformers import pipeline
import librosa
import soundfile as sf

def transcribe_audio(audio_path, model_size="small"):
    """
    Transcribe audio using Faster Whisper
    
    Args:
        audio_path: Path to the audio file
        model_size: Size of the Whisper model to use
        
    Returns:
        str: Transcribed text
    """
    print(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print(f"Transcribing audio from {audio_path}...")
    segments, _ = model.transcribe(audio_path)
    
    # Collect all segments into a single transcript
    full_transcript = ""
    for segment in segments:
        full_transcript += segment.text + " "
        print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
    
    return full_transcript.strip()

def detect_deepfake(audio_path):
    """
    Detect if audio is AI-generated using MelodyMachine/Deepfake-audio-detection-V2
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        dict: Classification results
    """
    print("Loading deepfake detection model...")
    pipe = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2")
    
    print(f"Analyzing audio to detect if AI-generated...")
    results = pipe(audio_path)
    
    return results

def preprocess_audio_if_needed(audio_path):
    """
    Check if audio needs preprocessing for compatibility with the deepfake detection model
    - Works with any audio format (MP3, WAV, etc.) that librosa can read
    - Convert to WAV for processing if needed
    - Ensure correct sample rate (16kHz)
    - Ensure mono channel
    
    Args:
        audio_path: Path to the audio file (can be MP3, WAV, or other formats)
        
    Returns:
        str: Path to the processed audio file
    """
    file_ext = os.path.splitext(audio_path)[1].lower()
    print(f"Processing {file_ext} audio file...")
    
    # Always convert to the required format for consistency
    try:
        # Librosa can handle various audio formats including MP3 and WAV
        print(f"Loading audio from {audio_path}...")
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Create a temporary WAV file with the correct parameters
        temp_path = os.path.splitext(audio_path)[0] + "_temp.wav"
        
        # Convert to mono if needed
        if audio.ndim > 1:
            print("Converting stereo to mono...")
            audio = librosa.to_mono(audio)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            print(f"Resampling audio from {sr}Hz to 16000Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Save as WAV
        print(f"Saving processed audio to {temp_path}...")
        sf.write(temp_path, audio, 16000)
        return temp_path
            
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        print("Attempting to continue with original file...")
        return audio_path

def analyze_audio(audio_path):
    """
    Analyze audio file - transcribe and detect if AI-generated
    
    Args:
        audio_path: Path to the audio file
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    # Process audio for compatibility
    processed_path = preprocess_audio_if_needed(audio_path)
    
    # Get transcript
    transcript = transcribe_audio(processed_path)
    
    # Detect if AI-generated
    detection_results = detect_deepfake(processed_path)
    
    # Clean up temp file if created
    if processed_path != audio_path and os.path.exists(processed_path):
        os.remove(processed_path)
        
    # Get the most likely classification
    is_real = True
    for result in detection_results:
        if result["label"] == "fake" and result["score"] > 0.5:
            is_real = False
            break
    
    # Print results
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Audio content: {transcript}")
    print(f"Audio classification: {'REAL HUMAN AUDIO' if is_real else 'AI-GENERATED AUDIO'}")
    print(f"Detailed classification results: {detection_results}")
    print("="*50)

if __name__ == "__main__":
    # Define the audio file path variable here - change this to your audio file path
    AUDIO_PATH = "fake.wav"  # MODIFY THIS LINE with your actual file path
    
    # Alternatively, you can use command line arguments
    parser = argparse.ArgumentParser(description="Analyze audio: transcribe and detect if AI-generated")
    parser.add_argument("--audio_path", default=AUDIO_PATH, 
                       help="Path to the audio file to analyze (MP3, WAV, or other formats)")
    parser.add_argument("--whisper_model", default="small", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size to use (default: small)")
    
    args = parser.parse_args()
    
    # Use the path from command line or default variable
    audio_path = args.audio_path
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        print("Please provide a valid path to an audio file (MP3, WAV, etc.).")
        print("You can modify the AUDIO_PATH variable in the script or use --audio_path argument.")
        exit(1)
        
    # Check file extension
    file_ext = os.path.splitext(audio_path)[1].lower()
    print(f"File type detected: {file_ext}")
    
    # Analyze the audio
    analyze_audio(audio_path)