# Content Verification System

A comprehensive system for analyzing and verifying different types of content (video, image, audio, and text) and determining if the content is AI-generated or human-created.

## Overview

This system integrates several specialized modules:

- **Video Analysis**: Analyzes video content frame by frame, detects and blurs faces, transcribes audio, and determines if content is AI-generated
- **Image Analysis**: Detects and blurs faces, extracts text via OCR, analyzes content, and determines if AI-generated
- **Audio Analysis**: Transcribes audio and detects if AI-generated
- **Text Analysis**: Determines if text is AI-generated with probability score
- **Agent Verification**: Categorizes and verifies the authenticity of the content

## Requirements

- Python 3.8+
- Various dependencies for each module (see Installation section)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/content-verification-system.git
   cd content-verification-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv faceblur_env
   
   # On Windows
   faceblur_env\Scripts\activate
   
   # On macOS/Linux
   source faceblur_env/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   # Add other API keys as needed
   ```

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Follow the interactive prompts:
   - Select the type of content to analyze (video, image, audio, or text)
   - Provide the path to the file (for video, image, and audio) or enter text directly
   - View the analysis results
   - Optionally verify the authenticity with the agent system
   - Optionally save the results to a JSON file

## Module Details

### Video Analysis (`video_main.py`)
- Extracts frames and audio from video
- Detects and blurs faces using RetinaFace
- Analyzes frames with Ollama
- Transcribes audio with Whisper
- Detects if content is AI-generated

### Image Analysis (`image_main.py`)
- Detects and blurs faces with RetinaFace
- Extracts text using Tesseract OCR
- Analyzes image content with Ollama
- Detects if the image is AI-generated, real, or a deepfake

### Audio Analysis (`audio_main.py`)
- Preprocesses audio for compatibility
- Transcribes audio using faster-whisper
- Detects if audio is AI-generated using deepfake detection

### Text Analysis (`text_main.py`)
- Uses a fine-tuned model to determine if text is AI-generated
- Provides a probability score

### Agent Verification (`agent_main.py`)
- Uses a team of specialized agents to classify and verify content
- Categories include meme, news/information, technology, medical, and others
- Provides detailed verification with evidence and sources

## Output

All processed files are saved in the `temp` directory. For detailed analysis, you can save results to a JSON file.

## Note

This system is for educational and research purposes. The AI detection models may not be 100% accurate and should be used as tools to assist human judgment, not as definitive truth.

## License

[Your license information here]