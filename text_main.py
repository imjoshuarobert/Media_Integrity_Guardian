from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
import torch
import torch.nn.functional as F

# Use the alternative model fine-tuned for AI detection
model_name = "wangkevin02/AI_Detect_Model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def detect_ai_text(text: str) -> float:
    """
    Returns the probability (as a percentage) that the provided text is AI-generated.
    """
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    outputs = model(**inputs)
    # Compute probabilities using softmax
    probabilities = F.softmax(outputs.logits, dim=1)
    # Assume index 1 corresponds to the "AI-generated" class
    ai_probability = probabilities[0][1].item()
    return ai_probability * 100  # Return as percentage

if __name__ == "__main__":
    sample_text = (
        """
Whispers of the Wind

The wind hums songs of days gone by,
A fleeting echo, a whispered sigh.
It dances through the ancient trees,
Carrying secrets on the breeze.

The ocean listens, deep and wide,
Its waves embracing time and tide.
The stars blink down, both old and new,
A canvas brushed in silver hue.

And here we stand, with hearts untamed,
Chasing dreams yet half-unnamed.
Like autumn leaves in golden flight,
Drifting toward the endless night."""
    )
    result = detect_ai_text(sample_text)
    print(f"Text Content:\n{sample_text}\n")
    print(f"AI-generated probability: {result:.1f}%")