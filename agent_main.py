from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Use your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'YOUR_API_KEY_HERE')
# Don't hardcode API keys in production code

# Category Classifier Agent - Identifies the type of content
category_classifier = Agent(
    name="Category Classifier",
    role="Identify the category of the input content",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    instructions=[
        "Analyze the input and determine which of the following categories it belongs to:",
        "1. Meme",
        "2. News/Information",
        "3. Technology",
        "4. Medical",
        "5. Others",
        "Return only the category number and name, e.g., '1. Meme'",
        "Be precise in your classification based on the content's nature."
    ],
    markdown=True,
)

# Meme Verifier Agent - Checks if meme content is appropriate
meme_verifier = Agent(
    name="Meme Verifier",
    role="Determine if meme content is appropriate for all users",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    instructions=[
        "Analyze the meme content and determine its appropriateness level:",
        "1. Everyone can see - The meme is appropriate for general audiences",
        "2. Selected users may be offended - The meme contains content that could be offensive to certain groups",
        "Provide your classification with a brief explanation of reasoning.",
        "Consider cultural sensitivities, potential stereotypes, and whether the humor relies on targeting specific groups."
    ],
    markdown=True,
)

# Create a custom DuckDuckGo instance with properly named functions
# This is the likely source of the error
custom_ddg = DuckDuckGo()
# Ensure all function names in this tool follow Google's naming requirements

# News/Information Verifier Agent - Fact checks news content
news_verifier = Agent(
    name="News_Information_Verifier",  # Removed slash from name
    role="Verify the accuracy of news and information",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    tools=[custom_ddg],
    instructions=[
        "Fact check the provided news/information content using your knowledge and web search:",
        "1. True - The information is factually accurate",
        "2. False - The information contains significant inaccuracies",
        "3. Not Sure - There isn't enough information to make a determination",
        "Use the DuckDuckGo search tool to find relevant information.",
        "Provide your verdict with supporting evidence and sources."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Technology Verifier Agent - Verifies technology claims
technology_verifier = Agent(
    name="Technology_Verifier",
    role="Verify the accuracy of technology-related content",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    tools=[custom_ddg],
    instructions=[
        "Verify the accuracy of the technology-related content using your knowledge and web search:",
        "1. True - The technology information is accurate",
        "2. False - The technology information contains significant inaccuracies",
        "3. Not Sure - There isn't enough information to make a determination",
        "Use the DuckDuckGo search tool to find relevant information.",
        "Provide your verdict with supporting evidence and sources."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Medical Verifier Agent - Verifies medical claims
medical_verifier = Agent(
    name="Medical_Verifier",
    role="Verify the accuracy of medical-related content",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    tools=[custom_ddg],
    instructions=[
        "Verify the accuracy of the medical-related content using your knowledge and web search:",
        "1. True - The medical information is accurate",
        "2. False - The medical information contains significant inaccuracies",
        "3. Not Sure - There isn't enough information to make a determination",
        "Use the DuckDuckGo search tool to find relevant information.",
        "Provide your verdict with supporting evidence and sources.",
        "Include appropriate disclaimers about not providing medical advice."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Other Content Verifier Agent - Verifies miscellaneous claims
other_verifier = Agent(
    name="Other_Content_Verifier",
    role="Verify the accuracy of miscellaneous content",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    tools=[custom_ddg],
    instructions=[
        "Verify the accuracy of the content using your knowledge and web search:",
        "1. True - The information is accurate",
        "2. False - The information contains significant inaccuracies",
        "3. Not Sure - There isn't enough information to make a determination",
        "Use the DuckDuckGo search tool to find relevant information.",
        "Provide your verdict with supporting evidence and sources.",
        "At the end, summarize with: Field of the context: [specific field], Authenticity of the context: [verdict], Reason for that: [brief explanation]"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Main Verification System Agent - Coordinates the team
verification_system = Agent(
    name="Content_Verification_System",
    model=Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    team=[
        category_classifier,
        meme_verifier,
        news_verifier,
        technology_verifier,
        medical_verifier,
        other_verifier
    ],
    instructions=[
        "Process content verification in multiple steps:",
        "1. Use Category Classifier to determine the type of content",
        "2. Based on the category, send the content to the appropriate verifier:",
        "   - If category is 'Meme' (1), use Meme Verifier",
        "   - If category is 'News/Information' (2), use News Verifier",
        "   - If category is 'Technology' (3), use Technology Verifier",
        "   - If category is 'Medical' (4), use Medical Verifier",
        "   - If category is 'Others' (5), use Other Verifier",
        "3. Present the final result with clear category identification and verification outcome",
        "4. Always include the following format at the end:",
        "   - Field of the context: [field]",
        "   - Authenticity of the context: [verdict]",
        "   - Reason for that: [explanation]"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Example usage
if __name__ == "__main__":
    # Test with sample content
    test_content = """
    
NASA's Webb Space Telescope recently detected carbon dioxide in the atmosphere of an exoplanet for the first time. 
This discovery, made on the hot gas giant WASP-39b, represents a significant advancement in our ability to 
analyze the atmospheres of distant worlds and search for potentially habitable planets.
    
    """
    
    print("\n=== CONTENT VERIFICATION ===")
    verification_system.print_response(
        f"Verify the following content: {test_content}",
        stream=True
    )