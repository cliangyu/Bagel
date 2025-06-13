#  pip install google-generativeai
import os
import time
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO


# Function to Gemini API call with max 3 retries
def generate_content_with_retries(
    client, 
    contents, 
    model="gemini-2.0-flash-preview-image-generation",
    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]), 
    retries=3, 
    wait_time=30
):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retries failed. Exiting.")
                return None


client = genai.Client(os.getenv("GOOGLE_API_KEY"))

# # Text and image input (**THIS OFTEN FAILS WITH CODE 500 INTERNAL ERROR)
# image_path = "/home/jiahuikchen/BAGEL/mmvp_imgs/ladybug_up.jpg"
# image = Image.open(image_path)
# prompt = "Edit the image so the bug is pink."
# response = generate_content_with_retries(
#     client=client,
#     model="gemini-2.0-flash-preview-image-generation",
#     contents=[prompt, image],
#     config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
# )
# Text input only
response = generate_content_with_retries(
    client=client,
    model="gemini-2.0-flash-preview-image-generation",
    contents=["Generate an image of \"A shelf with 5 candles\""],
    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
)

if response:
    # There's no response.image field (yet?) need to parse
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save('gemini_test.jpg')