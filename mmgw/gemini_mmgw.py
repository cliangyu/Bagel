#  pip install google-generativeai
import google.generativeai as genai
import os
import httpx
import base64

genai.configure(api_key="YOUR_API_KEY")
# Retrieve an image
image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg/2560px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg"
image = httpx.get(image_path)

# Choose a Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Create a prompt
prompt = "Caption this image."
response = model.generate_content(
    [
        {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image.content).decode("utf-8"),
        },
        prompt,
    ]
)

print(">" + response.text)