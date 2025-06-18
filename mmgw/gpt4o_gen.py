import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-06-01",
    api_key=os.getenv("OPENAI_KEY"),	
    azure_endpoint=f'https://{os.getenv("OPEN_AI_HOST")}'	
)

response = client.responses.create(
    model="gpt-4o",
    input="Generate an image of \"A shelf with 5 candles\"",
    tools=[{"type": "image_generation"}]
)

# Extract and save the image (base64-encoded)
image_data = [
    output.result for output in response.output
    if output.type == "image_generation_call"
]
if image_data:
    import base64
    with open("gpt4o_test.jpg", "wb") as f:
        f.write(base64.b64decode(image_data[0]))