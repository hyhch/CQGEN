import openai
 
openai.api_type = "azure"
openai.api_base = "https://YOUR_AZURE_ENDPOINT.openai.azure.com/"
openai.api_version = "2024-10-01-preview"
openai.api_key = "YOUR_API_KEY_HERE"
 
response = openai.ChatCompletion.create(
    engine="gpt-4o",  # 你在 Azure 中部署时取的名字
    messages=[
        {"role": "user", "content": "9.9和9.11谁大？"}
    ]
)
 
print(response['choices'][0]['message']['content'])