import google.generativeai as genai

# ⚠️ Paste your API key TEMPORARILY here JUST for testing
# (Remove it later after you find the correct model)
genai.configure(api_key="AIzaSyCWGdcZHrd6QwCzhIC8jRNibG6eW089XJo")

# List all available models
models = genai.list_models()

print("\n✅ Supported Gemini Models for Your API Key:\n")

for model in models:
    print("Model name:", model.name)
    print("Supported methods:", model.supported_generation_methods)
    print("-" * 50)
