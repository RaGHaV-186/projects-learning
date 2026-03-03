from transformers import pipeline

print("Loading the AI brain...")
generator = pipeline("text-generation", model="distilgpt2")

prompt = "The secret to becoming a great AI software engineer is"
print(f"Prompt: {prompt}\n")

# We updated the parameters to fix the warnings and stop the looping
result = generator(
    prompt,
    max_new_tokens=40,         # Tells it exactly how many new words to add
    num_return_sequences=1,
    repetition_penalty=1.2,    # Punishes the AI for repeating the same words!
    temperature=0.7,           # Makes the AI a little more creative
    pad_token_id=50256         # Silences the harmless "pad_token" warning
)

print("AI Generated Finish:")
print(result[0]['generated_text'])