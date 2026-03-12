from transformers import pipeline
import gradio as gr


generator = pipeline("text-generation",model="gpt2",framework="pt")

def chat(user_messgae):
    response = generator(user_messgae,max_length=50,do_sample=True,temperature=0.7)

    return response[0]['generated_text']

interface = gr.Interface(fn=chat,inputs="text",outputs="text",title="AI chatbot")

interface.launch()