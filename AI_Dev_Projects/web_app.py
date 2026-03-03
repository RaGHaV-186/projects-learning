import gradio as gr
from matplotlib.pyplot import title
from transformers import pipeline

analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(user_text):

    result = analyzer(user_text)

    label = result[0]['label']

    score = round(result[0]['score'] * 100,2)

    return f"The AI says this is {label} with {score}% confidence"


demo = gr.Interface(
    fn = analyze_sentiment,
    inputs= gr.Textbox(lines=3,placeholder="Type a sentence here"),
    outputs = "text",
    title = "My First AI Web App",
    description="Type any sentence and the AI will tell you if the vibe is positive or negative!"
)

if __name__ == "__main__":
    demo.launch()