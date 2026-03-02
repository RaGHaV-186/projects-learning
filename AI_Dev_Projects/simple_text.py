from transformers import pipeline

analyzer = pipeline("sentiment-analysis")

result = analyzer("I hate this life")

print(result)