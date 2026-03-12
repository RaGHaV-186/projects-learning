from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = "I hate my life"

result = classifier(text)

print(result)
