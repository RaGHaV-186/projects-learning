from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"
)

text = """
Artificial intelligence is transforming many industries including healthcare,
finance, and transportation. AI systems can analyze large amounts of data,
identify patterns, and help organizations make better decisions.
Researchers continue to improve these systems so they become more accurate
and more useful in solving real-world problems.
"""

summary = summarizer(text, max_length=25, min_length=20, do_sample=False)

print(summary)

