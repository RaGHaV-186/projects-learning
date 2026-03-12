from pypdf import  PdfReader

reader = PdfReader("document.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

print(text[:500])