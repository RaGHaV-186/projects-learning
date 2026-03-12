from transformers import pipeline

generator = pipeline("text-generation",model="gpt2",framework="pt")

print("AI chatbot (type exit to stop)\n")

while True:
    user_input = input("You:")

    if user_input.lower() == "exit":
        break

    response = generator(user_input,max_length= 50,num_return_sequences=1)

    print("AI:",response[0]["generated_text"])