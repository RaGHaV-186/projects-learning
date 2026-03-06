from transformers import  AutoTokenizer,AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history =  []

print("\nChatbot is ready! Type your message below (or press Ctrl+C to exit).")

while True:
    try:

        history_string = "\n".join(conversation_history)

        input_text = input("> ")

        full_prompt = history_string + "\n" + input_text if history_string else input_text

        inputs = tokenizer(full_prompt,return_tensors="pt")

        outputs = model.generate(**inputs)

        response = tokenizer.decode(outputs[0],skip_special_tokens=True)

        print(f"Bot:{response}")

        conversation_history.append(input_text)
        conversation_history.append(response)

    except KeyboardInterrupt:
         print("\Exiting chat.Goodbye!")
         break



