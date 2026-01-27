from art import logo

print(logo)



alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# def encrypt(original_text,shift_amount):
#
#     encrypted_text = ""
#     for letter in original_text:
#         new_index = (alphabet.index(letter) + shift_amount)%26
#         encrypted_text += alphabet[new_index]
#     print(encrypted_text)
#
# def decrypt(original_text,shift_amount):
#
#     decrypted_text = ""
#     for letter in original_text:
#         new_index = (alphabet.index(letter) + (shift_amount * -1))%26
#         decrypted_text += alphabet[new_index]
#     print(decrypted_text)
#
# encrypt("hello",1)
# decrypt("ifmmp",1)

def caesar(original_text, shift_amount, encode_or_decode):

    end_text = ""
    if encode_or_decode == "decode":
        shift_amount *= -1
    for letter in original_text:
        if letter in alphabet:
            new_index = (alphabet.index(letter)+shift_amount)%26
            end_text += alphabet[new_index]
        else:
            end_text += letter

    print(f"Here is the {encode_or_decode}d result: {end_text}")



continue_or_not = True
while continue_or_not:
    direction = input("Type 'encode' to encrypt, type 'decode' to decrypt:\n").lower()
    text = input("Type your message:\n").lower()
    shift = int(input("Type the shift number:\n"))

    caesar(original_text=text, shift_amount=shift, encode_or_decode=direction)

    end_or_not = input("enter y to continue and n to stop")
    if end_or_not == "n":
        continue_or_not = False
        print("Goodbye!")




