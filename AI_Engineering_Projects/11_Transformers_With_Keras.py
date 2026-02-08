import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, AdditiveAttention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.simplefilter('ignore', FutureWarning)

input_texts = [
    "Hello.", "How are you?", "I am learning machine translation.", "What is your name?", "I love programming."
]
target_texts_raw = [
    "Hola.", "¿Cómo estás?", "Estoy aprendiendo traducción automática.", "¿Cuál es tu nombre?", "Me encanta programar."
]

target_texts = ["startseq " + x + " endseq" for x in target_texts_raw]

# --- Tokenization ---
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(target_texts)
output_sequences = output_tokenizer.texts_to_sequences(target_texts)

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

# --- Padding ---
max_input_length = max([len(seq) for seq in input_sequences])
max_output_length = max([len(seq) for seq in output_sequences])

encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding="post")
output_sequences_padded = pad_sequences(output_sequences, maxlen=max_output_length, padding="post")

# --- Data Preparation (Teacher Forcing) ---
decoder_input_data = output_sequences_padded[:, :-1]
decoder_target_data_raw = output_sequences_padded[:, 1:]

# One-Hot Encoding Targets
num_samples = len(input_texts)
decoder_target_data = np.zeros(
    (num_samples, max_output_length - 1, output_vocab_size),
    dtype='float32'
)

for i, seq in enumerate(decoder_target_data_raw):
    for t, word_index in enumerate(seq):
        if word_index > 0:
            decoder_target_data[i, t, word_index] = 1.0

# --- Model Architecture ---
# Encoder
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, 256, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_output_length - 1,))
decoder_embedding = Embedding(output_vocab_size, 256, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention
attention_layer = AdditiveAttention()
attention_output = attention_layer([decoder_outputs, encoder_outputs])

# Concatenate & Output
decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention_output])
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_final_output = decoder_dense(decoder_concat)

# --- Compile & Train ---
model = Model([encoder_inputs, decoder_inputs], decoder_final_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    epochs=100,
    batch_size=2,
    verbose=1
)

# --- Plot ---
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()