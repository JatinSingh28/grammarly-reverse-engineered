import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb

# Other necessary imports

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define configurations
epochs = 20
batch_size = 32
validation_split = 0.2
model_name = "baseModelV1"
n_rows = 1000

wandb.init(
    # set the wandb project where this run will be logged
    project="grammarly-reverse-engineered-base-model",

    # track hyperparameters and run metadata with wandb.config
    config={
        "data:":"C4 200M",
        "model": model_name,
        "files_taken":1,
        "n_rows":n_rows,
        "latent_dim":256,
        "epoch": epochs,
        "batch_size": batch_size,
    }
)

config = wandb.config

# Example dataset with incorrect and correct sentences
df = pd.read_csv("./data/data1.csv",nrows=n_rows)

incorrect_sentences = df['input']
correct_sentences = df['target']
# incorrect_sentences = [
#     'She is go too school.',
#     'He hav a red car.',
#     'I am alwayz happy.'
# ]
# correct_sentences = [
#     'She goes to school.',
#     'He has a red car.',
#     'I am always happy.'
# ]



# Generate vocabulary sets
input_characters = set(' '.join(incorrect_sentences + correct_sentences))
target_characters = input_characters

# Generate character-level index mappings
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = input_token_index

# Define max sequence lengths
max_encoder_seq_length = max([len(txt) for txt in incorrect_sentences])
max_decoder_seq_length = max([len(txt) for txt in correct_sentences])

dataDict = {
    "input_characters":input_characters,
    "target_characters":target_characters,
    "input_token_index":input_token_index,
    "target_token_index":target_token_index,
    "max_encoder_seq_length":max_encoder_seq_length,
    "max_decoder_seq_length":max_decoder_seq_length,
}

file_path = "dataDict"
with open(file_path, 'wb') as file:
    pickle.dump(dataDict, file)

# Create encoder and decoder data
encoder_input_data = np.zeros((len(incorrect_sentences), max_encoder_seq_length, len(input_characters)), dtype='float32')
decoder_input_data = np.zeros((len(incorrect_sentences), max_decoder_seq_length, len(target_characters)), dtype='float32')
decoder_target_data = np.zeros((len(incorrect_sentences), max_decoder_seq_length, len(target_characters)), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(incorrect_sentences, correct_sentences)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# Define Seq2Seq model
latent_dim = 256  # dimensionality of the encoding space


# Define Seq2Seq model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        output = self.embedding(input).unsqueeze(0)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.softmax(self.out(output[0]))
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        outputs = torch.zeros(decoder_input.shape[0], decoder_input.shape[1], len(target_characters)).to(device)
        
        for t in range(decoder_input.shape[1]):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input[:, t], decoder_hidden, decoder_cell)
            outputs[:, t] = decoder_output.squeeze(1)
        
        return outputs

# Initialize the models
encoder = Encoder(len(input_characters), latent_dim).to(device)
decoder = Decoder(latent_dim, len(target_characters)).to(device)
model = Seq2Seq(encoder, decoder).to(device)
wandb.watch(model)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Convert data to PyTorch tensors
encoder_input_data = torch.tensor(encoder_input_data, dtype=torch.long).to(device)
decoder_input_data = torch.tensor(decoder_input_data, dtype=torch.long).to(device)
decoder_target_data = torch.tensor(decoder_target_data.argmax(axis=2), dtype=torch.long).to(device)

# Create DataLoader for training
train_data = torch.utils.data.TensorDataset(encoder_input_data, decoder_input_data, decoder_target_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (encoder_input, decoder_input, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(encoder_input, decoder_input)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        targets = targets.view(-1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    wandb.log(epoch_loss)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")

# Save model
model.to_onnx()
wandb.save("model.onnx")
torch.save(model.state_dict(), f"models/{model_name}.pt")
