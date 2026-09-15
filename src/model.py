import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define some helper words
PADDING, START, END, UNKOWN = "<PAD>", "<START>", "<END>", "<UNK>"

class Vocabulary:
    def __init__(self, all_sentences):
        all_tokens = [token for sentence in all_sentences for token in sentence.split()]
        self.w2i = {PADDING: 0, START: 1, END: 2, UNKOWN: 3} 
        self.w2i.update({token: i + 4 for i, token in enumerate(sorted(list(set(all_tokens))))})
        self.i2w = {i: w for w, i in self.w2i.items()}

    def encode_sentence(self, sentence):
        return [self.w2i.get(token, self.w2i[UNKOWN]) for token in sentence.split()]

    def decode_sentence(self, indices):
        return " ".join([self.i2w.get(i, UNKOWN) for i in indices])

    def __len__(self):
        return len(self.w2i)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, pairs):
        raise NotImplementedError

    def predict(self, sentence):
        raise NotImplementedError

class Seq2SeqModel(Model):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_embedding = nn.Embedding(len(src_vocab), embedding_dim)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(tgt_vocab))

    def forward(self, src, tgt):
        _, (hidden, cell) = self.encoder(self.src_embedding(src))
        outputs, _ = self.decoder(self.tgt_embedding(tgt), (hidden, cell))
        return self.fc(outputs)

    def fit(self, pairs, epochs=10, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for _ in tqdm(range(epochs)):
            random.shuffle(pairs)
            total_loss = 0
            for src_sentence, tgt_sentence in pairs:
                src_ids = torch.tensor(self.src_vocab.encode_sentence(src_sentence) + [self.src_vocab.w2i[END]]).unsqueeze(0)
                tgt_ids = torch.tensor([self.tgt_vocab.w2i[START]] + self.tgt_vocab.encode_sentence(tgt_sentence) + [self.tgt_vocab.w2i[END]]).unsqueeze(0)

                optimizer.zero_grad()
                output = self.forward(src_ids, tgt_ids[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt_ids[:, 1:].reshape(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def predict(self, src_sentence, max_length=20):
        self.eval()
        with torch.no_grad():
            src_ids = torch.tensor(self.src_vocab.encode_sentence(src_sentence) + [self.src_vocab.w2i[END]]).unsqueeze(0)
            _, (hidden, cell) = self.encoder(self.src_embedding(src_ids))

            tgt_ids = [self.tgt_vocab.w2i[START]]
            decoder_input = torch.tensor([[self.tgt_vocab.w2i[START]]])
            for _ in tqdm(range(max_length)):
                output, (hidden, cell) = self.decoder(self.tgt_embedding(decoder_input), (hidden, cell))
                next_token_logits = self.fc(output[:, -1, :])
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                if next_token_id == self.tgt_vocab.w2i[END]:
                    break
                tgt_ids.append(next_token_id)
                decoder_input = torch.tensor(tgt_ids).unsqueeze(0)

        return self.tgt_vocab.decode_sentence(tgt_ids[1:])