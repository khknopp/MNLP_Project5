import random
from pathlib import Path

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
    def __init__(self, src_vocab, tgt_vocab, embedding_dim=256, hidden_dim=512, **kwargs):
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

    def fit(self, pairs, epochs=10, learning_rate=0.001, device="cpu"):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.to(device)

        for _ in tqdm(range(epochs)):
            random.shuffle(pairs)
            total_loss = 0
            for src_sentence, tgt_sentence in pairs:
                src_ids = torch.tensor(self.src_vocab.encode_sentence(src_sentence) + [self.src_vocab.w2i[END]]).unsqueeze(0)
                tgt_ids = torch.tensor([self.tgt_vocab.w2i[START]] + self.tgt_vocab.encode_sentence(tgt_sentence) + [self.tgt_vocab.w2i[END]]).unsqueeze(0)

                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)

                optimizer.zero_grad()
                output = self.forward(src_ids, tgt_ids[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt_ids[:, 1:].reshape(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def predict(self, src_sentence, max_length=20, verbose=True, device="cpu"):
        self.eval()
        with torch.no_grad():

            src_ids = torch.tensor(self.src_vocab.encode_sentence(src_sentence) + [self.src_vocab.w2i[END]]).unsqueeze(0)
            src_ids = src_ids.to(device)

            src_embedding = self.src_embedding.to(device)
            tgt_embedding = self.tgt_embedding.to(device)

            _, (hidden, cell) = self.encoder(src_embedding(src_ids))

            tgt_ids = [self.tgt_vocab.w2i[START]]
            decoder_input = torch.tensor([[self.tgt_vocab.w2i[START]]])

            for _ in tqdm(range(max_length), disable=not verbose):
                decoder_input = decoder_input.to(device)
                output, (hidden, cell) = self.decoder(tgt_embedding(decoder_input), (hidden, cell))
                next_token_logits = self.fc(output[:, -1, :])
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                if next_token_id == self.tgt_vocab.w2i[END]:
                    break
                tgt_ids.append(next_token_id)
                decoder_input = torch.tensor(tgt_ids).unsqueeze(0)

        return self.tgt_vocab.decode_sentence(tgt_ids[1:])

class CopyModel(Model):
    def __init__(self, src_vocab, tgt_vocab, **kwargs):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def fit(self, pairs, **kwargs):
        pass

    def predict(self, src_sentence, max_length=20, verbose=True):
        return src_sentence

NLLB_CODES = {"en": "eng_Latn", "az": "azj_Latn", "be": "bel_Cyrl",
              "tr": "tur_Latn", "uk": "ukr_Cyrl", "ru": "rus_Cyrl"}

class NLLBModel(Model):
    def __init__(self, src_vocab=None, tgt_vocab=None, src_lang="en", tgt_lang="az",
                 model_name="facebook/nllb-200-distilled-600M", beams=5, **kwargs):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.src_lang = NLLB_CODES.get(src_lang, src_lang)
        self.tgt_lang = NLLB_CODES.get(tgt_lang, tgt_lang)
        self.beams = beams

    def fit(self, pairs, **kwargs):
        pass

    def predict(self, src_sentence, max_length=128, verbose=True):
        self.hf_model.eval()
        self.tokenizer.src_lang = self.src_lang
        bos = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        enc = self.tokenizer([src_sentence], return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            gen = self.hf_model.generate(**enc, forced_bos_token_id=bos, num_beams=self.beams, max_new_tokens=max_length)
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

MODELS = {"seq2seq": Seq2SeqModel, "copy": CopyModel, "nllb": NLLBModel}

def build_model(name, src_vocab, tgt_vocab, **kwargs):
    return MODELS[name](src_vocab, tgt_vocab, **kwargs)

def translate_corpus(model, sources, max_length=20, device="cpu"):
    return [model.predict(src, max_length=max_length, verbose=False, device=device) for src in tqdm(sources)]

def save_translations(out_dir, direction, sources, references, hypotheses):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, lines in (("src", sources), ("ref", references), ("hyp", hypotheses)):
        (out_dir / f"{direction}.{ext}").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")