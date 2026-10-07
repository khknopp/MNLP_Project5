import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def predict(self, src_sentence, max_length=20, verbose=True, **kwargs):
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

class SubwordTokenizer:
    """SentencePiece unigram model trained on the given sentences (in memory, no files). ids: pad 0, unk 1, bos 2, eos 3."""
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, sentences, vocab_size=4000):
        import io
        import sentencepiece as spm
        buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(sentences), model_writer=buffer, vocab_size=vocab_size, model_type="unigram",
            character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False, minloglevel=2)
        self._proto = buffer.getvalue()
        self._load()

    def _load(self):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_proto=self._proto)

    def __getstate__(self):
        return {"_proto": self._proto}

    def __setstate__(self, state):
        self._proto = state["_proto"]
        self._load()

    def encode(self, sentence):
        return self.sp.encode(sentence)

    def decode(self, ids):
        return self.sp.decode([i for i in ids if i > self.BOS])

    def __len__(self):
        return self.sp.get_piece_size()


class AttentionSeq2Seq(Model):
    """Stronger LSTM baseline: subword units, 2-layer bidirectional LSTM encoder, 2-layer LSTM decoder with
    Luong attention and input feeding, tied target embeddings, dropout, label smoothing, length-bucketed
    batches, gradient clipping, LR decay and dev-loss early stopping, and beam search at inference."""

    def __init__(self, src_tok, tgt_tok, embedding_dim=256, hidden_dim=256, layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.src_tok, self.tgt_tok = src_tok, tgt_tok
        self.layers, self.dropout_p = layers, dropout
        dec_dim = 2 * hidden_dim
        self.src_embedding = nn.Embedding(len(src_tok), embedding_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(len(tgt_tok), embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True, bidirectional=True,
                               dropout=dropout if layers > 1 else 0)
        self.decoder = nn.LSTM(2 * embedding_dim, dec_dim, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0)
        self.attn_proj = nn.Linear(dec_dim, dec_dim, bias=False)
        self.combine = nn.Linear(2 * dec_dim, embedding_dim)
        self.out_bias = nn.Parameter(torch.zeros(len(tgt_tok)))
        self.drop = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

    @classmethod
    def from_pairs(cls, pairs, vocab_size=4000, **kwargs):
        return cls(SubwordTokenizer([s for s, _ in pairs], vocab_size), SubwordTokenizer([t for _, t in pairs], vocab_size), **kwargs)

    # ---- network
    def encode(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.src_embedding(src)), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        join = lambda x: x.view(self.layers, 2, x.size(1), -1).permute(0, 2, 1, 3).reshape(self.layers, x.size(1), -1)
        return out, (join(h), join(c))

    def step(self, y_prev, feed, state, enc_out, mask):
        x = torch.cat([self.drop(self.tgt_embedding(y_prev)), feed], dim=-1).unsqueeze(1)
        out, state = self.decoder(x, state)
        h = out.squeeze(1)
        score = torch.bmm(enc_out, self.attn_proj(h).unsqueeze(2)).squeeze(2).masked_fill(~mask, float("-inf"))
        context = torch.bmm(F.softmax(score, dim=-1).unsqueeze(1), enc_out).squeeze(1)
        feed = self.drop(torch.tanh(self.combine(torch.cat([context, h], dim=-1))))
        return F.linear(feed, self.tgt_embedding.weight, self.out_bias), feed, state

    def forward(self, src, lengths, tgt_in):
        enc_out, state = self.encode(src, lengths)
        mask = src != 0
        feed = enc_out.new_zeros(src.size(0), self.embedding_dim)
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, feed, state = self.step(tgt_in[:, t], feed, state, enc_out, mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    # ---- training
    def _encode_pairs(self, pairs, max_len):
        return [(self.src_tok.encode(s)[:max_len - 1] + [SubwordTokenizer.EOS],
                 [SubwordTokenizer.BOS] + self.tgt_tok.encode(t)[:max_len - 2] + [SubwordTokenizer.EOS]) for s, t in pairs]

    def _batches(self, data, batch_size, shuffle):
        order = list(range(len(data)))
        if shuffle:
            random.shuffle(order)
        chunk = batch_size * 50
        batches = []
        for i in range(0, len(order), chunk):
            part = sorted(order[i:i + chunk], key=lambda j: len(data[j][0]))
            batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
        if shuffle:
            random.shuffle(batches)
        return batches

    def _tensors(self, data, idx, device):
        pad = lambda seqs: torch.tensor([x + [0] * (max(map(len, seqs)) - len(x)) for x in seqs], device=device)
        src = pad([data[j][0] for j in idx])
        tgt = pad([data[j][1] for j in idx])
        return src, torch.tensor([len(data[j][0]) for j in idx]), tgt

    def _loss(self, data, idx, device, smoothing):
        src, lengths, tgt = self._tensors(data, idx, device)
        logits = self(src, lengths, tgt[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=smoothing)

    def fit(self, pairs, dev_pairs=None, epochs=50, learning_rate=1e-3, batch_size=64, label_smoothing=0.1,
            clip=1.0, patience=6, max_len=100, device="cpu", verbose=True):
        self.to(device)
        train = self._encode_pairs(pairs, max_len)
        dev = self._encode_pairs(dev_pairs, max_len) if dev_pairs else None
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(epochs):
            self.train()
            total = 0.0
            for idx in self._batches(train, batch_size, True):
                optimizer.zero_grad()
                loss = self._loss(train, idx, device, label_smoothing)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
                total += loss.item()
            if dev is None:
                continue
            self.eval()
            with torch.no_grad():
                dev_loss = sum(self._loss(dev, idx, device, 0.0).item() * len(idx) for idx in self._batches(dev, batch_size, False)) / len(dev)
            scheduler.step(dev_loss)
            if verbose:
                print(f"epoch {epoch + 1} train {total / max(1, len(train) // batch_size):.3f} dev {dev_loss:.3f}", flush=True)
            if dev_loss < best:
                best, stale = dev_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)

    # ---- inference
    def predict(self, src_sentence, max_length=100, verbose=True, device="cpu", beam=5, alpha=0.6):
        self.to(device).eval()
        EOS, V = SubwordTokenizer.EOS, len(self.tgt_tok)
        ids = self.src_tok.encode(src_sentence)[:max_length] + [EOS]
        with torch.no_grad():
            src = torch.tensor([ids], device=device)
            enc_out, state = self.encode(src, torch.tensor([len(ids)]))
            enc_out = enc_out.expand(beam, -1, -1)
            mask = (src != 0).expand(beam, -1)
            state = tuple(x.repeat(1, beam, 1) for x in state)
            feed = enc_out.new_zeros(beam, self.embedding_dim)
            y = torch.full((beam,), SubwordTokenizer.BOS, device=device)
            scores = torch.zeros(beam, device=device)
            scores[1:] = float("-inf")
            seqs = [[] for _ in range(beam)]
            finished = []
            lp = lambda n: ((5 + n) / 6) ** alpha
            for t in range(max_length):
                logits, feed, state = self.step(y, feed, state, enc_out, mask)
                logp = F.log_softmax(logits, dim=-1)
                logp[:, [0, SubwordTokenizer.BOS]] = float("-inf")
                top, idx = (scores.unsqueeze(1) + logp).view(-1).topk(2 * beam)
                alive_src, alive_tok, alive_score = [], [], []
                for score, i in zip(top.tolist(), idx.tolist()):
                    b, w = divmod(i, V)
                    if score == float("-inf"):
                        continue
                    if w == EOS:
                        finished.append((score / lp(t + 1), seqs[b]))
                    elif len(alive_src) < beam:
                        alive_src.append(b); alive_tok.append(w); alive_score.append(score)
                if not alive_src or len(finished) >= beam:
                    break
                pad = beam - len(alive_src)  # fewer live candidates than beams: duplicate the best one with -inf
                alive_src += [alive_src[0]] * pad; alive_tok += [alive_tok[0]] * pad; alive_score += [float("-inf")] * pad
                sel = torch.tensor(alive_src, device=device)
                state = tuple(x[:, sel] for x in state)
                feed, y, scores = feed[sel], torch.tensor(alive_tok, device=device), torch.tensor(alive_score, device=device)
                seqs = [seqs[b] + [w] for b, w in zip(alive_src, alive_tok)]
            if not finished:
                finished = [(scores[0].item() / lp(len(seqs[0])), seqs[0])]
        return self.tgt_tok.decode(max(finished, key=lambda f: f[0])[1])

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