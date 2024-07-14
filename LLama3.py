# First, install the required libraries
!pip install tiktoken matplotlib

import os
import requests
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

# Download necessary files
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Replace these URLs with the actual URLs of your model files
model_url = "https://example.com/Meta-Llama-3-8B/consolidated.00.pth"
tokenizer_url = "https://example.com/Meta-Llama-3-8B/tokenizer.model"
params_url = "https://example.com/Meta-Llama-3-8B/params.json"

download_file(model_url, "consolidated.00.pth")
download_file(tokenizer_url, "tokenizer.model")
download_file(params_url, "params.json")

tokenizer_path = "tokenizer.model"
special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",  # end of turn
] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

print(tokenizer.decode(tokenizer.encode("hello world!")))

model = torch.load("consolidated.00.pth", map_location=torch.device('cpu'))
print(json.dumps(list(model.keys())[:20], indent=4))

with open("params.json", "r") as f:
    config = json.load(f)
print(config)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)

embedding_layer = torch.nn.Embedding(vocab_size, dim)
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
norm_weights = model["tok_embeddings.weight"]
token_embeddings = token_embeddings_unnormalized / norm_weights.norm(dim=-1, keepdim=True)

rope_cache = torch.zeros(2, 2048, dim // 2, dtype=torch.bfloat16)
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
t = torch.arange(2048, dtype=torch.float)
freqs = torch.einsum("i,j->ij", t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
rope_cache[0, :, :dim // 2] = torch.sin(emb).to(torch.bfloat16)
rope_cache[1, :, :dim // 2] = torch.cos(emb).to(torch.bfloat16)

rms_norm = lambda x, w: x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm_eps) * w
final_embedding = token_embeddings

qkv_attention_store = []
for layer in range(n_layers):
    q_per_token = torch.matmul(final_embedding, model[f"layers.{layer}.attention.wq.weight"].T)
    k_per_token = torch.matmul(final_embedding, model[f"layers.{layer}.attention.wk.weight"].T)
    v_per_token = torch.matmul(final_embedding, model[f"layers.{layer}.attention.wv.weight"].T)
    q_per_token_rotated = torch.cat((q_per_token * rope_cache[0][:q_per_token.shape[0]], q_per_token * rope_cache[1][:q_per_token.shape[0]]), dim=-1)
    k_per_token_rotated = torch.cat((k_per_token * rope_cache[0][:k_per_token.shape[0]], k_per_token * rope_cache[1][:k_per_token.shape[0]]), dim=-1)
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (dim ** 0.5)
    mask = torch.full((len(token_embeddings), len(token_embeddings)), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=-1).to(torch.bfloat16)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
w_layer = model[f"layers.{layer}.attention.wo.weight"]
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
embedding_after_edit = final_embedding + embedding_delta
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
output_after_feedforward = torch.matmul(torch.nn.functional.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
final_embedding = embedding_after_edit + output_after_feedforward

final_embedding = rms_norm(final_embedding, model["norm.weight"])
print(final_embedding.shape)

print(model["output.weight"].shape)

logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
print(logits.shape)

next_token = torch.argmax(logits, dim=-1)
print(next_token)

print(tokenizer.decode([next_token.item()]))
