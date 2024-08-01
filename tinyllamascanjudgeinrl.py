.import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor
from clay import init_clay
from tqdm import tqdm


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        x = self.tok_embeddings(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, x, temp=1.0):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))
        cache = []
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)
        x = self.tok_embeddings(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            cache.append(c)
        x = self.norm(x)
        y = self.output(x[:, -1])
        y = sample(y)
        yield y
        while True:
            x = y[:, None]
            x = self.tok_embeddings(x)
            for i in range(len(cache)):
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = sample(self.output(x[:, -1]))

            yield y


def few_shot_generate(prompt, question):
    def possible_end(s):
        word = "[Instruction]"
        for i in range(len(word) - 1, 0, -1):
            if s[-i:] == word[:i]:
                return 0
        if s[-len(word):] == word:
            return 1
        return -1

    def generate(string):
        x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(string)])
        tokens = []
        for token in model.generate(x, 0):
            tokens.append(token)
            if len(tokens) == 1:
                mx.eval(token)
            if len(tokens) >= 100:
                break
            mx.eval(tokens)
            token_list = [t.item() for t in tokens]
            s = tokenizer.decode(token_list)
            end = possible_end(s)
            if end == 0:
                continue
            if end == 1:
                break
            if token_list[-1] == tokenizer.eos_id():
                break
        mx.eval(tokens)
        s = tokenizer.decode([t.item() for t in tokens])
        return s

    out = generate(prompt.replace("{}", question))
    try:
        split = out.split("- ")
        return split[0], int(split[1][0])
    except:
        return None, None

def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config


def load_model(model_path):
    # find the TinyLlama weights here: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main
    model_path = Path(model_path)
    unsharded_weights_path = Path(model_path / "weights.npz")
    if unsharded_weights_path.is_file():
        weights = mx.load(str(unsharded_weights_path))
    else:
        sharded_weights_glob = str(model_path / "weights.*.npz")
        weight_files = glob.glob(sharded_weights_glob)
        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found in {}".format(model_path))
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf).items())
    with open(model_path / "config.json", "r") as f:
        config = sanitize_config(json.loads(f.read()), weights)
        quantization = config.pop("quantization", None)
    model = Llama(ModelArgs(**config))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    return model, tokenizer


def tiny_llama_model(prompt, to_judge, seed):
    mx.random.seed(seed)
    answer, rating = few_shot_generate(prompt, to_judge)
    return answer, rating


def tiny_llama_inference_loop(prompt, num_llamas=10):
    prompt = open('sample.txt').read().strip()
    llamas = [random.randint(1, 100) for _ in range(num_llamas)]
    final_answer, final_rating = None, 0
    for i in range(num_llamas):
        answer, rating = tiny_llama_model(prompt, to_judge, llamas[i])
        if int(rating) > final_rating:
            final_answer = answer
            final_rating = rating
    return final_answer


if __name__ == "__main__":
    model, tokenizer = load_model("mlx_model")
    answer = tiny_llama_inference_loop("prompt_goes_here", num_llamas=5)
    print(answer)
