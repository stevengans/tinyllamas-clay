import mlx.core as mx
import argilla as rg
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from tinyllamascanjudge import load_model
gte_model = SentenceTransformer("thenlper/gte-small")


def few_and_zero_shot_generate(prompt, question=None):
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
        for token in model.generate(x, 1):
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

    if question is not None:
        out = generate(prompt.replace("{}", question))
        try:
            split = out.split("- ")
            return split[0], int(split[1][0])
        except:
            return None, None
    else:
        out = generate(prompt)
        return out


def red_loop(dataset_remote, generate=10000):
    records = list()
    for _ in tqdm(range(generate), desc="Red Teaming"):
        harmful_prompts = open("red_names.txt").read().strip()
        prompt = few_and_zero_shot_generate(harmful_prompts)
        response = few_and_zero_shot_generate(prompt)
        harmful_index_prompt = open("red.txt").read().strip()
        answer, rating = few_and_zero_shot_generate(harmful_index_prompt, prompt)
        if rating is None:
            continue
        record = rg.FeedbackRecord(
            fields={
                "prompt": prompt,
                "response": response
            },
            vectors={"rationale_embedding": gte_model.encode(response).tolist()[0:10]})
        record.suggestions = [
            {
                "question_name": "harmful_index",
                "value": int(rating),
                "agent": "TinyLlamas",
            },
            {
                "question_name": "harmful_rationale",
                "value": str(answer),
                "agent": "TinyLlamas"
            }
        ]
        records.append(record)
        if len(records) >= 20:
            dataset_remote.add_records(records)
            records = list()


if __name__ == "__main__":
    # initialize Argilla and return a dataset_remote
    model, tokenizer = load_model("mlx_model")
    red_loop(dataset_remote)
