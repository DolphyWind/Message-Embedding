import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, RobertaTokenizerFast
from sentence_transformers import SentenceTransformer, util


def show_params(model: nn.Module):
    total_params: int = 0
    for name, params in model.named_parameters():
        p: int = params.numel()
        print(f"{name} has {p} parameters.")
        total_params += p

    print(f"\nTotal model parameters: {total_params}.")


@torch.no_grad
def inf_message_loop(tokenizer: RobertaTokenizerFast, model: nn.Module, device: torch.device):
    while True:
        msg = input("> ")
        if not msg:
            break
        token_ids: list[int] = tokenizer.encode(msg)
        decoded: list[str] = [tokenizer.decode(token) for token in token_ids]
        token_list: list[str] = [f"{t}:{i}" for i, t in zip(token_ids, decoded)]
        print(f"Tokens are: {token_list}")

        model_in: torch.Tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        out: torch.Tensor = model(model_in)
        lhs: torch.Tensor = out.last_hidden_state
        po: torch.Tensor = out.pooler_output
        print(f"Embedded vector shape: {lhs.shape} {po.shape}")


@torch.no_grad
def embedding_test(tokenizer: RobertaTokenizerFast, model: nn.Module, device: torch.device):
    sentences = ['kral', 'kralice', 'erkek', 'kadın']
    tokenizer_out: dict[str, torch.Tensor] = tokenizer(sentences, padding=True, return_tensors='pt')
    token_ids: torch.Tensor = tokenizer_out['input_ids'].to(device)
    attn_mask: torch.Tensor = tokenizer_out['attention_mask'].to(device)
    model_in: torch.Tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    out = model(token_ids, attn_mask).pooler_output
    diff_1 = (out[0] - out[1]).unsqueeze(0)
    diff_2 = (out[2] - out[3]).unsqueeze(0)
    sim: float = F.cosine_similarity(diff_1, diff_2).item()
    print(f"Similarity score: {sim}")


@torch.no_grad
def real_embedding_test(tokenizer: RobertaTokenizerFast, model: nn.Module, device: torch.device):
    sentences = [
        'istanbul, boğaziçi köprüsüyle iki kıtayı birleştiren, tarihi ve kültürel zenginlikleriyle ünlü bir şehirdir.',
        'istanbul boğaziçi'
    ]
    # tokenizer_out: dict[str, torch.Tensor] = tokenizer(sentences, padding=True, return_tensors='pt')
    # token_ids: torch.Tensor = tokenizer_out['input_ids'].to(device)
    # attn_mask: torch.Tensor = tokenizer_out['attention_mask'].to(device)
    # model_in: torch.Tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    # attn_mask_unsq: torch.Tensor = attn_mask.unsqueeze(-1)
    # out = (out * attn_mask_unsq).sum(dim=1) / attn_mask_unsq.sum(dim=1)
    # v1 = out[0].unsqueeze(0)
    # v2 = out[1].unsqueeze(0)
    out = model.encode(sentences)
    sim: float = util.cos_sim(out[0], out[1]).item()
    print(f"Similarity score: {sim}")


def main():
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    MODEL_NAME: str = 'TURKCELL/roberta-base-turkish-uncased'
    MODEL_NAME: str = 'dbmdz/bert-base-turkish-128k-uncased'
    MODEL_NAME: str = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'
    # https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr
    tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model: nn.Module = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model: nn.Module = SentenceTransformer(MODEL_NAME)

    # show_params(model)
    # inf_message_loop(tokenizer, model, device)
    # embedding_test(tokenizer, model, device)
    real_embedding_test(tokenizer, model, device)


if __name__ == '__main__':
    main()
