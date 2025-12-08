from typing import Any, Literal, Optional

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class MessageEmbeddingModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        message_context_length: int,
        pooling_mode: Literal['mean', 'attention'] = 'mean',
        use_lora: bool = False,
        lora_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self._base = AutoModel.from_pretrained(base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._pooling_mode: Literal['mean', 'attention'] = pooling_mode
        self._embedding_dim: int = self._base.config.hidden_size
        self._message_context_length: int = message_context_length
        self._use_lora: bool = use_lora
        self._lora_config: dict[str, Any] = {} if lora_config is None else lora_config
        # https://stackoverflow.com/questions/76547541/huggingface-how-do-i-find-the-max-length-of-a-model#comment136833899_77286207
        # self.token_context_length: int = self._tokenizer.model_max_length
        # self.token_context_length: int = self._base.config.max_position_embeddings
        self.token_context_length: int = 512

        self._old_vocab_size: int = len(self._tokenizer)
        new_tokens = ['<video>', '</video>', '<link/>', '<attachment/>', '</user>'] + [f"<user{i}>" for i in range(self._message_context_length)]
        self._num_new_tokens: int = self._tokenizer.add_tokens(new_tokens)
        self._base.resize_token_embeddings(len(self._tokenizer))

        self.adapter = NewTokenEmbeddingAdapter(
            new_tokens.__len__(),
            self._base.config.hidden_size,
        )
        if self._pooling_mode == 'attention':
            self.attention_query: nn.Linear = nn.Linear(self._embedding_dim, 1)

        if self._use_lora:
            config = LoraConfig(
                **self._lora_config,
            )
            self._base = get_peft_model(self._base, config)

        self._base = WrappedModel(self._base, self.adapter, self._old_vocab_size)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        last_hidden_state = self._base(
            input_ids,
            *args,
            attention_mask=attention_mask,
            **kwargs
        ).last_hidden_state
        pooler_out = self.pool(last_hidden_state, attention_mask)
        return pooler_out

    def pool(self, x, attn_mask):
        if self._pooling_mode == 'mean':
            if attn_mask is None:
                return x.mean(dim=1)
            else:
                mask = attn_mask.unsqueeze(-1).to(x.dtype)  # [B, seq_len, 1]
                return (x * mask).sum(dim=1) / mask.sum(dim=1)
        elif self._pooling_mode == 'attention':
            scores = self.attention_query(x).squeeze(-1)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, float('-inf'))

            weights = F.softmax(scores, dim=-1)
            pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
            return pooled
        else:
            raise NotImplementedError(f'{self._pooling_mode} is not a valid pooling mode!')

    def get_param_groups(self) -> dict[str, Any]:
        # input_embeddings = self._base.get_input_embeddings()
        base_model_params = [p for n, p in self._base.named_parameters()]
        additional_params = []  # list(self.adapter.parameters())
        if self._pooling_mode == 'attention':
            additional_params.extend(self.attention_query.parameters())

        return {
            'base': base_model_params,
            'additional': additional_params
        }

    @property
    def tokenizer(self):
        return self._tokenizer


class NewTokenEmbeddingAdapter(torch.nn.Module):
    def __init__(self, num_new_tokens: int, d_model: int):
        super().__init__()
        self.new_emb: nn.Embedding = nn.Embedding(
            num_new_tokens,
            d_model,
        )

    def forward(self, new_token_ids: torch.Tensor):
        return self.new_emb(new_token_ids)


class WrappedModel(torch.nn.Module):
    def __init__(self, base_model, adapter, old_vocab_size):
        super().__init__()
        self.base = base_model
        self.adapter = adapter
        self.old_vocab_size = old_vocab_size

    def forward(self, input_ids, *args, **kwargs):
        embed = self.base.get_input_embeddings()(input_ids)

        mask = input_ids >= self.old_vocab_size
        if mask.any():
            new_ids = input_ids[mask] - self.old_vocab_size
            embed[mask] = self.adapter(new_ids)

        return self.base(inputs_embeds=embed, *args, **kwargs)
