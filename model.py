from typing import Any, Literal, Optional

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


type PoolingType = Literal['max', 'mean', 'attention']


class MessageEmbeddingModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        message_context_length: int,
        token_context_length: int,
        pooling_mode: PoolingType = 'mean',
        use_lora: bool = False,
        lora_config: Optional[dict[str, Any]] = None,
        initialize_new: bool = False,
    ) -> None:
        super().__init__()

        if initialize_new:
            self._base = AutoModel.from_config(AutoConfig.from_pretrained(base_model))
        else:
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
        self.token_context_length: int = token_context_length

        self._old_vocab_size: int = len(self._tokenizer)
        new_tokens = ['<video>', '</video>', '<link/>', '<attachment/>', '</user>'] + [f"<user{i}>" for i in range(self._message_context_length)]
        self._num_new_tokens: int = self._tokenizer.add_tokens(new_tokens)
        self._base.resize_token_embeddings(len(self._tokenizer))

        self.adapter = NewTokenEmbeddingAdapter(
            new_tokens.__len__(),
            self._base.config.hidden_size,
        )
        if self._pooling_mode == 'attention':
            self.pooler = AttentionPoolingModule(self._embedding_dim)
        elif self._pooling_mode == 'mean':
            self.pooler = MeanPoolingModule()
        elif self._pooling_mode == 'max':
            self.pooler = MaxPoolingModule()

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
        pooler_out = self.pooler(last_hidden_state, attention_mask)
        return pooler_out

    def get_param_groups(self) -> dict[str, Any]:
        base_model_params = [p for n, p in self._base.named_parameters()]
        additional_params = []
        if self._pooling_mode == 'attention':
            additional_params.extend(self.pooler.parameters())

        return {
            'base': base_model_params,
            'additional': additional_params
        }

    def unwrap_base(self) -> None:
        if not isinstance(self._base, WrappedModel):
            raise RuntimeError("Base model is already unwrapped")

        inner_base = self._base.base
        base_emb = self._base.base.get_input_embeddings()
        adapter_weights = self.adapter.new_emb.weight.data.to(
            device=base_emb.weight.device,
            dtype=base_emb.weight.dtype
        )

        base_emb.weight.data[self._old_vocab_size:] = adapter_weights
        self._base = inner_base
        del self.adapter

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class NewTokenEmbeddingAdapter(nn.Module):
    def __init__(self, num_new_tokens: int, d_model: int):
        super().__init__()
        self.new_emb: nn.Embedding = nn.Embedding(
            num_new_tokens,
            d_model,
        )

    def forward(self, new_token_ids: torch.Tensor):
        return self.new_emb(new_token_ids)


class WrappedModel(nn.Module):
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


class MeanPoolingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_mask):
        mask = attn_mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1)


class MaxPoolingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_mask):
        x = x.masked_fill(attn_mask == 0, float('-inf'))
        return x.max(dim=1).values


class AttentionPoolingModule(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention_query: nn.Linear = nn.Linear(embedding_dim, 1)

    def forward(self, x, attn_mask):
        scores = self.attention_query(x).squeeze(-1)
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled
