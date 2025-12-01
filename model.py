import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Literal, Any


class MessageEmbeddingModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        pooling_mode: Literal['mean', 'attention'],
        message_context_length: int,
    ) -> None:
        super().__init__()

        self._underlying_model: AutoModel = AutoModel.from_pretrained(base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._pooling_mode: Literal['mean', 'attention'] = pooling_mode
        self._embedding_dim: float = self._underlying_model.config.hidden_size
        self._message_context_length: int = message_context_length
        self.token_context_length: int = self._tokenizer.model_max_length

        new_tokens = ['<video>', '</video>', '<link/>', '<attachment/>', '</user>'] + [f"<user{i}>" for i in range(self._message_context_length)]
        self._num_new_tokens: int = self._tokenizer.add_tokens(new_tokens)
        self._underlying_model.resize_token_embeddings(len(self._tokenizer))

        if self._pooling_mode == 'attention':
            self.attention_query: nn.Linear = nn.Linear(self._embedding_dim, 1)

    def forward(self, x, attn_mask, *args, **kwargs):
        last_hidden_state = self._underlying_model(x, attn_mask, *args, **kwargs).last_hidden_state
        pooler_out = self.pool(last_hidden_state, attn_mask)
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
        # input_embeddings = self._underlying_model.get_input_embeddings()
        old_params = [p for n, p in self._underlying_model.named_parameters() if 'attention_query' not in n]
        new_params = []
        if self._pooling_mode == 'attention':
            new_params = [self.attention_query.parameters()]

        return {
            'old': old_params,
            'new': new_params
        }

    @property
    def tokenizer(self):
        return self._tokenizer
