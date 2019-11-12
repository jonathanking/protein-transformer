import torch
import numpy as np

class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled, softmax attention module for Transformer as defined by
    Attention(Q, K, V) on pg 4. Returns the final attention vectors as well as
    the attention matrices (pairwise scores). """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dropout=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / np.sqrt(K.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = self.softmax(scores)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, V), scores

class MultiHeadedAttention(torch.nn.Module):
    """
    Multi-headed attention layer for the Transformer model. Wraps
    ScaledDotProductAttention. Assumes n_heads are applied by splitting up
    model in to n_heads, each of size dm / n_heads. Guided by
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, dm, n_heads, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert dm % n_heads == 0, "The dimension of the model must be evenly divisible by the number of attn heads."
        self.dm = dm
        self.dk = dm // n_heads
        self.n_heads = n_heads

        self.wq = torch.nn.Linear(self.dm, self.dm)
        self.wk = torch.nn.Linear(self.dm, self.dm)
        self.wv = torch.nn.Linear(self.dm, self.dm)
        self.wo = torch.nn.Linear(self.dm, self.dm)

        self.attn_scores = None
        self.attn = ScaledDotProductAttention()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, preQ, preK, preV, mask=None):
        n_batch = preQ.shape[0]
        Q, K, V = self.wq(preQ), self.wk(preK), self.wv(preV)

        # Split into heads, being careful to keep batch, head dims in front of L and dk dims.
        # Q                         = [ n_batch x L x dm ] = [ n_batch x L x (n_heads * dk) ]
        # Q.view                    = [ n_batch x L x n_heads x dk ]
        # Q.transpose               = [ n_batch x n_heads x L x dk ]
        Q, K, V = (x.view(n_batch, -1, self.n_heads, self.dk).transpose(1, 2) for x in (Q, K, V))

        # Apply scaled dot-product attention across batch, head dims. Add head dim to mask for broadcasting.
        # attn_output               = [ n_batch x n_heads x L x dk ]
        mask = mask.unsqueeze(1) if mask is not None else None
        attn_output, self.attn_scores = self.attn(Q, K, V, mask, self.dropout)

        # Concatenate output from attn heads
        # attn_output.transpose     = [ n_batch x L x n_heads x dk ]
        # attn_output.contiguous    = [ (n_batch * L * n_heads * dk) ]
        # attn_output.view          = [ n_batch x L x dm ]
        # Because transpose is a view operation, we call contiguous to be sure our next view operation is valid.
        # See https://stackoverflow.com/questions/48915810/pytorch-contiguous for notes.
        attn_output = attn_output.transpose(1, 2).contiguous().view(n_batch, -1, self.dm)
        return self.wo(attn_output)


if __name__ == "__main__":
    dm = 128
    seq = torch.zeros(8, 31, dm)
    mhattn = MultiHeadedAttention(dm, 4)
    out = mhattn(seq)
    print(out.shape)