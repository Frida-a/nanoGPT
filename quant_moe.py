# adapted from: https://github.com/antonio-f/mixture-of-experts-from-scratch/blob/main/moe.ipynb
import torch.nn as nn
from torch.nn import functional as F
import torch
import inspect
from dataclasses import dataclass
# from quantization import MXFp4QuantLinear
from bitsandbytes.nn import Linear8bitLt


class MoeLayer(nn.Module):
    def __init__(self, experts, gate, k=1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.k = k

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        weights, selected_experts = torch.topk(
            gate_logits, self.k
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)

class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout, quant_format):
        super().__init__()
        # if quant_format == 'mxfp4':
        #     self.key = MXFp4QuantLinear(n_embed, head_size, bias = False)
        #     self.query = MXFp4QuantLinear(n_embed, head_size, bias = False)
        #     self.value = MXFp4QuantLinear(n_embed, head_size, bias = False)
        if quant_format == '8bit':
            self.key = Linear8bitLt(n_embed, head_size, bias = False, has_fp16_weights=False)
            self.query = Linear8bitLt(n_embed, head_size, bias = False, has_fp16_weights=False)
            self.value = Linear8bitLt(n_embed, head_size, bias = False, has_fp16_weights=False)
        else:
            self.key = nn.Linear(n_embed, head_size, bias = False)
            self.query = nn.Linear(n_embed, head_size, bias = False)
            self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MulitHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout, quant_format):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout, quant_format) for _ in range(num_heads)])
        # if quant_format == 'mxfp4':
        #     self.proj = MXFp4QuantLinear(n_embed, n_embed)
        if quant_format == '8bit':
            self.proj = Linear8bitLt(n_embed, n_embed, has_fp16_weights=False)
        else:
            self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x =  torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout, quant_format):
        super().__init__()
        # if quant_format == 'mxfp4':
        #     self.net = nn.Sequential(
        #         MXFp4QuantLinear(n_embed, 4* n_embed),
        #         nn.ReLU(),
        #         MXFp4QuantLinear(4 * n_embed, n_embed),
        #     nn.Dropout(dropout))
        if quant_format == '8bit':
            self.net = nn.Sequential(
                Linear8bitLt(n_embed, 4* n_embed, has_fp16_weights=False),
                nn.ReLU(),
                Linear8bitLt(4 * n_embed, n_embed, has_fp16_weights=False),
            nn.Dropout(dropout))
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embed, 4* n_embed),
                nn.ReLU(),
                nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_head= MulitHeadAttention(config.n_head, config.n_embed//config.n_head, config.n_embed, config.block_size, config.dropout, config.quant_format)
        if config.quantize_gate:
            # if config.quant_format == 'mxfp4':
            #     self.ffw = MoeLayer(
            #         experts=[FeedForward(config.n_embed, config.dropout, config.quant_format) for _ in range(config.num_experts)],
            #         gate=MXFp4QuantLinear(config.n_embed, config.num_experts, bias=False),
            #     )
            if config.quant_format == '8bit':
                self.ffw = MoeLayer(
                    experts=[FeedForward(config.n_embed, config.dropout, config.quant_format) for _ in range(config.num_experts)],
                    gate=Linear8bitLt(config.n_embed, config.num_experts, bias=False, has_fp16_weights=False),
                )
        else:
            self.ffw = MoeLayer(
                experts=[FeedForward(config.n_embed, config.dropout, config.quant_format) for _ in range(config.num_experts)],
                gate=nn.Linear(config.n_embed, config.num_experts, bias=False),
            )

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

@dataclass
class MOEConfig:
    block_size: int = 256
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # moe config
    n_embed: int = 384
    device: str = 'cuda'
    num_experts: int = 4
    quant_format: str = None
    quantize_gate: bool = False
    quantize_head: bool = False


class MOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed, device=config.device)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed, device=config.device)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        if config.quantize_head:
            # if config.quant_format == 'mxfp4':
            #     self.lm_head = MXFp4QuantLinear(config.n_embed, config.vocab_size)
            if config.quant_format == '8bit':
                self.lm_head = Linear8bitLt(config.n_embed, config.vocab_size, has_fp16_weights=False)
        else:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.config = config
        self.device = config.device


    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T).to(self.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # TODO: chech this is correct for MOE
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
  
    @torch.no_grad()
    def generate(self, idx, max_new_tokes):
        for _ in range(max_new_tokes):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):

        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised

        return mfu


