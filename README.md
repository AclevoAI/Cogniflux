# Cogniflux

An open-source framework for training a ~1.2B parameter transformer with **reasoning baked into the architecture** — not prompted in, not wrapped in chain-of-thought scaffolding, but built directly into the model's modules.

The design is inspired by the human prefrontal cortex (PFC): the brain region responsible for working memory, executive control, inhibitory filtering, and forward planning. Each of those functions has a direct architectural counterpart in Cogniflux.

Built on PyTorch with ROCm for dual AMD GPU training.

---

## Architecture

### Core Transformer
| Component | Choice | Reason |
|---|---|---|
| Attention | Grouped Query Attention (16 Q / 4 KV heads) | 4× KV cache reduction with no quality loss |
| Position encoding | RoPE (rotary) | Length generalisation, zero learned position params |
| FFN activation | SwiGLU | ~10% better than ReLU at equivalent compute |
| Normalisation | RMSNorm | Faster than LayerNorm, equally stable |
| Flash Attention | `F.scaled_dot_product_attention` | Native ROCm support in PyTorch 2.x |
| Precision | BFloat16 | Preferred over FP16 on AMD GPUs (no overflow) |

### PFC Modules

Five PFC module blocks are interleaved throughout the transformer stack (every 4 layers). Each block adds four neuro-inspired components:

```
Transformer Layer
├── RMSNorm → GQA Attention
│   └── [PFC block every 4 layers]
│       ├── WorkingMemoryModule     ← dorsolateral PFC
│       └── ExecutiveControlModule  ← anterior cingulate cortex
├── RMSNorm → SwiGLU FFN
│   └── [PFC block every 4 layers]
│       └── InhibitoryGate          ← orbitofrontal cortex
└── [Top 4 layers only]
    └── PlanningModule              ← ventromedial PFC
```

**WorkingMemoryModule** — A 64-slot cross-attention buffer that persists across the sequence. Tokens read from and write back to this buffer each layer, giving the model a scratchpad for multi-step reasoning. Analogous to the ~7-item working memory capacity of the human dorsolateral PFC.

**ExecutiveControlModule** — A learned gate that routes between two paths: the standard attention output and a second cross-attention pass over working memory. The gate decides how much to rely on direct context versus stored reasoning state. Analogous to the anterior cingulate's top-down attentional control.

**InhibitoryGate** — A sigmoid mask applied to FFN outputs, conditioned on the current working memory state. Suppresses associations that are irrelevant to the active reasoning context. Analogous to the orbitofrontal cortex's inhibitory projections.

**PlanningModule** — Applied only to the top 4 layers. Runs the representation through 3 iterative mini-transformer refinement steps in activation space. The model effectively "thinks harder" about its answer without generating extra tokens. Analogous to the ventromedial PFC's forward simulation during planning.

### Parameter Count
| Component | Params |
|---|---|
| Transformer (20 layers, hidden=2048) | ~902M |
| Embeddings (vocab=50304, tied in/out) | ~103M |
| PFC modules (5 blocks) | ~240M |
| **Total** | **~1.245B** |

---

## Hardware Requirements

- **2× AMD GPUs, 16 GB VRAM each** (tested on RX 7600 XT + RX 9060 XT)
- ROCm 6.x or 7.x
- PyTorch 2.1+ with ROCm build
- ~60 GB disk for full dataset shards

Single GPU works too — remove `--nproc_per_node=2` from the train command.

---

## Installation

```bash
# 1. Install PyTorch with ROCm (get the right wheel for your ROCm version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

# 2. Clone and install
git clone https://github.com/Codalorian/Cogniflux.git
cd Cogniflux
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1. Prepare training data

```bash
# Download and tokenise the CodeX-2M-Thinking reasoning dataset (~4.8B tokens)
python scripts/prepare_data.py --datasets codex_thinking --out_dir data/train_shards
```

Each sample is formatted as:

```
<|system|>
You are a high capacity reasoning agent called Cogniflux.
<|user|>
{problem}
<|assistant|>
<think>
{reasoning chain}
</think>
{answer}
<|endoftext|>
```

To build a larger 25–30B token mix, add more datasets:

```bash
python scripts/prepare_data.py \
  --datasets codex_thinking openwebtext2 pile_stackexchange pile_arxiv wikipedia \
  --out_dir data/train_shards
```

Available datasets: `codex_thinking`, `openwebtext2`, `pile_stackexchange`, `pile_arxiv`, `wikipedia`, `dm_mathematics`

### 2. Train

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 torchrun --nproc_per_node=2 scripts/train.py --config configs/cogniflux_1b.yaml
```

Training logs every 10 steps:

```
step=    10 | loss=10.82 | lr=6.00e-06 | tok/s=38,000 | dt=5.2s
step=    20 | loss=9.41  | lr=1.20e-05 | tok/s=39,000 | dt=5.1s
...
step=  5000 | loss=2.81  | lr=3.00e-05 | tok/s=40,000 | dt=5.0s
```

Checkpoints are saved every 1,000 steps to `checkpoints/cogniflux_1b/`.

### 3. Resume a stopped run

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/cogniflux_1b.yaml \
  --resume checkpoints/cogniflux_1b/step_0001000.pt
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/cogniflux_1b/step_0005000.pt \
  --config configs/cogniflux_1b.yaml
```

Outputs perplexity on the eval shard and sample generations for qualitative reasoning inspection.

### 5. Inference

```python
import torch
from cogniflux.model import CognifluxConfig, CognifluxTransformer
from cogniflux.data import encode, decode

cfg   = CognifluxConfig()
model = CognifluxTransformer(cfg).cuda().eval()

ckpt  = torch.load("checkpoints/cogniflux_1b/step_0005000.pt")
model.load_state_dict(ckpt["model"])

prompt = (
    "<|system|>\nYou are a high capacity reasoning agent called Cogniflux.\n"
    "<|user|>\nIf all A are B and all B are C, what can we conclude?\n"
    "<|assistant|>\n"
)
ids = torch.tensor([encode(prompt)], device="cuda")
out = model.generate(ids, max_new_tokens=256, temperature=0.7, top_p=0.9)
print(decode(out[0].tolist()))
```

---

## Configuration

All hyperparameters live in `configs/cogniflux_1b.yaml`.

```yaml
model:
  hidden_dim: 2048
  num_layers: 20
  num_heads: 16
  num_kv_heads: 4         # GQA
  ffn_dim: 5632           # SwiGLU intermediate
  max_seq_len: 2048
  pfc_layer_interval: 4   # insert PFC block every N layers
  working_memory_slots: 64
  planning_iters: 3       # refinement steps in PlanningModule

training:
  total_steps: 5000
  max_lr: 3.0e-4
  per_device_batch_size: 2
  grad_accum_steps: 128   # effective batch ≈ 1M tokens
  dtype: bfloat16
```

**Memory tuning** — if you hit OOM:
| Setting | Default | Change to |
|---|---|---|
| `per_device_batch_size` | 2 | 1 |
| `grad_accum_steps` | 128 | 256 |
| `max_seq_len` | 2048 | 1024 |

---

## Project Structure

```
Cogniflux/
├── cogniflux/
│   ├── model/
│   │   ├── config.py           # CognifluxConfig dataclass
│   │   ├── layers.py           # RMSNorm, RoPE
│   │   ├── attention.py        # Grouped Query Attention + Flash Attention
│   │   ├── transformer.py      # Full model, chunked cross-entropy
│   │   └── pfc/
│   │       ├── working_memory.py    # 64-slot cross-attention buffer
│   │       ├── executive_control.py # Gated WM/attention routing
│   │       ├── inhibitory_gate.py   # FFN output filtering
│   │       └── planning.py          # Iterative refinement module
│   ├── training/
│   │   ├── trainer.py          # Training loop, checkpointing
│   │   ├── optimizer.py        # AdamW + cosine LR schedule
│   │   └── distributed.py      # DDP setup for mixed AMD GPU architectures
│   ├── data/
│   │   ├── tokenizer.py        # tiktoken GPT-2 BPE wrapper
│   │   ├── dataset.py          # Memory-mapped sharded token dataset
│   │   └── dataloader.py       # DDP-aware DataLoader builder
│   └── kernels/
│       ├── fused_ops.cpp       # C++ extensions: RMSNorm, SwiGLU, RoPE
│       └── build.py            # Builds kernels for CUDA or ROCm
├── configs/
│   └── cogniflux_1b.yaml
├── scripts/
│   ├── train.py
│   ├── prepare_data.py
│   └── evaluate.py
├── requirements.txt
└── setup.py
```

---

## License

See [LICENSE](LICENSE).
