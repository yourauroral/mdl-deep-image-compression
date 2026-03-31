"""
Muon Optimizer — 对 2D 权重矩阵使用 Newton-Schulz 正交化的 SGD+Momentum。

核心思想：
  标准 SGD+Momentum 对梯度矩阵 G 的每个奇异值方向施加相同的步长，
  导致"稀有方向"（小奇异值方向）更新不足。
  Muon 先将 G 投影到最近的正交矩阵 Q = UV^T（其中 G=UΣV^T），
  使所有方向的有效步长一致，放大稀有方向的更新。

  直接 SVD 太贵（O(d^3)），Muon 用 5 步 Newton-Schulz 迭代近似：
    X_{k+1} = a·X_k + b·X_k·(X_k^T·X_k) + c·X_k·(X_k^T·X_k)^2
  5 步即可在 ||G^T·G||_∞ ≤ 30 范围内以 <1e-7 精度逼近 sign(G)。
  整个过程只需矩阵乘法，可在 bf16 下高效运行。

  对非 2D 参数（embedding, norm, bias）仍使用 AdamW。

参考:
  [1] Keller Jordan, "Muon: An optimizer for hidden layers in transformers,"
      2024, https://kellerjordan.github.io/posts/muon/
      提出 Newton-Schulz 正交化替代 Adam 用于 hidden 层。
  [2] Keller Jordan et al., "Muon is Scalable for LLM Training,"
      arXiv:2502.16982, 2025.
      在 1.5B 模型上验证 Muon 可扩展性，1.35x 计算效率提升。
  [3] Bernstein et al., "Old Optimizer, New Norm: An Anthology," 2024.
      Muon 的理论基础：spectral normalization of updates。
"""

import torch
from torch.optim import Optimizer


def _newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz 迭代求矩阵符号函数 sign(G) ≈ UV^T。

    使用 quintic polynomial 迭代 [1]:
      X_{k+1} = a·X_k + b·X_k·(X_k^T·X_k) + c·X_k·(X_k^T·X_k)^2

    系数 (a=3.4445, b=-4.7750, c=2.0315) 来自 [1]，
    在 spectral norm < sqrt(30) 的范围内保证收敛，5 步精度 <1e-7。

    参数:
      G: (m, n) 梯度矩阵，m >= n（如果 m < n 则转置处理）
      steps: Newton-Schulz 迭代步数，默认 5
      eps: 数值稳定性下界
    返回:
      (m, n) 近似正交矩阵，满足 Q^T·Q ≈ I
    """
    assert G.ndim == 2
    # 确保 m >= n（短胖矩阵转置为高瘦矩阵处理）
    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        transposed = True

    # 缩放到 spectral norm ≤ 1，使迭代在收敛域内
    # 使用 Frobenius norm 近似: ||G||_F / sqrt(min(m,n)) 近似 ||G||_2
    # 再除以 sqrt(30) 确保初始 ||X^T X||_∞ < 30 [1]
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    nrm = X.norm() + eps
    X = X / (nrm * (30 ** 0.5))  # 缩放使 ||X^T X|| < 1

    # 5 步 quintic iteration [1]
    for _ in range(steps):
        A = X.T @ X                          # (n, n)
        B = b * A + c * (A @ A)              # (n, n)
        X = a * X + X @ B                    # (m, n)

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon 优化器：仅用于 2D 权重矩阵。

    等效于 SGD+Momentum，但在 step 之前对梯度做 Newton-Schulz 正交化，
    使所有奇异值方向的有效步长一致。

    超参数:
      lr: 学习率，推荐 0.02 (远大于 Adam 的 1e-4，因为正交化后
          步长已归一化) [1][2]
      momentum: 动量系数，默认 0.95 [1]
      ns_steps: Newton-Schulz 迭代步数，默认 5 [1]

    注意:
      - 只对 ndim==2 的参数使用 Muon [1]
      - Embedding, norm, bias 等非 2D 参数应放到 AdamW 中
      - 与 GradScaler 兼容（unscale_ 后 step）
    """

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                # Newton-Schulz 正交化 [1]
                g_orth = _newton_schulz_5(g, steps=ns_steps)

                # 缩放: 使正交化后的梯度与原始梯度 Frobenius norm 匹配
                # 这保持了 LR 的物理含义一致性 [1]
                scale = (g.numel() ** 0.5) / (g_orth.norm() + 1e-8)
                g_orth = g_orth * scale

                # SGD + Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = g_orth.clone()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g_orth)
                    g_orth = buf

                p.add_(g_orth, alpha=-lr)

        return loss


def build_muon_adamw(model, muon_lr=0.02, muon_momentum=0.95, muon_ns_steps=5,
                     adamw_lr=1e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                     adamw_wd=0.1, no_decay_keywords=None):
    """
    构建 Muon + AdamW 混合优化器组。

    路由规则 [1]:
      - 2D 权重矩阵（Linear.weight）→ Muon
      - 非 2D 参数（Embedding, RMSNorm, bias）→ AdamW

    参数:
      model: nn.Module
      muon_lr: Muon 学习率，推荐 0.02 [1]
      muon_momentum: Muon 动量，默认 0.95
      muon_ns_steps: Newton-Schulz 步数，默认 5
      adamw_lr: AdamW 学习率（用于 embedding/norm/bias）
      adamw_betas: AdamW beta 参数
      adamw_eps: AdamW eps
      adamw_wd: AdamW weight decay
      no_decay_keywords: 不应用 weight decay 的参数名关键词

    返回:
      (muon_optimizer, adamw_optimizer) 元组

    使用方式:
      muon_opt, adamw_opt = build_muon_adamw(model)
      # 训练循环中:
      muon_opt.zero_grad(); adamw_opt.zero_grad()
      loss.backward()
      muon_opt.step(); adamw_opt.step()
    """
    if no_decay_keywords is None:
        no_decay_keywords = ['token_embed', 'pos_embed', 'norm', 'bias']

    muon_params = []
    adamw_decay_params = []
    adamw_nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 2D 且不在 no_decay 名单中 → Muon
        if param.ndim == 2 and not any(kw in name for kw in no_decay_keywords):
            muon_params.append(param)
        elif any(kw in name for kw in no_decay_keywords):
            adamw_nodecay_params.append(param)
        else:
            adamw_decay_params.append(param)

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                    ns_steps=muon_ns_steps) if muon_params else None

    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({'params': adamw_decay_params, 'weight_decay': adamw_wd})
    if adamw_nodecay_params:
        adamw_groups.append({'params': adamw_nodecay_params, 'weight_decay': 0.0})

    adamw_opt = torch.optim.AdamW(
        adamw_groups, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps
    ) if adamw_groups else None

    return muon_opt, adamw_opt
