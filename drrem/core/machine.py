"""Радиальная машина: состояние, связность W = S + γA, хоп, фазы, чтение, энергия.

Соглашения (RESEARCH_PROGRAM.md §2):
  * сообщение нейрону j от нейрона i равно W[j, i] * s_i, т.е. drive = s @ W.T;
  * s = rho(x - theta) — событие; x — мембранное состояние;
  * хоп: x <- (1 - alpha) x + alpha (W s + I + force);
  * чтение уровня l: p_l = softmax(E_r s_l / tau_r); потеря C = Σ_l w_l CE(p_l, y);
  * сила подталкивания: -∂C/∂s = Σ_l w_l E_rᵀ (y - p_l) / tau_r  (проверяется autograd в тестах);
  * энергия Хопфилда (1984) в координатах s при γ = 0 — функция Ляпунова свободной динамики.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from drrem.config import MachineConfig
from drrem.frontend.bytes_cnn import ByteCNN

NO_BYTE = 256  # индекс «байта нет» для окна CNN до начала документа


class Machine:
    def __init__(self, cfg: MachineConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        N, L, D = cfg.N, cfg.L, cfg.D
        gen = torch.Generator().manual_seed(cfg.seed)
        lvl = torch.arange(D) // N
        self.mask = ((lvl[:, None] - lvl[None, :]).abs() <= 1).float().to(self.device)
        G1 = torch.randn(D, D, generator=gen) / math.sqrt(N)
        G2 = torch.randn(D, D, generator=gen) / math.sqrt(N)
        self.S = (cfg.g_S * 0.5 * (G1 + G1.T)).to(self.device) * self.mask
        self.A = (cfg.g_A * 0.5 * (G2 - G2.T)).to(self.device) * self.mask
        self.E_in = (torch.randn(256, N, generator=gen) * cfg.g_in).to(self.device)
        self.E_r = (torch.randn(256, N, generator=gen) * cfg.g_r / math.sqrt(N)).to(self.device)
        self.theta = torch.full((D,), float(cfg.theta), device=self.device)
        if cfg.readout_levels == "all":
            self.level_w = torch.full((L,), 1.0 / L, device=self.device)
        else:
            self.level_w = torch.zeros(L, device=self.device)
            self.level_w[-1] = 1.0
        self.frontend = ByteCNN(N, cfg.cnn_window, cfg.seed, cfg.g_in, self.device) if cfg.frontend == "cnn" else None

    # ------------------------------------------------------------------ параметры
    def W(self) -> torch.Tensor:
        return self.S + self.cfg.gamma_in * self.A

    @contextlib.contextmanager
    def instrumented(self):
        """Подменяет W и E_r листовыми тензорами с градиентом — autograd как измерительный прибор.
        Внутри блока W берётся из self._W_leaf, E_r — из self.E_r (подменён). После — восстанавливается."""
        W_leaf = self.W().detach().clone().requires_grad_(True)
        E_leaf = self.E_r.detach().clone().requires_grad_(True)
        E_saved = self.E_r
        self.E_r = E_leaf
        try:
            yield W_leaf, E_leaf
        finally:
            self.E_r = E_saved

    def spectral(self) -> dict:
        """Спектральные радиусы: S (симметричная, вещественный спектр), W (полная), A (мнимый спектр)."""
        eS = torch.linalg.eigvalsh(self.S)
        eW = torch.linalg.eigvals(self.W())
        eA = torch.linalg.eigvals(self.A)
        return {
            "S_max_eig": float(eS.max()),
            "S_min_eig": float(eS.min()),
            "W_rho": float(eW.abs().max()),
            "A_rho": float(eA.abs().max()),
            "S_fro": float(self.S.norm()),
            "A_fro": float(self.A.norm()),
            "E_r_fro": float(self.E_r.norm()),
        }

    # ------------------------------------------------------------------ вход
    def input_drive(self, x_bytes: torch.Tensor, t: int) -> torch.Tensor:
        """Входной ток на такте t: (B, D); ненулевой только на уровне 1."""
        B = x_bytes.shape[0]
        if self.frontend is None:
            I1 = self.E_in[x_bytes[:, t]]
        else:
            K = self.cfg.cnn_window
            lo = t - K + 1
            if lo < 0:
                pad = torch.full((B, -lo), NO_BYTE, dtype=torch.long, device=x_bytes.device)
                window = torch.cat([pad, x_bytes[:, : t + 1]], dim=1)
            else:
                window = x_bytes[:, lo : t + 1]
            I1 = self.frontend(window)
        I = torch.zeros(B, self.cfg.D, device=self.device, dtype=I1.dtype)
        I[:, : self.cfg.N] = I1
        return I

    # ------------------------------------------------------------------ динамика
    def rho(self, x: torch.Tensor) -> torch.Tensor:
        z = x - self.theta
        if self.cfg.rho == "hardsig":
            return z.clamp(0.0, 1.0)
        return F.relu(z)

    def hop(self, x: torch.Tensor, I: torch.Tensor, W: torch.Tensor | None = None, force: torch.Tensor | None = None):
        W = self.W() if W is None else W
        s = self.rho(x)
        drive = s @ W.T + I
        if force is not None:
            drive = drive + force
        return (1.0 - self.cfg.alpha) * x + self.cfg.alpha * drive

    def run_free(self, x, I, H: int, W=None, active=None, record: bool = False):
        traj = [self.rho(x)] if record else None
        for _ in range(H):
            xn = self.hop(x, I, W)
            x = torch.where(active[:, None], xn, x) if active is not None else xn
            if record:
                traj.append(self.rho(x))
        return x, traj

    def run_nudged(self, x, I, H: int, beta: float, y, W=None, active=None, record: bool = False):
        traj = [self.rho(x)] if record else None
        for _ in range(H):
            s = self.rho(x)
            force = beta * self.nudge_force(s, y)
            xn = self.hop(x, I, W, force)
            x = torch.where(active[:, None], xn, x) if active is not None else xn
            if record:
                traj.append(self.rho(x))
        return x, traj

    def fixed_point_residual(self, x, I, W=None) -> torch.Tensor:
        """|x - (W s + I)| / |x| по образцам: 0 в неподвижной точке свободной динамики."""
        W = self.W() if W is None else W
        s = self.rho(x)
        r = x - (s @ W.T + I)
        return r.norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)

    # ------------------------------------------------------------------ чтение
    def logits(self, s: torch.Tensor) -> torch.Tensor:
        B = s.shape[0]
        return (s.view(B, self.cfg.L, self.cfg.N) @ self.E_r.T) / self.cfg.tau_r  # (B, L, 256)

    def probs(self, s: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(s), dim=-1)

    def loss_per_sample(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Взвешенная по уровням кросс-энтропия, наты, (B,)."""
        B, L = s.shape[0], self.cfg.L
        lg = self.logits(s).reshape(B * L, 256)
        ce = F.cross_entropy(lg, y.repeat_interleave(L), reduction="none").view(B, L)
        return (ce * self.level_w).sum(-1)

    def loss_per_level(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, L = s.shape[0], self.cfg.L
        lg = self.logits(s).reshape(B * L, 256)
        return F.cross_entropy(lg, y.repeat_interleave(L), reduction="none").view(B, L)

    def nudge_force(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """-∂C/∂s = Σ_l w_l (y - p_l) E_r / tau_r, (B, D)."""
        B = s.shape[0]
        p = self.probs(s)  # (B, L, 256)
        yoh = F.one_hot(y, 256).to(p.dtype)
        f = ((yoh[:, None, :] - p) @ self.E_r) / self.cfg.tau_r * self.level_w[None, :, None]
        return f.reshape(B, self.cfg.D)

    # ------------------------------------------------------------------ энергия
    def energy(self, x: torch.Tensor, I: torch.Tensor, W=None) -> torch.Tensor:
        """E(s) = -½ sᵀ W s - sᵀ I + Σ Φ(s), Φ(s) = s²/2 + θ s  (rho⁻¹(s) = s + θ). (B,)"""
        W = self.W() if W is None else W
        s = self.rho(x)
        quad = -0.5 * ((s @ W.T) * s).sum(-1)
        return quad - (s * I).sum(-1) + (0.5 * s * s + self.theta * s).sum(-1)

    def energy_from_s(self, s: torch.Tensor, I: torch.Tensor, W=None) -> torch.Tensor:
        W = self.W() if W is None else W
        quad = -0.5 * ((s @ W.T) * s).sum(-1)
        return quad - (s * I).sum(-1) + (0.5 * s * s + self.theta * s).sum(-1)

    def to_dtype(self, dtype: torch.dtype) -> "Machine":
        """Перевод всех параметров в dtype (float64 — для проверок конечными разностями)."""
        for name in ("S", "A", "E_in", "E_r", "theta", "mask", "level_w"):
            setattr(self, name, getattr(self, name).to(dtype))
        if self.frontend is not None:
            self.frontend.to(dtype)
        return self

    # ------------------------------------------------------------------ состояние
    def init_state(self, B: int) -> torch.Tensor:
        return torch.zeros(B, self.cfg.D, device=self.device)

    def state_dict(self) -> dict:
        d = {"S": self.S, "A": self.A, "E_in": self.E_in, "E_r": self.E_r, "theta": self.theta}
        if self.frontend is not None:
            d["frontend"] = self.frontend.state_dict()
        return d
