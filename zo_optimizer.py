from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.05,
        eps: float = 0.5,
        perturbation_mode: str = "gaussian",
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        n_spsa: int = 100,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.n_spsa = n_spsa
        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank
        self.step_count = 0

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        self.layer_names: list[str] = [
            "fc.weight",
            "fc.bias",
        ]

        self._lora: dict[str, dict] = {}
        self._direct: dict[str, nn.Parameter] = {}

        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

        self._setup()

    def _setup(self) -> None:
        named = dict(self.model.named_parameters())
        for name in self.layer_names:
            if name not in named:
                continue
            param = named[name]

            if param.dim() >= 2:
                out_dim = param.shape[0]
                in_dim  = param.numel() // out_dim
                r = min(self.lora_rank, out_dim, in_dim)

                B = torch.randn(out_dim, r, device=param.device, dtype=param.dtype) * 0.02
                A = torch.zeros(r, in_dim, device=param.device, dtype=param.dtype)

                self._lora[name] = {
                    "W0":    param.data.reshape(out_dim, in_dim).clone(),
                    "B":     B,   # fixed
                    "A":     A,   # optimised
                    "shape": param.shape,
                    "out":   out_dim,
                    "in":    in_dim,
                }
                self._m[f"{name}_A"] = torch.zeros_like(A)
                self._v[f"{name}_A"] = torch.zeros_like(A)
                param.data.copy_(self._effective(name))
            else:
                self._direct[name] = param
                self._m[name] = torch.zeros_like(param)
                self._v[name] = torch.zeros_like(param)

    def _effective(self, name: str) -> torch.Tensor:
        ls = self._lora[name]
        dev = ls["B"].device
        if ls["W0"].device != dev:
            ls["W0"] = ls["W0"].to(dev)
        return (ls["W0"] + self.lora_scale * ls["B"] @ ls["A"]).reshape(ls["shape"])

    def _apply_all(self, params: dict) -> None:
        for name in self._lora:
            if name in params:
                ls = self._lora[name]
                dev = params[name].device
                if ls["B"].device != dev:
                    ls["B"]  = ls["B"].to(dev)
                    ls["A"]  = ls["A"].to(dev)
                    ls["W0"] = ls["W0"].to(dev)
                params[name].data.copy_(self._effective(name))

    def _sample(self, t: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(t)
        return torch.rand_like(t) * 2.0 - 1.0

    def _estimate_grad(
        self, loss_fn: Callable[[], float], params: dict
    ) -> dict[str, torch.Tensor]:
        acc: dict[str, torch.Tensor] = {}
        for name in self._lora:
            acc[f"{name}_A"] = torch.zeros_like(self._lora[name]["A"])
        for name in self._direct:
            acc[name] = torch.zeros_like(self._direct[name])

        with torch.no_grad():
            for _ in range(self.n_spsa):
                dirs: dict[str, torch.Tensor] = {}
                for name in self._lora:
                    dirs[f"{name}_A"] = self._sample(self._lora[name]["A"])
                for name in self._direct:
                    dirs[name] = self._sample(self._direct[name])

                # +ε
                for name, ls in self._lora.items():
                    ls["A"].add_(self.eps * dirs[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._direct.items():
                    p.data.add_(self.eps * dirs[name])
                f_plus = loss_fn()

                # −ε
                for name, ls in self._lora.items():
                    ls["A"].sub_(2.0 * self.eps * dirs[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._direct.items():
                    p.data.sub_(2.0 * self.eps * dirs[name])
                f_minus = loss_fn()

                # Restore
                for name, ls in self._lora.items():
                    ls["A"].add_(self.eps * dirs[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._direct.items():
                    p.data.add_(self.eps * dirs[name])

                coeff = (f_plus - f_minus) / (2.0 * self.eps)
                for key, u in dirs.items():
                    acc[key].add_(coeff * u)

        return {k: v / self.n_spsa for k, v in acc.items()}

    def _adam_step(self, key: str, grad: torch.Tensor) -> torch.Tensor:
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        t = self.step_count
        if self._m[key].device != grad.device:
            self._m[key] = self._m[key].to(grad.device)
            self._v[key] = self._v[key].to(grad.device)
        self._m[key] = b1 * self._m[key] + (1.0 - b1) * grad
        self._v[key] = b2 * self._v[key] + (1.0 - b2) * grad * grad
        m_hat = self._m[key] / (1.0 - b1 ** t)
        v_hat = self._v[key] / (1.0 - b2 ** t)
        return self.lr * m_hat / (v_hat.sqrt() + eps_a)

    def _update_params(self, params: dict, grads: dict) -> None:
        with torch.no_grad():
            for name, ls in self._lora.items():
                key = f"{name}_A"
                if key in grads:
                    ls["A"].sub_(self._adam_step(key, grads[key]))
                    params[name].data.copy_(self._effective(name))
            for name, p in self._direct.items():
                if name in grads:
                    p.data.sub_(self._adam_step(name, grads[name]))

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Layer names not found in model: {missing}")
        return {n: named[n] for n in self.layer_names}

    def step(self, loss_fn: Callable[[], float]) -> float:
        """One ZO step.  Forward passes per call: 1 + 2*n_spsa."""
        self.step_count += 1
        params = self._active_params()
        self._apply_all(params)

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)