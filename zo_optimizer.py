from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(self,
        model: nn.Module,
        lr: float = 0.05, # learning rate
        eps: float = 0.5, # perturbation value
        perturbation_mode: str = "gaussian", # perturbation mode
        lora_rank: int = 8, # LoRA rank
        lora_alpha: float = 8.0, # LoRA alpha
        n_spsa: int = 100, # Number of SPSA forward passes per batch
    ) -> None:
        """
        Initializes the Zero-Order optimizer with SPSA gradient estimation 
         and LoRA (Low-Rank Adaptation) for parameter efficiency.
        """
        self.model = model
        self.lr = lr
        self.eps = eps
        self.n_spsa = n_spsa
        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank
        self.step_count = 0
        
        self._flat: dict[str, nn.Parameter] = {}
        self._lora: dict[str, dict] = {}

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

        self._v: dict[str, torch.Tensor] = {}
        self._m: dict[str, torch.Tensor] = {}

        self._set()

    def _set(self) -> None:
        """
        Identifies target layers and initializes LoRA matrices (A, B) for weights 
        or prepares direct optimization for biases and other low-dim parameters.
        """

        named = dict(self.model.named_parameters())
        for name in self.layer_names:
            if name not in named:
                continue
            p = named[name]

            if p.dim() >= 2:
                out_dim = p.shape[0]
                in_dim  = p.numel() // out_dim
                r = min(self.lora_rank, out_dim, in_dim)

                A = torch.zeros(r, in_dim, device=p.device, dtype=p.dtype)
                B = torch.randn(out_dim, r, device=p.device, dtype=p.dtype) * 0.02

                self._lora[name] = {
                    "W0":    p.data.reshape(out_dim, in_dim).clone(),
                    "B":     B,   # fixed
                    "A":     A,   # optimising
                    "shape": p.shape,
                    "out":   out_dim,
                    "in":    in_dim,
                }
                self._m[f"{name}_A"] = torch.zeros_like(A)
                self._v[f"{name}_A"] = torch.zeros_like(A)
                p.data.copy_(self._effective(name))
            else:
                self._flat[name] = p
                self._m[name] = torch.zeros_like(p)
                self._v[name] = torch.zeros_like(p)

    def _effective(self, name: str) -> torch.Tensor:
        """
        Computes the effective weight matrix by combining the frozen base weight 
        with the trained low-rank adapter matrices.
        """

        lrm = self._lora[name]
        dev = lrm["B"].device
        if lrm["W0"].device != dev:
            lrm["W0"] = lrm["W0"].to(dev)
        return (lrm["W0"] + self.lora_scale * lrm["B"] @ lrm["A"]).reshape(lrm["shape"])

    def _apply_all(self, params: dict) -> None:
        """
        Syncs the current state of LoRA adapters and direct parameters back 
        into the original model's parameter data tensors.
        """

        for name in self._lora:
            if name in params:
                lrm = self._lora[name]
                dev = params[name].device
                if lrm["B"].device != dev:
                    lrm["B"]  = lrm["B"].to(dev)
                    lrm["A"]  = lrm["A"].to(dev)
                    lrm["W0"] = lrm["W0"].to(dev)
                params[name].data.copy_(self._effective(name))

    def _sample(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generates a random perturbation tensor (noise) using either 
        Gaussian or Uniform distribution to probe the loss landscape.
        """

        if self.perturbation_mode == "gaussian":
            return torch.randn_like(t)
        return torch.rand_like(t) * 2.0 - 1.0

    def _estimate_grad(
        self, loss_fn: Callable[[], float], params: dict
    ) -> dict[str, torch.Tensor]:
        """
        Estimates gradients via SPSA by sampling perturbations and 
        measuring the change in loss over multiple forward passes.
        """

        accuracy: dict[str, torch.Tensor] = {}
        for name in self._lora:
            accuracy[f"{name}_A"] = torch.zeros_like(self._lora[name]["A"])
        for name in self._flat:
            accuracy[name] = torch.zeros_like(self._flat[name])

        with torch.no_grad():
            for _ in range(self.n_spsa):
                directions: dict[str, torch.Tensor] = {}
                for name in self._lora:
                    directions[f"{name}_A"] = self._sample(self._lora[name]["A"])
                for name in self._flat:
                    directions[name] = self._sample(self._flat[name])

                # f(x + eps * u)
                for name, lrm in self._lora.items():
                    lrm["A"].add_(self.eps * directions[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._flat.items():
                    p.data.add_(self.eps * directions[name])
                f_plus = loss_fn()

                # f(x - eps * u)  — restore then subtract
                for name, lrm in self._lora.items():
                    lrm["A"].sub_(2.0 * self.eps * directions[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._flat.items():
                    p.data.sub_(2.0 * self.eps * directions[name])
                f_minus = loss_fn()

                # Restore original value
                for name, lrm in self._lora.items():
                    lrm["A"].add_(self.eps * directions[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._flat.items():
                    p.data.add_(self.eps * directions[name])

                # Calculate the direction
                coeff = (f_plus - f_minus) / (2.0 * self.eps)
                for key, u in directions.items():
                    accuracy[key].add_(coeff * u)

        return {k: v / self.n_spsa for k, v in accuracy.items()}

    def _adam_step(self, key: str, grad: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Adam update step for a specific parameter using 
        running estimates of first and second moments.
        """
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
        """
        Updates the optimized parameters (LoRA 'A' matrices and direct parameters) 
        using the computed gradients and the Adam update rule.
        """
        with torch.no_grad():
            for name, lrm in self._lora.items():
                key = f"{name}_A"
                if key in grads:
                    lrm["A"].sub_(self._adam_step(key, grads[key]))
                    params[name].data.copy_(self._effective(name))
            for name, p in self._flat.items():
                if name in grads:
                    p.data.sub_(self._adam_step(name, grads[name]))
    
    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def step(self, loss_fn: Callable[[], float]) -> float:
        """
        Performs a single optimization step: applies current parameters, 
        estimates gradients via loss function calls, and updates parameters.
        """
        self.step_count += 1
        params = self._active_params()
        self._apply_all(params)

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)
