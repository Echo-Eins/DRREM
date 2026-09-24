import torch
from torch.nn import functional as F

from scripts.probe_phase_addressing import delta_coefficients


def test_coefficients_match_independent_sequential_memory():
    torch.manual_seed(63)
    q = F.normalize(torch.randn(2, 3, 11, 8, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(2, 3, 11, 5, dtype=torch.float64)
    beta = torch.rand(2, 3, 11, dtype=torch.float64)
    beta[0, :, :3] = 0  # masked prefix, no write
    memory = torch.zeros(2, 3, 8, 5, dtype=torch.float64)
    for t in range(q.shape[2]):
        coefficient = delta_coefficients(q[:, :, t], k[:, :, :t], beta[:, :, :t])
        prediction = (coefficient[..., None] * v[:, :, :t]).sum(2)
        expected = torch.einsum('bhk,bhkv->bhv', q[:, :, t], memory)
        torch.testing.assert_close(prediction, expected, rtol=1e-12, atol=1e-12)
        residual = v[:, :, t] - torch.einsum('bhk,bhkv->bhv', k[:, :, t], memory)
        memory = memory + beta[:, :, t, None, None] * k[:, :, t, :, None] * residual[..., None, :]


def test_address_collision_overwrites_even_without_time_decay():
    k = torch.tensor([[[[1., 0.], [0., 1.], [1., 0.]]]], dtype=torch.float64)
    q = k[:, :, 0]
    beta = torch.ones(1, 1, 3, dtype=torch.float64)
    coefficient = delta_coefficients(q, k, beta)
    torch.testing.assert_close(coefficient, torch.tensor([[[0., 0., 1.]]], dtype=torch.float64))
