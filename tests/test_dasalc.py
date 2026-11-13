"""Checks for the DASALC model and loss. Run: uv run pytest -q"""
import numpy as np
import torch

from ltr.dasalc import DASALC, log1p_transform, softmax_ce_loss

torch.manual_seed(0)
F_IN = 10


def _model(**kw):
    m = DASALC(F_IN, hidden=16, layers=2, attn_layers=3, heads=3, head_size=8, **kw)
    # Put BatchNorm statistics in a non-trivial state, then switch to eval mode.
    m.train()
    m(torch.randn(4, 7, F_IN), torch.ones(4, 7, dtype=torch.bool))
    return m.eval()


def test_loss_matches_paper_formula():
    s = torch.tensor([[2.0, 1.0, 0.0], [0.5, -1.0, float("-inf")]])
    y = torch.tensor([[2.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    m = torch.tensor([[True, True, True], [True, True, False]])
    want = (-(y[0] * torch.log_softmax(s[0], 0)).sum()
            - (y[1, :2] * torch.log_softmax(s[1, :2], 0)).sum()) / 2
    assert torch.allclose(softmax_ce_loss(s, y, m), want)


def test_loss_skips_queries_without_relevant_docs():
    s = torch.tensor([[2.0, 1.0], [0.3, 0.1]])
    m = torch.ones(2, 2, dtype=torch.bool)
    y_one = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(softmax_ce_loss(s, y_one, m), softmax_ce_loss(s[:1], y_one[:1], m[:1]))


def test_empty_batch_gives_zero_not_nan():
    s = torch.tensor([[1.0, float("-inf")]], requires_grad=True)
    loss = softmax_ce_loss(s, torch.zeros(1, 2), torch.tensor([[True, False]]))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(s.grad).all()


def test_permutation_equivariance():  # paper Proposition 1
    m = _model()
    x = torch.randn(1, 6, F_IN)
    mask = torch.ones(1, 6, dtype=torch.bool)
    perm = torch.randperm(6)
    with torch.no_grad():
        assert torch.allclose(m(x, mask)[0, perm], m(x[:, perm], mask)[0], atol=1e-5)


def test_padding_does_not_change_scores():
    m = _model()
    x = torch.randn(2, 9, F_IN)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 5:] = False  # list 0 has 5 docs, padded to 9
    with torch.no_grad():
        batched = m(x, mask)[0, :5]
        alone = m(x[:1, :5], torch.ones(1, 5, dtype=torch.bool))[0]
    assert torch.allclose(batched, alone, atol=1e-5)
    assert torch.isinf(m(x, mask)[0, 5:]).all()


def test_gradients_finite_with_padding():
    m = _model()
    m.train()
    x = torch.randn(3, 8, F_IN)
    mask = torch.ones(3, 8, dtype=torch.bool)
    mask[1, 4:] = False
    y = torch.randint(0, 3, (3, 8)).float()
    softmax_ce_loss(m(x, mask), y, mask).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_plain_dnn_has_no_attention():
    m = DASALC(F_IN, hidden=16, layers=2, use_attn=False)
    assert not hasattr(m, "attn")


def test_log1p_switch():
    X = np.array([[np.nan, 3.0, -1.0]])
    assert np.allclose(log1p_transform(X, False), [[0.0, 3.0, -1.0]])
    assert np.allclose(log1p_transform(X, True), [[0.0, np.log(4.0), -np.log(2.0)]])
