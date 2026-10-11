import torch
from core.corr import CorrBlockGather1D


def test_half_correlation_pooling_does_not_overflow():
    # Each correlation is 40000: summing two before division overflows fp16.
    features = torch.full((1, 1, 1, 16), 200.0, dtype=torch.float16)
    block = CorrBlockGather1D(features, features)
    for level in block.corr_pyramid:
        torch.testing.assert_close(level, torch.full_like(level, 40000.0))


def test_missing_optional_sampler_has_actionable_error(monkeypatch):
    import core.corr as corr
    import pytest

    monkeypatch.setattr(corr, "corr_sampler", None)
    with pytest.raises(RuntimeError, match="optional sampler extension"):
        corr.CorrSampler.apply(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), 1)


def test_centered_native_instance_norm():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from core.cuda_kernels import Kernels
    from core.general import GeneralRAFT, S

    engine = GeneralRAFT.__new__(GeneralRAFT)
    engine.k = Kernels.get()
    count, channels = 2051, 32  # final partial chunk is not full
    x = (
        (1000 + (torch.arange(count)[:, None] % 3 - 1) * 0.5)
        .expand(count, channels)
        .half()
        .cuda()
        .contiguous()
    )
    stats = engine.in_stats(S(x.reshape(1, count, channels)))
    y = torch.empty_like(x)
    engine.k.launch(
        "in_apply", engine.k.linear(x.numel() // 8), x, stats, 0, 0, y, x.numel(), channels, 0
    )
    reference = x.cpu().double()
    var, mean = torch.var_mean(reference, dim=0, correction=0)
    reference = ((reference - mean) * torch.rsqrt(var + 1e-5)).half().relu().cuda()
    torch.testing.assert_close(y, reference, atol=3e-5, rtol=0.002)
