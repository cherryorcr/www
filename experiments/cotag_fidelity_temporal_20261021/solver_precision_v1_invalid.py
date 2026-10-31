"""Float64 Sinkhorn arithmetic, preserving the existing objective and tolerance."""


def install(module):
    original = module._log_sinkhorn

    def precise(scores, column_mass, *args, **kwargs):
        result = original(scores.double(), column_mass.double(), *args, **kwargs)
        return result.to(scores.dtype)

    module._log_sinkhorn = precise
    return original
