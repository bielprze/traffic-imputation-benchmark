# PEP 8 formatting adapted for this benchmark on 2026-11-04.
# Algorithms and public interfaces are preserved.
# Only used by the unused upstream CLI argument helper.
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(value)
