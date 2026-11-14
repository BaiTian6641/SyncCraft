"""Planned trainer boundary; configs/training_static.yaml is an intent template."""


def train(*args, **kwargs):
    raise NotImplementedError(
        "P1-09 requires verified model/data manifests, loss-kernel parity, actual "
        "VRAM profiling, and full-state atomic checkpoint/resume. No job was launched."
    )
