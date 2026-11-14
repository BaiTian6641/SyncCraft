"""Reserved Phase 3 adapter boundary; deliberately fails until implemented."""

from .common import ExportError


def export_bedrock(*args, **kwargs):
    raise ExportError("Bedrock geometry/animation export is not implemented. Complete Phase 3 versioned schemas and client-import tests first.")


def export_geckolib(*args, **kwargs):
    raise ExportError("GeckoLib export is not implemented. Validate its geometry, animation and expression subset separately from Bedrock.")
