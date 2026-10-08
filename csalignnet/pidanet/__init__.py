"""CSAlignNet model package."""

__all__ = [
    "CSAlignNet",
    "CSAlignLoss",
    "SessionInvariantHead",
    "FingerVeinROIExtractor",
]


def __getattr__(name: str):
    if name == "CSAlignNet":
        from .cs_model import CSAlignNet

        return CSAlignNet
    if name == "CSAlignLoss":
        from .cs_losses import CSAlignLoss

        return CSAlignLoss
    if name == "SessionInvariantHead":
        from .session_head import SessionInvariantHead

        return SessionInvariantHead
    if name == "FingerVeinROIExtractor":
        from .roi import FingerVeinROIExtractor

        return FingerVeinROIExtractor
    raise AttributeError(name)
