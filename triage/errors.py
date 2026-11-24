"""Reason codes and narrow exception types for model loading, scoring and calibration."""
from __future__ import annotations

import json
import pickle

UNAVAILABLE_MESSAGE = "Assessment unavailable — follow standard procedure."
LOW_RANK_ADVISORY = ("A low model ranking is not a statement that a call is safe or non-urgent. "
                     "Follow standard procedure and dispatcher judgement.")


class Reason:
    """Stable machine-readable reason codes. Never reuse a code for a different meaning."""
    MODEL_ARTIFACT_MISSING = "MODEL_ARTIFACT_MISSING"
    MODEL_ARTIFACT_UNREADABLE = "MODEL_ARTIFACT_UNREADABLE"
    MODEL_METADATA_INVALID = "MODEL_METADATA_INVALID"
    ARTIFACT_UNREGISTERED = "ARTIFACT_UNREGISTERED"
    ARTIFACT_CHECKSUM_MISMATCH = "ARTIFACT_CHECKSUM_MISMATCH"
    REGISTRY_INVALID = "REGISTRY_INVALID"
    CONFIG_INVALID = "CONFIG_INVALID"
    AUDIT_WRITE_FAILED = "AUDIT_WRITE_FAILED"
    MODEL_NOT_APPROVED = "MODEL_NOT_APPROVED"
    PREDICTION_FAILED = "PREDICTION_FAILED"
    V2_UNAVAILABLE_V1_USED = "V2_UNAVAILABLE_V1_USED"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"
    EXPLANATION_UNAVAILABLE = "EXPLANATION_UNAVAILABLE"
    NONE = None


class TriageError(Exception):
    """Base class. `reason_code` is safe to return to clients; `detail` is for logs only."""

    reason_code = Reason.PREDICTION_FAILED

    def __init__(self, detail: str = "", reason_code: str | None = None):
        super().__init__(detail)
        self.detail = detail
        if reason_code:
            self.reason_code = reason_code


class ModelUnavailableError(TriageError):
    reason_code = Reason.MODEL_ARTIFACT_MISSING


class IntegrityError(TriageError):
    reason_code = Reason.ARTIFACT_CHECKSUM_MISMATCH


class CalibrationError(TriageError):
    reason_code = Reason.CALIBRATION_FAILED


class PredictionError(TriageError):
    reason_code = Reason.PREDICTION_FAILED


def _lightgbm_error():
    try:
        from lightgbm.basic import LightGBMError
        return LightGBMError
    except ImportError:  # pragma: no cover - lightgbm is a hard requirement
        return RuntimeError


# What joblib/pickle/json can raise for a missing, truncated or incompatible artifact.
LOAD_ERRORS = (OSError, EOFError, ValueError, KeyError, IndexError, ImportError, AttributeError, TypeError,
               NotImplementedError, pickle.UnpicklingError, json.JSONDecodeError)
# What a loaded model/pandas can raise for a row it cannot score.
SCORE_ERRORS = (ValueError, KeyError, TypeError, IndexError, AttributeError, _lightgbm_error())
