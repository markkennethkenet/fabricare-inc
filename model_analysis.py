"""Model-only scan result decisions and upload validation."""
import os

from PIL import Image, UnidentifiedImageError


def confidence_threshold() -> float:
    try:
        value = float(os.getenv("HYBRID_ML_CONFIDENCE_THRESHOLD", "85"))
    except ValueError as error:
        raise RuntimeError("HYBRID_ML_CONFIDENCE_THRESHOLD must be between 0 and 100") from error
    if not 0 <= value <= 100:
        raise RuntimeError("HYBRID_ML_CONFIDENCE_THRESHOLD must be between 0 and 100")
    return value


def validate_image_file(path: str) -> None:
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPG and PNG images are supported.")
            if image.width < 1 or image.height < 1 or image.width * image.height > 24_000_000:
                raise ValueError("Image dimensions are too large to analyze.")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as error:
        raise ValueError("Cannot read image. Upload a valid JPG or PNG photo.") from error


def _fabric_result(ml: dict | None, supported: list[str]) -> dict:
    ml_label = ml.get("fabric_type") if ml else None
    score = ml.get("confidence") if ml else None
    threshold = confidence_threshold()
    high = isinstance(score, (int, float)) and score >= threshold

    if ml is None:
        status, final, review = "unable_to_determine", "uncertain", True
        reason = "No trained fabric prediction is available; check the care label or review the photo manually."
    elif ml_label == "uncertain":
        status, final, review = "ml_uncertain_review", "uncertain", True
        reason = "The trained model could not identify the fabric reliably; manual identification is recommended."
    elif ml_label not in supported:
        status, final, review = "unsupported_model_class", "uncertain", True
        reason = f"The active trained model does not support '{ml_label}'. Its model or class labels need an update."
    elif score is None:
        status, final, review = "score_unavailable_review", ml_label, True
        reason = "Prediction confidence is unavailable for this model. Check the garment care label."
    elif high:
        status, final, review = "model_result", ml_label, False
        reason = "The trained model score meets the configured experimental threshold."
    else:
        status, final, review = "low_confidence_review", ml_label, True
        reason = "The trained model score is below the configured experimental threshold; manual review is recommended."

    return {
        "ml_prediction": ml_label,
        "ml_confidence": score,
        "ml_score_kind": (ml or {}).get("score_kind", "model_score"),
        "final_prediction": final,
        "verification_status": status,
        "review_required": review,
        "threshold_percent": threshold,
        "supported_ml_classes": supported,
        "reasoning": reason,
    }


def _damage_result(detections: list[dict] | None, model_available: bool, supported: list[str]) -> dict:
    ml_types = sorted({item["type"].lower() for item in (detections or [])}) if model_available else None
    if not model_available:
        status, final, review = "unable_to_determine", [], True
        reason = "Damage could not be assessed; retake the photo or review it manually."
    elif ml_types:
        status, final, review = "model_only", ml_types, True
        reason = "Trained detector findings are possible damage and need human review."
    else:
        status, final, review = "no_visible_damage", [], False
        reason = "The detector flagged no supported findings in the processed photos. Unseen or missed damage may still be present."
    return {
        "ml_predictions": ml_types,
        "final_predictions": final,
        "verification_status": status,
        "review_required": review,
        "ml_model_available": model_available,
        "ml_supported_classes": supported,
        "reasoning": reason,
    }


def model_result(ml_fabric, ml_damages, supported_labels, damage_model_available, supported_damage_labels):
    """Build scan guidance from the active trained models without external services."""
    return {
        "fabric": _fabric_result(ml_fabric, list(supported_labels)),
        "damage": _damage_result(ml_damages, damage_model_available, list(supported_damage_labels)),
        "analysis_status": "completed",
        "fabric_limitations": "Appearance alone may not reliably reveal fiber composition. Denim describes construction and may be made from different fibers.",
    }
