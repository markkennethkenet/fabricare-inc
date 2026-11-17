"""Optional generative wording for the care advice, written by Google Gemini.

The trained care model still decides the laundry mode, and the referenced rule tips (care_model.care_tips and
care_model.laundry_tips) stay the facts Gemini works from. Gemini only rewords and adapts them for the garment
it sees, within the care label's limits. With no GEMINI_API_KEY, or on any error, timeout or quota limit, the
API keeps the rule tips, so a scan never waits on or fails because of Gemini.

Gemini's free tier charges nothing for Flash models, but Google may use free-tier requests to improve its
products. Set GEMINI_SEND_PHOTO=false to send only the scan results, not the front photo.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_TIPS = 3
MAX_CHARACTERS = 220
PHOTO_SIDE = 768
log = logging.getLogger(__name__)

SYSTEM = (
    "You write garment care advice for the FabriCare app, for the one garment described. "
    "Write at most 3 care recommendations and at most 3 laundry steps, each a single plain sentence under 25 words. "
    "Use exactly the wash method in laundry_mode_to_follow; it already applies the care label and the fabric "
    "certainty, so never suggest a different method. Never go beyond the care label limits given. "
    "Put damage first when there is any. If the fabric is unconfirmed, keep to gentle care and say to check the "
    "fibre label. Use only the facts given and what the photo plainly shows; do not invent fabric, damage, "
    "chemicals, brands or numbers. Draw on the verified guidance provided."
)
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "care_tips": {"type": "ARRAY", "items": {"type": "STRING"}},
        "laundry_tips": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["care_tips", "laundry_tips"],
}


def model_name():
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def enabled():
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _photo_part(path):
    with Image.open(path) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    photo.thumbnail((PHOTO_SIDE, PHOTO_SIDE), Image.Resampling.LANCZOS)
    with io.BytesIO() as encoded:
        photo.save(encoded, format="JPEG", quality=85)
        data = base64.b64encode(encoded.getvalue()).decode("ascii")
    return {"inlineData": {"mimeType": "image/jpeg", "data": data}}


def _clean(values):
    if not isinstance(values, list):
        return None
    tips = [" ".join(value.split()) for value in values if isinstance(value, str) and value.strip()]
    tips = [tip if tip.endswith((".", "!")) else tip + "." for tip in tips[:MAX_TIPS]]
    if not tips or any(len(tip) > MAX_CHARACTERS for tip in tips):
        return None
    return tips


def write_care_advice(facts, photo_path=None):
    """Return {"care_tips": [...], "laundry_tips": [...]} (1–3 sentences each), or None to keep the rule tips."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    parts = [{"text": "Garment scan results as JSON:\n" + json.dumps(facts, ensure_ascii=False)}]
    if photo_path and os.environ.get("GEMINI_SEND_PHOTO", "true").strip().lower() != "false":
        try:
            parts.append(_photo_part(photo_path))
        except Exception:
            log.warning("Gemini care advice: the front photo could not be prepared; sending results only")
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA,
                             "temperature": 0.6, "maxOutputTokens": 600},
    }
    request = Request(API_URL.format(model=model_name()), data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urlopen(request, timeout=float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "8"))) as response:
            answer = json.load(response)
        text = "".join(part.get("text", "") for part in answer["candidates"][0]["content"]["parts"])
        written = json.loads(text)
    except HTTPError as error:
        # 429 = the free quota is used up; the rule tips cover it. Never log the key.
        log.warning("Gemini care advice unavailable: HTTP %s", error.code)
        return None
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
        log.warning("Gemini care advice unavailable; keeping the rule tips")
        return None
    care, laundry = _clean(written.get("care_tips")), _clean(written.get("laundry_tips"))
    if care is None or laundry is None:
        log.warning("Gemini care advice did not match the expected shape; keeping the rule tips")
        return None
    return {"care_tips": care, "laundry_tips": laundry}


def scan_facts(fabric_type, fabric_confirmed, laundry_mode_label, headline, row, user_inputs, care_tips,
               laundry_tips):
    """The scan results Gemini may use: what the models found, the user's details and the referenced tips."""
    from care_model import DAMAGE_TYPES, is_dark_or_strong

    label = {name: value for name, value in (user_inputs.get("care_label") or {}).items()
             if value not in (None, "unknown")}
    habits = {name: user_inputs.get(name) for name in ("use_frequency", "washing_frequency", "laundering_practice")
              if user_inputs.get(name) not in (None, "", "unspecified")}
    # The headline, not the care model's raw mode, is what the card shows: it applies the care label and,
    # for an unconfirmed fabric, the gentlest wash. Gemini must not contradict it.
    return {
        "fabric": fabric_type if fabric_confirmed else f"{fabric_type} (unconfirmed)",
        "laundry_mode_to_follow": headline,
        **({"care_model_suggestion_before_safety_checks": laundry_mode_label}
           if laundry_mode_label != headline else {}),
        "damage_found": [kind for kind in DAMAGE_TYPES if row.get(f"{kind}_count", 0) > 0],
        "colour": ("dark or strongly coloured" if is_dark_or_strong(row)
                   else "light" if row.get("luminance_mean", 0) >= 170 else "medium"),
        "care_label_entered": label,
        "wear_and_washing": habits,
        "verified_guidance": [tip["text"] for tip in care_tips + laundry_tips],
    }
