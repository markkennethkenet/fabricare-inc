"""Versioned condition and care rules for the scan and damage-review endpoints."""
import json
import re
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from care_model import finding_counts
from model_analysis import review_threshold

CONFIG = json.loads((Path(__file__).parent / "assessment_config.json").read_text(encoding="utf-8"))
METHOD_VERSION = "visible-condition-and-label-care/2.0"
INSUFFICIENT = "insufficient_evidence"
GUIDANCE = "rule_based_guidance"
ESTIMATE = "experimental_estimate"
LIFESPAN_MESSAGE = "Insufficient evidence for a reliable remaining lifespan estimate."

# Garment lifespan figures from Laitala and Klepp (2020), reference D, section 4.1 (46,857 garments, five countries).
# The mean current garment age was 2.6 years, giving an expected total lifespan of about 5.2 years. In the regression,
# cotton is the reference; silk lasts 2.2 years longer, wool 1.2 years, synthetics 5.5 months and regenerated
# cellulose 9.7 months. The cotton level follows from the 5.2-year mean and the fibre mix (wool 16.8%,
# synthetics 14.5%, regenerated cellulose 2.9%) reported in the same study.
SURVEY_TOTAL_YEARS = 5.2
SURVEY_CURRENT_AGE_YEARS = 2.6
COTTON_TOTAL_YEARS = SURVEY_TOTAL_YEARS - (0.168 * 1.2 + 0.145 * 5.5 / 12 + 0.029 * 9.7 / 12)
FIBRE_OFFSET_YEARS = {"cotton": 0.0, "denim": 0.0, "linen": 0.0, "polyester": 5.5 / 12, "silk": 2.2, "wool": 1.2}
WEARS_PER_YEAR = {"rarely": 12, "weekly": 52, "several_times_a_week": 156, "daily": 312}
WASHES_PER_WEEK = {"rarely": 1 / 12, "monthly": 0.25, "weekly": 1.0, "multiple_per_week": 3.0}
# Use effects are the unstandardised coefficients (B) of the study's final lifespan-in-years model (Block 4):
# each extra wear per year changes lifespan by -0.018 years (10 wears = about 2 months), and each higher laundering
# bracket adds 0.792 years (bracket 6 = 4.8 years longer than washing after every wear). The laundering brackets are
# 0 = after every wear, 1 = 2-3 wears, 2 = 4-5, 3 = 6-10, 4 = 11-19, 5 = 20-29 and 6 = 30 or more wears.
YEARS_PER_WEAR_PER_YEAR = -0.018
YEARS_PER_LAUNDERING_BRACKET = 0.792
LAUNDERING_BRACKET_STARTS = (2, 4, 6, 11, 20, 30)  # wears per wash at which brackets 1-6 begin
# The effects are measured from the survey's average garment, which the 5.2-year mean describes (Table 2, all
# countries): "once a week" is the median typical wear frequency, and the mean laundering bracket is 1.128
# (43.7% every wear, 27.6% 2-3, 15.3% 4-5, 6.3% 6-10, 2.6% 11-19, 1.7% 20-29, 2.8% 30 or more).
REFERENCE_WEARS_PER_YEAR = 52
REFERENCE_LAUNDERING_BRACKET = 0.276 + 2 * 0.153 + 3 * 0.063 + 4 * 0.026 + 5 * 0.017 + 6 * 0.028
# The survey's highest wear category was "several times a week", so the wear effect is not extended past it.
SURVEY_MAX_WEARS_PER_YEAR = WEARS_PER_YEAR["several_times_a_week"]
# The study's model explains about 22% of the variation in lifespan, so estimates are shown as a wide range.
RANGE_LOW, RANGE_HIGH = 0.6, 1.4
# Fabric Health Score weights: the share of the score each damage type removes. They come from the expert AHP
# survey (ahp.py, Saaty 1980), the method Benkirane et al. (2019) used to weight garment damage, and sum to 1.
HEALTH_WEIGHTS = CONFIG["health_weights"]["weights"]
HEALTH_WEIGHTS_SOURCE = CONFIG["health_weights"]["source"]
if abs(sum(HEALTH_WEIGHTS.values()) - 1) > 0.01:
    raise RuntimeError("assessment_config.json health_weights must sum to 1.")
# Grade bands of the condition-based health index of Jahromi et al. (2009): 85, 70 and 50 out of 100.
HEALTH_GRADES = ((85, "Excellent"), (70, "Good"), (50, "Fair"), (0, "Poor"))

REFERENCES = {
    "A": ("ISO 15487:2018 · Appearance after washing and drying", "https://www.iso.org/standard/67602.html"),
    "B": ("ISO 12945-4:2020 · Visual pilling, fuzzing and matting", "https://www.iso.org/standard/75377.html"),
    "C": ("Klepp et al. (2020) · Measuring clothing lifespan", "https://doi.org/10.3390/su12156219"),
    "D": ("Laitala and Klepp (2020) · Garment lifespan factors", "https://doi.org/10.3390/su12219151"),
    "E": ("ISO 3758:2023 · Care symbols", "https://www.iso.org/standard/74401.html"),
    "F": ("ISO 6330:2021 · Laundry procedures for textile testing", "https://www.iso.org/standard/75934.html"),
    "G": ("GINETEX · Garment care labelling", "https://www.ginetex.net/gb/labelling/care-symbols.asp"),
    "H": ("Woolmark · How to wash wool", "https://www.woolmark.com/care/how-to-wash-wool/"),
    "I": ("ISO 13934-1:2013 · Tensile strip test", "https://www.iso.org/standard/60676.html"),
    "J": ("Woolmark · Pilling and rubbing", "https://www.woolmark.com/care/pilling/"),
    "K": ("Levi’s · Denim care and washing", "https://help.levi.com/hc/en-us/articles/15463553339917-Denim-care-and-washing"),
    "L": ("ISO 13937-1:2000 · Fabric tear force", "https://www.iso.org/standard/23369.html"),
    "M": ("University of Georgia Extension · Fiber properties and care", "https://site.extension.uga.edu/textiles/textile-basics/understand-your-fibers/"),
    "N": ("Oregon State University Extension · Stain removal", "https://extension.oregonstate.edu/sites/extd8/files/2023-07/Oh%2C%20No%21%20Stains%21.pdf"),
    "P": ("Levi’s · Dark denim and dye transfer", "https://help.levi.com/hc/en-us/articles/34797976189709-Dark-denim-and-dye-transfer"),
    "Q": ("Cotton et al. (2020) · Colour change and dye transfer in colder, shorter wash cycles", "https://eprints.whiterose.ac.uk/id/eprint/154480/"),
    "R": ("Canadian Conservation Institute · Textile care and risks from water", "https://www.canada.ca/en/conservation-institute/services/preventive-conservation/guidelines-collections/textiles-costumes.html"),
    "S": ("Jahromi et al. (2009) · Condition-based health index and its grade bands", "https://doi.org/10.1109/MEI.2009.4802595"),
    "T": ("Benkirane et al. (2019) · Garment damage weighted by AHP", "https://doi.org/10.1177/1558925019854773"),
}

SUPPORT = {
    "hole": "An opening was marked. Its cause and the strength of the surrounding textile are unknown.",
    "pilling": "Surface pills were marked. This describes appearance in the photographed area.",
    "stain": "A mark resembling a stain was identified. Its cause and effect on the fibers are unknown.",
    "needlemark": "A mark resembling a needle mark was identified. Its origin and effect on strength are unknown.",
    "discoloration": "An area of faded or changed colour was marked. Its cause (bleach, sunlight, washing) is unknown.",
}
LABEL_FIELDS = {
    "wash": ("unknown", "conflicting", "professional_only", "dry_clean_only", "do_not_wash", "hand", "machine_normal",
             "machine_mild", "machine_very_mild"),
    "dry": ("unknown", "do_not_tumble", "line", "flat", "tumble_low", "tumble_normal"),
    "iron": ("unknown", "do_not_iron", "low", "medium", "high"),
    "bleach": ("unknown", "do_not_bleach", "oxygen_only", "allowed"),
    "professional": ("unknown", "do_not_dry_clean"),
}
WASHABLE = {"hand", "machine_normal", "machine_mild", "machine_very_mild"}


def assessment_today():
    return datetime.now(timezone(timedelta(hours=CONFIG["timezone_offset_hours"]))).date()


def normalize_user_inputs(value=None, today=None):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("user_inputs must be a JSON object")
    result = {}
    for key, options in CONFIG["history"].items():
        setting = value.get(key, "unspecified")
        if not isinstance(setting, str) or setting not in options:
            raise ValueError(f"Invalid {key}")
        result[key] = setting
    purchased = value.get("purchase_date")
    known = value.get("purchase_date_known", bool(purchased))
    if type(known) is not bool:
        raise ValueError("purchase_date_known must be true or false")
    if known and not purchased:
        raise ValueError("Enter the known purchase date or mark it unknown")
    if purchased is not None and purchased != "":
        try:
            purchased = date.fromisoformat(purchased)
        except (ValueError, TypeError):
            raise ValueError("purchase_date must be an ISO date (YYYY-MM-DD)") from None
        if purchased > (today or assessment_today()):
            raise ValueError("Purchase date cannot be in the future")
    result["purchase_date"] = purchased.isoformat() if purchased else None
    result["purchase_date_known"] = bool(purchased)
    for field in ("composition", "usage_category", "previous_repairs", "care_practices"):
        entry = value.get(field, "")
        if not isinstance(entry, str) or len(entry) > 250:
            raise ValueError(f"{field} must be text of at most 250 characters")
        result[field] = entry.strip()
    practice = value.get("laundering_practice", "unspecified")
    if practice not in ("unspecified", "gentle handling", "rubbing or wringing", "professional cleaning"):
        raise ValueError("Invalid laundering practice")
    result["laundering_practice"] = practice
    label = value.get("care_label", {})
    if not isinstance(label, dict):
        raise ValueError("care_label must be an object")
    normalized_label = {}
    for field, options in LABEL_FIELDS.items():
        entry = label.get(field, "unknown")
        if not isinstance(entry, str) or entry not in options:
            raise ValueError(f"Invalid care label {field}")
        normalized_label[field] = entry
    temperature = label.get("max_temperature_c")
    if temperature is not None and (
        type(temperature) is not int or not 1 <= temperature <= 95 or normalized_label["wash"] not in WASHABLE
    ):
        raise ValueError("Enter an explicit maximum of 1–95 °C only for a label that permits washing")
    normalized_label["max_temperature_c"] = temperature
    result["care_label"] = normalized_label
    return result



def section(status, provenance, explanation, refs, **fields):
    return dict(status=status, provenance=provenance, explanation=explanation, reference_ids=refs, **fields)


def photo_location(detection):
    view = detection.get("view")
    photo = {"front": "Front photo", "back": "Back photo", "close_up": "Close-up photo"}.get(view, "Photo")
    box = detection.get("bbox")
    if not box:
        return photo
    x, y, width, height = box
    column = min(2, int((x + width / 2) * 3))
    row = min(2, int((y + height / 2) * 3))
    return f"{photo} · {('upper', 'middle', 'lower')[row]} {('left', 'center', 'right')[column]}"


def condition_for(detections, detector_available, reviewed, quality_limited, coverage_limited):
    observations = []
    for detection in detections:
        kind = detection["type"]
        if kind not in SUPPORT:
            continue
        confirmed = detection.get("confirmed", reviewed)
        observations.append({
            "type": kind, "id": detection.get("id"), "view": detection.get("view"),
            "location": photo_location(detection), "confirmed": confirmed,
            "area_ratio": detection.get("area_ratio"),
            "model_score": detection.get("confidence") if not confirmed else None,
            "severity": None,
            "provenance": "user_reported" if confirmed else "model_predicted",
            "explanation": SUPPORT[kind],
        })
    confirmed_types = {item["type"] for item in observations if item["confirmed"]}
    pending = sum(not item["confirmed"] for item in observations)
    if "hole" in confirmed_types:
        category = "Opening confirmed · repair check suggested"
    elif "pilling" in confirmed_types:
        category = "Surface pilling confirmed"
    elif confirmed_types == {"discoloration"}:
        category = "Colour change confirmed"
    elif confirmed_types:
        category = "Surface marks confirmed"
    elif pending:
        names = sorted({item["type"] for item in observations})
        category = f"Possible {names[0]} · awaiting review" if len(names) == 1 else "Several possible findings · awaiting review"
    elif reviewed:
        category = "No supported findings confirmed in review"
    elif detector_available:
        category = "No supported findings flagged in the photos"
    else:
        category = "Damage assessment unavailable"
    provenance = ("mixed" if confirmed_types and pending else "user_reported" if confirmed_types or reviewed and not pending
                  else "model_predicted" if detector_available or pending else "unavailable")
    return section(
        GUIDANCE if confirmed_types else "experimental_estimate" if pending else INSUFFICIENT,
        provenance,
        "Severity and fabric strength need separate assessment. Box coverage measures the marked part of the photo "
        "and changes with framing. The scan covers only the photographed surfaces."
        + (" Photo quality limits the assessment." if quality_limited else "")
        + (" Detector coverage was incomplete." if coverage_limited else ""),
        ["A", "B", "I"], category=category, observations=observations, detector_available=detector_available,
        reviewed=reviewed, quality_limited=quality_limited, coverage_limited=coverage_limited,
    )


FABRIC_CONTEXT = {
    "cotton": ("Keep dark or strongly dyed items separate from lights before any permitted wash.",
               "Sorting by colour reduces dye transfer during laundering.", ["G"], True),
    "denim": ("Keep dark denim separate from pale items, including during wear if it transfers dye.",
              "Dark or raw denim can transfer dye to lighter fabric.", ["K", "P"], False),
    "linen": ("Check an apparent crease in even light before treating it as wear.",
              "Linen can crease naturally, and repeated sharp folds can affect the same area.", ["M"], False),
    "polyester": ("Keep the garment away from direct high heat while checking its care limits.",
                  "Polyester is heat-sensitive, and trims or finishes may need different handling.", ["M", "E"], False),
    "silk": ("Avoid rubbing or wet spot treatments while checking the garment instructions.",
             "Water can leave marks on some silk, and vigorous handling can affect wet fibers.", ["M"], False),
    "wool": ("Reduce rubbing from straps or rough layers where possible.",
             "Friction during wear can contribute to pilling on wool.", ["J"], False),
}
RESTRICTED_WASH = {"professional_only", "dry_clean_only", "do_not_wash"}

SUPPORTED_FABRICS_TEXT = "cotton, denim, linen, polyester, silk and wool"
# Fiber words on care labels: the supported fabric each counts as, or None when the app has no model for it.
LABEL_FIBERS = {
    "cotton": "cotton", "denim": "denim", "linen": "linen", "flax": "linen", "polyester": "polyester",
    "silk": "silk", "wool": "wool", "merino": "wool", "lambswool": "wool", "cashmere": "wool",
    "nylon": None, "polyamide": None, "spandex": None, "elastane": None, "lycra": None, "acrylic": None,
    "rayon": None, "viscose": None, "modal": None, "lyocell": None, "tencel": None, "acetate": None,
    "triacetate": None, "bamboo": None, "hemp": None, "ramie": None, "jute": None, "polypropylene": None,
    "olefin": None, "leather": None, "suede": None,
}
LABEL_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*%|\b(" + "|".join(sorted(LABEL_FIBERS, key=len, reverse=True)) + r")\b")


def label_fibers(composition):
    """Fibers named in the entered fiber content, in label order, as (fiber, percent or None, supported fabric).

    A percentage belongs to the fiber right after it ("80% nylon") or, failing that, right before it ("nylon 80%").
    """
    entries, waiting, last = [], None, None
    for number, word in LABEL_TOKEN.findall(composition.casefold()):
        if number:
            if last is not None:
                last[1], last = float(number), None
            else:
                waiting = float(number)
            continue
        fabric = LABEL_FIBERS[word]
        entry = [fabric or word, waiting, fabric]
        entries.append(entry)
        last, waiting = (entry if waiting is None else None), None
    merged = {}
    for name, percent, fabric in entries:
        if name in merged:
            if percent is not None:
                merged[name][1] = (merged[name][1] or 0) + percent
        else:
            merged[name] = [name, percent, fabric]
    return [tuple(entry) for entry in merged.values()]


def main_label_fiber(fibers):
    """The fiber with the largest share, or the first one listed when the label gives no percentages."""
    if not fibers:
        return None
    if any(percent is not None for _, percent, _ in fibers):
        return max(fibers, key=lambda fiber: -1 if fiber[1] is None else fiber[1])
    return fibers[0]


def fabric_support(composition, fabric_type, fabric_confidence, fabric_threshold=None):
    """Flags a garment that is, or may be, outside the six fabrics the app identifies.

    The entered fiber content decides when it names a fiber; otherwise a photo score below the review threshold
    means the garment may be an unsupported fabric, because the classifier only knows the six supported ones.
    """
    fibers = label_fibers(composition)
    main = main_label_fiber(fibers)
    if main is not None and main[2] is None:
        share = f"{main[1]:g}% " if main[1] is not None else ""
        return {"detected": True, "trigger": "fiber_label", "fiber": main[0], "title": "Unsupported fabric type",
                "message": f"The fiber content you entered is mainly {share}{main[0]}, which is not one of the six "
                           f"supported fabrics ({SUPPORTED_FABRICS_TEXT}). Fabric-specific care and lifespan are "
                           "not applied; follow the garment's care label."}
    if main is not None:
        others = [name for name, _, fabric in fibers if fabric is None]
        note = (f"Blend: mainly {main[0]} with {', '.join(others)}. The main fiber is supported." if others else None)
        return {"detected": False, "trigger": None, "fiber": main[0], "note": note}
    if isinstance(fabric_confidence, (int, float)) and fabric_confidence < review_threshold(fabric_threshold):
        return {"detected": True, "trigger": "low_confidence", "fiber": None, "title": "Possibly unsupported fabric",
                "message": f"The photo does not clearly match any of the six supported fabrics "
                           f"({SUPPORTED_FABRICS_TEXT}). It may be an unsupported fabric such as nylon or spandex. "
                           "Enter the fiber content from the care label, or retake the photo."}
    return {"detected": False, "trigger": None, "fiber": None, "note": None}


def guidance(action, reason, selected_because, refs, restriction=""):
    return {"action": action, "reason": reason, "selected_because": selected_because, "restriction": restriction,
            "reference_ids": refs, "status": GUIDANCE, "provenance": "rule_derived"}


def care_for(fabric_type, fabric_confidence, inputs, condition, fabric_threshold=None):
    label = inputs["care_label"]
    wash = label["wash"]
    temperature = label["max_temperature_c"]
    composition = inputs["composition"].casefold()
    fiber = next((kind for kind in FABRIC_CONTEXT
                  if kind != "denim" and composition in {kind, f"100% {kind}"}), None)
    if composition == "100% merino wool":
        fiber = "wool"
    main = main_label_fiber(label_fibers(composition))
    if fiber is None and main is not None and main[2] not in (None, "denim"):
        fiber = main[2]  # a blend's main fiber, e.g. "95% cotton, 5% spandex"
    unsupported_label = main is not None and main[2] is None
    verified_wool = composition in {"100% wool", "100% merino wool"}
    predicted = fabric_type.casefold()
    needs_confirmation = (fabric_confidence is None or fabric_confidence < review_threshold(fabric_threshold)
                          or condition["quality_limited"] or predicted not in FABRIC_CONTEXT or unsupported_label)
    # A fibre the user entered counts; a photo guess that needs confirmation chooses no fabric-specific advice.
    material = fiber or (None if needs_confirmation else predicted)
    conflicts = wash == "conflicting" or (wash == "dry_clean_only" and label["professional"] == "do_not_dry_clean")
    domestic = wash in WASHABLE and not conflicts
    restricted = conflicts or wash in RESTRICTED_WASH
    confirmed = {item["type"] for item in condition["observations"] if item["confirmed"]}
    pending = {item["type"] for item in condition["observations"] if not item["confirmed"]}
    advice, general, risks = [], [], []

    if wash == "unknown":
        general.append(guidance(
            "Set the garment aside from the wash and avoid bleach or direct heat while choosing a cleaning method.",
            "Water, heat and chemicals can affect dyes, finishes and garment components differently.",
            "A readable washing instruction was unavailable.", ["R", "M"],
            "Look for care details for the exact garment or ask a textile cleaner to assess it.",
        ))
        if (material == "cotton" and "hole" not in (confirmed | pending)
                and condition["detector_available"] and not condition["coverage_limited"]):
            general.append(guidance(
                "If this is a plain cotton T-shirt and you establish that it can be machine washed, "
                "consider a cooler, shorter cycle within its known limits.",
                "A study on retail T-shirts found less colour loss and dye transfer with the cooler, shorter cycle tested.",
                "The available material information suggests cotton and the scan has no opening flagged.", ["Q"],
                "General evidence for the tested garments and wash programs. Check construction, finishes and trim.",
            ))

    context = FABRIC_CONTEXT.get(material)
    if context and not (context[3] and restricted):
        action, reason, refs, _ = context
        general.append(guidance(
            action, reason,
            f"You entered {inputs['composition']} from the fiber label." if fiber else f"The photo suggests {material}.",
            refs, "General precaution. Use treatment limits established for this garment.",
        ))
    if domestic and material == "wool":
        if verified_wool:
            action = "Use a mild detergent suitable for wool."
        else:
            action = "If the fiber label confirms wool, use a mild wool detergent."
        general.append(guidance(action, "Woolmark recommends mild detergent for washable wool.",
                                "The entered care label permits washing and the material information suggests wool.",
                                ["H", "G"], "Keep the entered wash process and limits."))
    if not general:
        general.append(guidance(
            "Set aside bleach and direct heat while checking the garment instructions." if not restricted
            else "Keep the garment separate and follow the full cleaning instructions.",
            "The complete label describes limits for the most sensitive part of the garment.",
            "There is no applicable material-specific precaution for the entered restrictions.", ["E", "G"],
        ))

    if "hole" in confirmed:
        risks.append(guidance("Inspect or mend the opening before cleaning.",
                              "Pulling can enlarge an existing opening.",
                              "You confirmed an opening in the photo.", ["I", "L"],
                              "The garment’s tear strength is unknown."))
    if "pilling" in confirmed:
        risks.append(guidance("Limit rubbing on the pilled area.",
                              "Repeated friction can make pilling more noticeable.",
                              "You confirmed pilling in the photo.", ["B", "J"]))
    if "stain" in confirmed or "stain" in pending:
        if restricted:
            stain_action = "If professional cleaning is permitted, point out the mark. For now, leave home spot treatment on hold."
        elif wash == "unknown":
            stain_action = "Avoid strong stain treatments and hot water until you identify the mark and verify a suitable method."
        else:
            stain_action = "Avoid strong stain treatments until you identify the mark and check the garment instructions."
        advice.append(guidance(
            stain_action,
            "The best treatment depends on the type of stain, the fabric, and the garment instructions.",
            "You confirmed a stain." if "stain" in confirmed else "The detector marked a possible stain for you to check.",
            ["N", "G"],
        ))
    if "discoloration" in confirmed or "discoloration" in pending:
        advice.append(guidance(
            "Follow the professional-care instructions and point out the faded area." if restricted else
            "Wash in cold water, inside out, without chlorine bleach, and dry it away from direct sun.",
            "Heat, bleach and sunlight can fade dyes further; colder, shorter washes reduce colour change.",
            "You confirmed a colour change." if "discoloration" in confirmed
            else "The detector marked a possible colour change for you to check.",
            ["Q", "G"],
        ))
    if "needlemark" in confirmed:
        advice.append(guidance("Check the marked area for loose threads before cleaning.",
                               "A needle-like mark may be cosmetic. An opening or loose threads warrant closer inspection.",
                               "You confirmed a needle-like mark.", ["A", "L"]))
    if conflicts:
        risks.append(guidance("Recheck the conflicting cleaning instructions.",
                              "Resolving the conflict establishes which treatments are allowed.",
                              "The entered cleaning instructions conflict.", ["E", "G"]))
    if label["dry"] == "do_not_tumble":
        risks.append(guidance("Avoid tumble drying.", "The drying symbol limits the allowable treatment.",
                              "The entered label prohibits tumble drying.", ["E", "G"]))

    wash_modes = {
        "dry_clean_only": ("Dry clean only", "Follow the professional dry-cleaning instructions."),
        "professional_only": ("Professional care only", "Follow the professional-care instructions."),
        "do_not_wash": ("Do not wash", "Check the professional-care symbols."),
        "hand": ("Hand wash only", "Handle gently and avoid wringing."),
        "machine_normal": ("Machine wash permitted", "Normal process"),
        "machine_mild": ("Mild machine process", "Reduced agitation"),
        "machine_very_mild": ("Very mild machine process", "Very low agitation"),
    }
    mode, intensity = ("Verify the care label", "") if conflicts else wash_modes.get(wash, ("Unknown", ""))
    dry_methods = {"do_not_tumble": "Do not tumble dry.", "line": "Line dry as directed on the label.",
                   "flat": "Dry flat as directed on the label.", "tumble_low": "Low-heat tumble drying permitted.",
                   "tumble_normal": "Normal tumble drying permitted."}
    iron_methods = {"do_not_iron": "Do not iron.", "low": "Low limit on label", "medium": "Medium limit on label",
                    "high": "High limit on label"}
    bleach_methods = {"do_not_bleach": "Do not bleach.", "oxygen_only": "Oxygen bleach only",
                      "allowed": "Bleach allowed within the label instructions."}
    # Keep independent prohibitions when washing is unspecified or restricted.
    dry = dry_methods.get(label["dry"]) if not restricted or label["dry"] == "do_not_tumble" else None
    iron = iron_methods.get(label["iron"]) if not restricted or label["iron"] == "do_not_iron" else None
    bleach = bleach_methods.get(label["bleach"]) if not restricted or label["bleach"] == "do_not_bleach" else None
    professional = "Do not dry clean." if label["professional"] == "do_not_dry_clean" else None
    confirmed_settings = {"Wash": mode} if domestic or (restricted and not conflicts) else {}
    if domestic and temperature is not None:
        confirmed_settings["Maximum wash temperature"] = f"{temperature} °C"
    for field, value in (("Dry", dry), ("Iron", iron), ("Bleach", bleach), ("Professional care", professional)):
        if value:
            confirmed_settings[field] = value
    unknown = []
    if conflicts:
        unknown.append("Resolve the conflicting cleaning instructions before selecting a wash process.")
    elif wash == "unknown":
        unknown.append("Wash mode, temperature and detergent compatibility are unconfirmed. See general precautions for what to do now.")
    elif domestic and temperature is None:
        unknown.append("The label maximum wash temperature was not entered.")
    elif restricted:
        unknown.append("Home-wash settings are withheld under the entered restriction.")
    unset = [name for name in ("Dry", "Iron", "Bleach") if name not in confirmed_settings]
    if unset and wash != "unknown":
        unknown.append("Settings to verify on the full label: " + ", ".join(name.lower() for name in unset) + ".")
    laundry = section(
        GUIDANCE if confirmed_settings else INSUFFICIENT,
        "user_reported" if confirmed_settings else "rule_derived",
        "The missing washing instruction leaves home and professional cleaning compatibility unconfirmed. "
        "Use the precautions above while you establish a suitable method." if wash == "unknown" else
        "Entered label settings are treated as limits. Check the full symbols for details such as shade, steam and solvents.",
        ["E", "G"], laundry_mode=mode, intensity=intensity, confirmed_settings=confirmed_settings,
        general_guidance=general, unknown_settings=unknown,
        professional_care=professional or "Check the professional-care symbol.",
        max_temperature_c=temperature if domestic else None,
        detergent="Mild wool detergent" if domestic and material == "wool" and verified_wool else "Check label and detergent directions" if domestic else "Unavailable",
        dry_method=dry or ("Drying method unknown." if domestic else "Check complete label."),
        iron_level=iron or ("Ironing limit unknown." if domestic else "Check complete label."),
        bleach_warning=bleach or ("Bleach permission unknown." if domestic else "Check complete label."),
    )
    return [*advice, *general], laundry, risks, needs_confirmation


def health_assessment(condition):
    """FHS = 100 x (1 - sum of the AHP weights of the damage types present), graded by the Jahromi et al. bands.

    A damage type is present when the user confirmed it or the detector found it at or above its tested confidence
    level (care_model.DAMAGE_MIN_CONFIDENCE); weaker candidates stay for review and remove nothing. Each type counts
    once. Types without a weight (wrinkles, or discoloration until it is surveyed) do not change the score.
    """
    if not condition["detector_available"] and not condition["observations"]:
        return None
    deductions = []
    for kind, weight in HEALTH_WEIGHTS.items():
        items = [item for item in condition["observations"]
                 if item["type"] == kind and finding_counts(kind, item.get("model_score"), item["confirmed"])]
        if not items:
            continue
        confirmed = any(item["confirmed"] for item in items)
        deductions.append({"type": kind, "weight": weight, "points": round(100 * weight, 1),
                           "provenance": "user_reported" if confirmed else "model_predicted"})
    score = max(0, round(100 * (1 - sum(d["weight"] for d in deductions))))
    grade = next(name for floor, name in HEALTH_GRADES if score >= floor)
    return {"score": score, "grade": grade, "deductions": deductions, "weights_source": HEALTH_WEIGHTS_SOURCE}


def lifespan_estimate(fabric_type, condition, inputs, today=None):
    """Referenced remaining-lifespan estimate in months, shown as a range.

    Baselines and use effects come from Laitala and Klepp (2020); damage effects are FabriCare assumptions.
    """
    today = today or assessment_today()
    fabric = str(fabric_type).casefold()
    known_fabric = fabric in FIBRE_OFFSET_YEARS
    total = COTTON_TOTAL_YEARS + FIBRE_OFFSET_YEARS[fabric] if known_fabric else SURVEY_TOTAL_YEARS
    factors = [{"factor": f"Typical total lifespan for {fabric}" if known_fabric else "Typical total lifespan (fabric unconfirmed)",
                "effect_months": round(total * 12), "source": "D"}]
    wears = WEARS_PER_YEAR.get(inputs["use_frequency"])
    if wears is not None:
        years = YEARS_PER_WEAR_PER_YEAR * (min(wears, SURVEY_MAX_WEARS_PER_YEAR) - REFERENCE_WEARS_PER_YEAR)
        total += years
        capped = " (counted as several times a week, the survey's highest category)" \
            if wears > SURVEY_MAX_WEARS_PER_YEAR else ""
        factors.append({"factor": f"Wear frequency: {inputs['use_frequency'].replace('_', ' ')}{capped}",
                        "effect_months": round(years * 12), "source": "D"})
    washes = WASHES_PER_WEEK.get(inputs["washing_frequency"])
    if wears is not None and washes is not None:
        per_wash = max(1, round((wears / 52) / washes))
        bracket = sum(per_wash >= start for start in LAUNDERING_BRACKET_STARTS)
        years = YEARS_PER_LAUNDERING_BRACKET * (bracket - REFERENCE_LAUNDERING_BRACKET)
        total += years
        factors.append({"factor": f"About {per_wash} wear(s) between washes", "effect_months": round(years * 12),
                        "source": "D"})
    if inputs["purchase_date"]:
        age = max(0.0, (today - date.fromisoformat(inputs["purchase_date"])).days / 365.25)
        factors.append({"factor": f"Garment age: {age:.1f} years (purchase date)", "effect_months": -round(age * 12),
                        "source": "user"})
    elif inputs["condition"] == "new":
        age = 0.0
        factors.append({"factor": "Garment reported as new", "effect_months": 0, "source": "user"})
    else:
        age = SURVEY_CURRENT_AGE_YEARS
        factors.append({"factor": "Age unknown: average current garment age assumed", "effect_months": -round(age * 12),
                        "source": "D"})
    remaining = max(0.0, total - age) * 12
    # Visible damage scales the remaining time by the Fabric Health Score: a condition-based health index ties a
    # lower condition score to a shorter remaining life (Jahromi et al., 2009). Low-score candidates change nothing.
    health = health_assessment(condition)
    if health is not None and health["score"] < 100:
        before = remaining
        remaining *= health["score"] / 100
        factors.append({"factor": f"Visible condition: Fabric Health Score {health['score']}/100",
                        "effect_months": -round(before - remaining), "source": "health_index"})
    central = round(remaining)
    return {"central": central, "low": round(remaining * RANGE_LOW), "high": round(remaining * RANGE_HIGH),
            "factors": factors}


def durability_and_outlook(condition, inputs, fabric_type="uncertain", today=None):
    confirmed = {item["type"] for item in condition["observations"] if item["confirmed"]}
    pending = any(not item["confirmed"] for item in condition["observations"])
    if "hole" in confirmed:
        summary = "Check the confirmed opening for repair before further wear or cleaning."
    elif "pilling" in confirmed:
        summary = "Surface pilling is visible. Reduce rubbing in the affected area and monitor appearance."
    elif confirmed:
        summary = "A surface mark was confirmed. Check whether it changes or whether threads loosen over time."
    elif pending:
        summary = "Review the marked areas first to see whether any need repair or special handling."
    else:
        summary = "Check seams and high-friction areas, and save a baseline photo for a later comparison."
    frequent = inputs["washing_frequency"] == "multiple_per_week"
    often_worn = inputs["use_frequency"] in {"daily", "several_times_a_week"}
    practice = inputs["laundering_practice"]
    wash_unknown = inputs["care_label"]["wash"] == "unknown"
    assumptions = []
    for field, label in (("use_frequency", "wear frequency"), ("washing_frequency", "wash frequency")):
        if inputs[field] != "unspecified":
            assumptions.append(f"{label} is {inputs[field].replace('_', ' ')}")
    if practice != "unspecified":
        assumptions.append(f"handling is {practice.replace('_', ' ')}")
    concerns, refs = [], ["A", "D", "F"]
    if "hole" in confirmed:
        concerns.append("If the opening keeps catching or stretching during use, it could enlarge. Compare the same opening after repair or further wear.")
        refs.append("L")
    if "pilling" in confirmed:
        concerns.append(("With the reported frequent wear, check whether the same rubbed areas develop more pills." if often_worn else
                         "If rubbing continues, pilling may become more visible in the same areas."))
        refs.append("J")
    if "stain" in confirmed:
        concerns.append("Save a photo of the mark now and compare it after any cleaning established as suitable."
                        if wash_unknown else
                        "Photograph the mark after a label-compatible cleaning to see whether it persists.")
    if "discoloration" in confirmed:
        concerns.append("Rescan the faded area in the same light after a few washes to see whether the colour change spreads.")
    if frequent or practice == "rubbing or wringing":
        concerns.append("Repeated laundering adds opportunities for appearance change. Use a method established as suitable and compare the same area across scans."
                        if practice != "rubbing or wringing" else
                        "If rubbing or wringing continues during cleaning, check the same areas for appearance changes and reassess the cleaning method.")
        refs.append("G")
    if not concerns:
        concerns.append("Use the same lighting and framing when you rescan. Compare seams, frequently rubbed areas and any marked spots.")
    estimate = lifespan_estimate(fabric_type, condition, inputs, today)
    lifespan = section(
        ESTIMATE, "reference_estimate",
        "An estimate, not a prediction for this garment. It starts from typical garment lifespans in a five-country "
        "wardrobe survey (Laitala and Klepp, 2020), adjusts for fibre, wear and washing frequency, subtracts the "
        "garment's age and scales the result by the Fabric Health Score, so visible damage shortens it. That survey's "
        "model explains about a fifth of the variation between garments, so the range is wide.",
        ["C", "D", "I", "S", "T"] + (["L"] if "hole" in confirmed else []),
        remaining_months=estimate["central"], remaining_months_low=estimate["low"],
        remaining_months_high=estimate["high"], factors=estimate["factors"], summary=summary,
    )
    outlook = section(
        INSUFFICIENT, "qualitative_scenario",
        "A six-month forecast requires dated repeat observations and an independently evaluated prediction method.",
        refs, current_condition=condition["category"],
        assumptions=("Assumes the reported " + ", ".join(assumptions) + " and those habits continue."
                     if assumptions else "Assumes the garment continues to be used. A cleaning method has not been established."
                     if wash_unknown else "Assumes the garment continues to be used and cleaning follows its label."),
        concerns=concerns,
    )
    return lifespan, outlook


def assess(fabric_type, detections, user_inputs=None, today=None, *, fabric_confidence=None, fabric_threshold=None,
           detector_available=True, reviewed=False, quality_limited=False, coverage_limited=False):
    inputs = normalize_user_inputs(user_inputs, today=today)
    condition = condition_for(detections, detector_available, reviewed, quality_limited, coverage_limited)
    advice, laundry, risks, needs_confirmation = care_for(fabric_type, fabric_confidence, inputs, condition,
                                                          fabric_threshold)
    # A fabric that needs confirmation does not set the lifespan baseline; the survey mean is used instead.
    lifespan, outlook = durability_and_outlook(condition, inputs, "uncertain" if needs_confirmation else fabric_type,
                                               today)
    health = health_assessment(condition)
    usage = [f"{name}: {inputs[field].replace('_', ' ')}"
             for field, name in (("condition", "Garment"), ("use_frequency", "Wear frequency"),
                                 ("washing_frequency", "Wash frequency")) if inputs[field] != "unspecified"]
    if inputs["purchase_date"]:
        usage.append(f"Purchased {inputs['purchase_date']}")
    usage_status = "User-reported · " + " · ".join(usage) if usage else "Usage history not supplied."
    ref_ids = sorted({ref for entry in [condition, lifespan, outlook, laundry, *advice, *risks]
                      for ref in entry["reference_ids"]})
    return {
        "methodology_version": METHOD_VERSION,
        "fabric_needs_confirmation": needs_confirmation,
        "unsupported_fabric": fabric_support(inputs["composition"], fabric_type, fabric_confidence, fabric_threshold),
        "condition_assessment": condition,
        "lifespan": lifespan,
        "six_month_outlook": outlook,
        "care_guidance": {"status": GUIDANCE, "provenance": "rule_derived", "recommendations": advice},
        "laundry_settings": laundry,
        "risk_alerts": risks,
        "references": [{"id": ref, "title": REFERENCES[ref][0], "url": REFERENCES[ref][1]} for ref in ref_ids],
        "user_inputs": inputs, "usage_status": usage_status,
        "health_assessment": health,
        "health_score": health["score"] if health else None,
        "health_grade": health["grade"] if health else None,
        "condition_rating": condition["category"],
        "estimated_lifespan": (f"About {lifespan['remaining_months_low']}–{lifespan['remaining_months_high']} "
                               "months remaining (estimate)"),
        "estimated_remaining_months": lifespan["remaining_months"],
        "estimate_method": "reference_estimate", "projected_condition_after_6_months": None,
        "care_recommendations": [entry["action"] for entry in advice],
        "care_recommendation": {**laundry, "care_advice": " ".join(entry["action"] for entry in advice)},
    }
