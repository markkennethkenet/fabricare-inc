"""Versioned condition and care rules for the scan and damage-review endpoints."""
import json
import math
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from model_analysis import confidence_threshold

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
# Ten more wears per year shortened lifespan by about two months; washing less often than every 30 wears added up
# to 4.8 years compared with washing after every wear (section 4.1).
WEARS_PER_YEAR = {"rarely": 12, "weekly": 52, "several_times_a_week": 156, "daily": 312}
WASHES_PER_WEEK = {"rarely": 1 / 12, "monthly": 0.25, "weekly": 1.0, "multiple_per_week": 3.0}
REFERENCE_WEARS_PER_YEAR = 52
REFERENCE_WEARS_PER_WASH = 3
# The study's model explains about 22% of the variation in lifespan, so estimates are shown as a wide range.
RANGE_LOW, RANGE_HIGH = 0.6, 1.4
# FabriCare assumptions, not study figures: how visible damage shortens the remaining estimate.
DAMAGE_LIFESPAN_FACTOR = {"hole": (0.75, None), "pilling": (0.9, 0.8), "stain": (0.95, 0.9), "needlemark": (0.98, 0.95)}
CONFIRMED_HOLE_MAX_MONTHS = 6
# FabriCare visible-condition health score: points removed per finding type (candidate, confirmed).
HEALTH_DEDUCTIONS = {"hole": (20, 40), "pilling": (10, 20), "stain": (8, 15), "needlemark": (5, 10)}
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
}

SUPPORT = {
    "hole": "An opening was marked. Its cause and the strength of the surrounding textile are unknown.",
    "pilling": "Surface pills were marked. This describes appearance in the photographed area.",
    "stain": "A mark resembling a stain was identified. Its cause and effect on the fibers are unknown.",
    "needlemark": "A mark resembling a needle mark was identified. Its origin and effect on strength are unknown.",
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


def guidance(action, reason, selected_because, refs, restriction=""):
    return {"action": action, "reason": reason, "selected_because": selected_because, "restriction": restriction,
            "reference_ids": refs, "status": GUIDANCE, "provenance": "rule_derived"}


def care_for(fabric_type, fabric_confidence, inputs, condition):
    label = inputs["care_label"]
    wash = label["wash"]
    temperature = label["max_temperature_c"]
    composition = inputs["composition"].casefold()
    fiber = next((kind for kind in FABRIC_CONTEXT
                  if kind != "denim" and composition in {kind, f"100% {kind}"}), None)
    if composition == "100% merino wool":
        fiber = "wool"
    verified_wool = composition in {"100% wool", "100% merino wool"}
    predicted = fabric_type.casefold()
    material = fiber or predicted
    needs_confirmation = (fabric_confidence is None or fabric_confidence < confidence_threshold()
                          or condition["quality_limited"] or predicted not in FABRIC_CONTEXT)
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
        prefix = f"If this garment is {material}, " if not fiber and needs_confirmation else ""
        general.append(guidance(
            prefix + (action[0].lower() + action[1:] if prefix else action), reason,
            f"You entered {inputs['composition']} from the fiber label." if fiber else
            f"The photo suggests {material}" + (" and the identification needs confirmation." if needs_confirmation else "."),
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
    """Score the visible condition from 0 to 100 using the damage findings.

    Each finding type removes points once; confirmed findings count fully and candidates count about half.
    Findings covering under 2% of the photo remove 70% of the points. The weights are FabriCare assumptions.
    """
    if not condition["detector_available"] and not condition["observations"]:
        return None
    deductions = []
    for kind, (candidate_points, confirmed_points) in HEALTH_DEDUCTIONS.items():
        items = [item for item in condition["observations"] if item["type"] == kind]
        if not items:
            continue
        confirmed = any(item["confirmed"] for item in items)
        area = max((item.get("area_ratio") or 0) for item in items)
        points = (confirmed_points if confirmed else candidate_points) * (1.0 if area >= 0.02 else 0.7)
        deductions.append({"type": kind, "points": round(points, 1),
                           "provenance": "user_reported" if confirmed else "model_predicted"})
    score = max(0, round(100 - sum(d["points"] for d in deductions)))
    grade = next(name for floor, name in HEALTH_GRADES if score >= floor)
    return {"score": score, "grade": grade, "deductions": deductions}


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
        months = max(-24.0, min(12.0, -2.0 * (wears - REFERENCE_WEARS_PER_YEAR) / 10))
        total += months / 12
        factors.append({"factor": f"Wear frequency: {inputs['use_frequency'].replace('_', ' ')}",
                        "effect_months": round(months), "source": "D"})
    washes = WASHES_PER_WEEK.get(inputs["washing_frequency"])
    if wears is not None and washes is not None:
        per_wash = max(1.0, (wears / 52) / washes)

        def washing_benefit(value):
            return 4.8 * math.log(min(value, 30)) / math.log(30)

        years = max(-2.0, min(2.0, washing_benefit(per_wash) - washing_benefit(REFERENCE_WEARS_PER_WASH)))
        total += years
        factors.append({"factor": f"About {per_wash:.0f} wear(s) between washes", "effect_months": round(years * 12),
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
    confirmed = {item["type"] for item in condition["observations"] if item["confirmed"]}
    pending = {item["type"] for item in condition["observations"] if not item["confirmed"]}
    for kind, (candidate_factor, confirmed_factor) in DAMAGE_LIFESPAN_FACTOR.items():
        if kind in confirmed and confirmed_factor is not None:
            factor = confirmed_factor
        elif kind in pending or kind in confirmed:
            factor = candidate_factor
        else:
            continue
        before = remaining
        remaining *= factor
        factors.append({"factor": f"{'Confirmed' if kind in confirmed else 'Possible'} {kind}",
                        "effect_months": -round(before - remaining), "source": "fabricare_assumption"})
    if "hole" in confirmed and remaining > CONFIRMED_HOLE_MAX_MONTHS:
        factors.append({"factor": "Confirmed opening: repair suggested before further wear",
                        "effect_months": -round(remaining - CONFIRMED_HOLE_MAX_MONTHS), "source": "fabricare_assumption"})
        remaining = CONFIRMED_HOLE_MAX_MONTHS
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
        "garment's age and reduces the result for visible damage. That survey's model explains about a fifth of the "
        "variation between garments, so the range is wide. Damage adjustments are FabriCare assumptions.",
        ["C", "D", "I"] + (["L"] if "hole" in confirmed else []),
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


def assess(fabric_type, detections, user_inputs=None, today=None, *, fabric_confidence=None,
           detector_available=True, reviewed=False, quality_limited=False, coverage_limited=False):
    inputs = normalize_user_inputs(user_inputs, today=today)
    condition = condition_for(detections, detector_available, reviewed, quality_limited, coverage_limited)
    advice, laundry, risks, needs_confirmation = care_for(fabric_type, fabric_confidence, inputs, condition)
    lifespan, outlook = durability_and_outlook(condition, inputs, fabric_type, today)
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
