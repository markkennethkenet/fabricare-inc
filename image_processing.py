"""Image quality and per-image detection deduplication; no training involved."""
import cv2
from assessment import CONFIG


def quality_for(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    policy = CONFIG['quality']
    reasons = []
    if blur < policy['minimum_blur_score']:
        reasons.append('Image is blurry, hold steady')
    if brightness < policy['minimum_brightness']:
        reasons.append('Too dark, move to better light')
    if brightness > policy['maximum_brightness']:
        reasons.append('Too bright, avoid direct glare')
    return {'blur_score': round(blur, 2), 'brightness': round(brightness, 2),
            'passed': not reasons, 'reason': '. '.join(reasons)}


def box_iou(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    width = max(0, min(a[0] + a[2], b[0] + b[2]) - x)
    height = max(0, min(a[1] + a[3], b[1] + b[3]) - y)
    intersection = width * height
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0


def merge_detections(detections):
    """NMS: keep highest confidence same-type box per overlapping region in ONE image.

    Different views/images are never merged: 2D coordinates cannot establish
    cross-view identity. The same physical finding may appear in multiple views.
    """
    kept = []
    for detection in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        duplicate = any(detection['image_index'] == old['image_index']
                        and detection['type'] == old['type']
                        and box_iou(detection['bbox'], old['bbox']) >= CONFIG['detection']['merge_iou']
                        for old in kept)
        if not duplicate:
            kept.append(detection)
    return sorted(kept, key=lambda d: (d['image_index'], d['id']))


def aggregate_fabrics(predictions):
    """Average fabric scores and abstain when the front and back disagree.

    Without model scores, select by vote and keep the score unavailable.
    Agreeing front/back views take precedence over a dissenting close-up.
    Labels come from model/config metadata, never a fixed client class list.
    """
    views = {p.get('view'): p for p in predictions}
    front, back, detail = (views.get(view) for view in ('front', 'back', 'close_up'))
    close_up_excluded = bool(
        front and back and detail
        and front.get('quality', {}).get('passed', True)
        and back.get('quality', {}).get('passed', True)
        and front['fabric_type'] == back['fabric_type'] != detail['fabric_type']
    )
    included = [front, back] if close_up_excluded else predictions
    if all(p.get('score_kind') == 'unavailable' for p in included):
        votes = [p['fabric_type'] for p in included]
        disagrees = front is not None and back is not None and front['fabric_type'] != back['fabric_type']
        return {
            'fabric_type': 'uncertain' if disagrees else max(votes, key=votes.count),
            'fabric_confidence': None,
            'fabric_probabilities': {},
            'fabric_close_up_excluded': close_up_excluded,
            **({'fabric_uncertainty_reason': 'front_back_disagree'} if disagrees else {}),
        }
    labels = sorted({label for prediction in included for label in prediction['probabilities']})
    averages = {label: sum(p['probabilities'].get(label, 0) for p in included) / len(included)
                for label in labels}
    if front and back and front['fabric_type'] != back['fabric_type']:
        return {'fabric_type': 'uncertain', 'fabric_confidence': None,
                'fabric_probabilities': averages, 'fabric_close_up_excluded': False,
                'fabric_uncertainty_reason': 'front_back_disagree'}
    winner = max(averages, key=averages.get)
    return {'fabric_type': winner, 'fabric_confidence': round(averages[winner] * 100, 2),
            'fabric_probabilities': averages, 'fabric_close_up_excluded': close_up_excluded}
