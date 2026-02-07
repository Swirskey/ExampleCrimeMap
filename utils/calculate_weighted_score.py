import ast
import json
import math

def _parse_weights(v):
    """Return weights as a dict with hourly_weights/monthly_weights lists, or raise ValueError."""
    if v is None:
        raise ValueError("Missing weights")

    # NaN check
    try:
        if isinstance(v, float) and math.isnan(v):
            raise ValueError("Weights is NaN")
    except Exception:
        pass

    # Already a dict
    if isinstance(v, dict):
        return v

    # If it's a string, try JSON then Python literal
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError("Weights is empty string")
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            raise ValueError(f"Could not parse weights string: {e}")

    raise ValueError(f"Unsupported weights type: {type(v)}")


def weighted_score(hour, month, distances, closest_landmarks):
    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 0 and 23.")
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12.")
    if len(distances) != len(closest_landmarks):
        raise ValueError("Distances and landmarks lists must have the same length.")

    # Inverse distance weighting
    inverse_distances = [1 / d if d != 0 else float("inf") for d in distances]
    total_inverse_distance = sum(inverse_distances)
    normalized_weights = [inv / total_inverse_distance for inv in inverse_distances]

    total_weighted_score = 0.0
    total_weight = 0.0

    for i, landmark in enumerate(closest_landmarks):
       
        weights = _parse_weights(landmark.get("weights"))

        hourly = weights.get("hourly_weights")
        monthly = weights.get("monthly_weights")
        if not isinstance(hourly, list) or len(hourly) < 24:
            raise ValueError("Invalid hourly_weights")
        if not isinstance(monthly, list) or len(monthly) < 12:
            raise ValueError("Invalid monthly_weights")

        time_weight = hourly[hour]
        month_weight = monthly[month - 1]  # month is 1-indexed
        danger_score = landmark.get("score", 0)
        normalized_weight = normalized_weights[i]

        weighted_severity = danger_score * time_weight * month_weight
        total_weighted_score += weighted_severity * normalized_weight
        total_weight += normalized_weight

    if total_weight == 0:
        raise ValueError("Total weight is zero, possibly due to invalid distances.")

    average_weighted_score = total_weighted_score / total_weight
    average_weighted_score *= 7684
    return average_weighted_score
