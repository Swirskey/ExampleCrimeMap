import numpy as np

# This can be adjusted to get better results. for now we multiply like this
SCALE = 7684


def weighted_score(hour, month, distances, closest_landmarks):
    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 0 and 23.")
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12.")
    if len(distances) != len(closest_landmarks):
        raise ValueError("Distances and landmarks lists must have the same length.")

    total_weighted_score = 0
    total_weight = 0

    inverse_distances = [1 / distance if distance != 0 else float('inf') for distance in distances]

    total_inverse_distance = sum(inverse_distances)

    normalized_weights = [inverse_distance / total_inverse_distance for inverse_distance in inverse_distances]

    # Step 4: Calculate the weighted score based on these normalized weights
    for i, landmark in enumerate(closest_landmarks):
        # Extract weights and danger score from the landmark's properties

        time_weight = landmark["weights"]["hourly_weights"][hour]
        month_weight = landmark["weights"]["monthly_weights"][month - 1]  # Month is 1-indexed
        danger_score = landmark["score"]
        normalized_weight = normalized_weights[i]

        # Calculate the weighted severity score for this landmark
        weighted_severity = danger_score * time_weight * month_weight
        # Weight it by the normalized inverse distance weight
        total_weighted_score += weighted_severity * normalized_weight

        total_weight += normalized_weight

    # Step 5: Calculate the average weighted by the normalized weights
    if total_weight == 0:
        raise ValueError("Total weight is zero, possibly due to invalid distances.")

    average_weighted_score = total_weighted_score / total_weight
    average_weighted_score *= SCALE
    return average_weighted_score


def weighted_score_batch(hour, month, distances, indices, scores, hourly_weights, monthly_weights):
    """
    Vectorized form of :func:`weighted_score` for many points at once.

    ``distances`` and ``indices`` are ``(n, k)`` arrays from
    ``KDTreeCache.query_batch``; the remaining arguments are the precomputed
    landmark tables held on the cache. Returns an ``(n,)`` array of scores.
    """
    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 0 and 23.")
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12.")

    distances = np.atleast_2d(np.asarray(distances, dtype=float))
    indices = np.atleast_2d(np.asarray(indices, dtype=np.intp))

    with np.errstate(divide="ignore"):
        inverse_distances = 1.0 / distances

    # A point sitting exactly on a landmark has infinite inverse distance. Give
    # those landmarks the whole weight rather than propagating inf/inf.
    exact_hits = np.isinf(inverse_distances)
    rows_with_hits = exact_hits.any(axis=1)
    if rows_with_hits.any():
        inverse_distances[rows_with_hits] = exact_hits[rows_with_hits].astype(float)

    total_inverse_distance = inverse_distances.sum(axis=1, keepdims=True)
    if not np.all(total_inverse_distance > 0):
        raise ValueError("Total weight is zero, possibly due to invalid distances.")
    normalized_weights = inverse_distances / total_inverse_distance

    weighted_severity = (
        scores[indices]
        * hourly_weights[indices, hour]
        * monthly_weights[indices, month - 1]
    )

    # The normalized weights sum to 1 per row, so the weighted average is the sum.
    return (weighted_severity * normalized_weights).sum(axis=1) * SCALE
