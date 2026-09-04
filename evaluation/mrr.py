def reciprocal_rank(rank: int | None) -> float:
    """Return reciprocal rank for one query, or 0.0 when no match was retrieved."""
    if rank is None:
        return 0.0
    return 1.0 / rank


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Return the mean reciprocal rank across evaluation queries."""
    if not ranks:
        return 0.0
    return sum(reciprocal_rank(rank) for rank in ranks) / len(ranks)
