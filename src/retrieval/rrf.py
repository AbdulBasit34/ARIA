def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int,
    limit: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
