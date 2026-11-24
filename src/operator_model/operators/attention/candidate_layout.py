"""Offline candidate address statistics for one Flash sparse-indexer request."""
from collections import Counter
from operator import index


def summarize_candidate_layout(rows, *, pa_block_size=128, page_table=None):
    """Summarize valid ordered 8-key block IDs, one iterable per query.

    Slice each row to its valid candidate count before calling. A supplied PA
    table maps logical page indices to physical pages in the same allocation.
    Without it, the result uses logical addresses within this request. Duplicate
    IDs retain their ranks. Rows from separate requests must be analyzed apart.
    """
    page_size = index(pa_block_size)
    if page_size <= 0 or page_size % 8:
        raise ValueError("pa_block_size must be a positive multiple of eight")
    blocks_per_page = page_size // 8
    rank_counts = []
    rank_totals = []
    matching = comparable = row_count = candidate_count = 0
    distinct_pages = set()
    for row in rows:
        row_count += 1
        for rank, value in enumerate(row):
            block = index(value)
            if block < 0:
                raise ValueError("pass only valid 8-key block IDs; slice off padding")
            page, offset = divmod(block, blocks_per_page)
            if page_table is not None:
                if page >= len(page_table):
                    raise ValueError("candidate block exceeds the page table")
                page = index(page_table[page])
                if page < 0:
                    raise ValueError("selected page has no physical mapping")
            if rank == len(rank_counts):
                rank_counts.append(Counter())
                rank_totals.append(0)
            record = page * blocks_per_page + offset
            matching += rank_counts[rank][record]
            comparable += rank_totals[rank]
            rank_counts[rank][record] += 1
            rank_totals[rank] += 1
            candidate_count += 1
            distinct_pages.add(page)
    return dict(candidate_order_overlap=matching / comparable if comparable else None,
                matching_pairs=matching, comparable_pairs=comparable,
                queries=row_count, valid_candidates=candidate_count,
                distinct_pages=len(distinct_pages),
                address_basis="physical" if page_table is not None else "logical")
