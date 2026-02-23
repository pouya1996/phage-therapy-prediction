"""
Phage Ranker — rank phages by predicted CI (Cocktail Interaction) score.

Supports two views:
  1. **Phage view** (normal): Best score per phage across concentrations.
  2. **Interaction view** (detailed): Every phage × concentration ranked.

USAGE:
    from src.prediction.ranker import PhageRanker

    ranker = PhageRanker(interaction_info, probabilities)
    top = ranker.get_top_k_phages(k=5)
    report = ranker.generate_recommendation_report()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_THRESHOLD = 0.5


class PhageRanker:
    """Rank phage candidates by CI score."""

    def __init__(
        self,
        interaction_info: List[Dict[str, Any]],
        probabilities: "np.ndarray",  # noqa: F821
        model_name: str = "unknown",
    ):
        """
        Args:
            interaction_info: List of dicts with keys
                ``phage_id``, ``morphology``, ``concentration``.
            probabilities: Array of CI scores (same length as interaction_info).
            model_name: Name of the model that produced the scores.
        """
        self.model_name = model_name
        self.df = pd.DataFrame(interaction_info)
        self.df["ci_score"] = np.asarray(probabilities).ravel()

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def rank_phages(self, view: str = "phage") -> pd.DataFrame:
        """
        Rank phages by CI score.

        Args:
            view:
              ``"phage"``  — every phage × concentration combination,
                             ranked by CI score (all combos visible).
              ``"interaction"`` — one row per unique phage, keeping only
                                  the best-scoring interaction.

        Returns:
            Sorted DataFrame (descending CI score).
        """
        if view == "interaction":
            return self._rank_best_per_phage()
        return self._rank_all_combos()

    def _rank_all_combos(self) -> pd.DataFrame:
        """All phage × concentration combos ranked by CI score."""
        ranked = self.df.sort_values("ci_score", ascending=False).reset_index(
            drop=True
        )
        ranked.index = ranked.index + 1
        ranked.index.name = "rank"
        return ranked[["phage_id", "morphology", "concentration", "ci_score"]]

    def _rank_best_per_phage(self) -> pd.DataFrame:
        """Unique phages only — best CI score across concentrations."""
        idx = self.df.groupby("phage_id")["ci_score"].idxmax()
        ranked = self.df.loc[idx].sort_values("ci_score", ascending=False)
        ranked = ranked.reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = "rank"
        return ranked[["phage_id", "morphology", "concentration", "ci_score"]]

    # ------------------------------------------------------------------
    # Top-K
    # ------------------------------------------------------------------
    def get_top_k_phages(
        self, k: int = 5, view: str = "phage"
    ) -> pd.DataFrame:
        """
        Return the top *k* candidates.

        Args:
            k: Number of top candidates.
            view: ``"phage"`` or ``"interaction"``.

        Returns:
            DataFrame with the top-k rows.
        """
        ranked = self.rank_phages(view=view)
        return ranked.head(k)

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------
    @staticmethod
    def check_feasibility(
        score: float, threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Return True if the CI score indicates a viable treatment."""
        return score >= threshold

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_recommendation_report(
        self,
        top_k: int = 10,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> str:
        """
        Generate a human-readable recommendation report.

        Args:
            top_k: Number of phages to include.
            threshold: Minimum CI score for a viable candidate.

        Returns:
            Formatted multi-line string.
        """
        ranked = self.rank_phages(view="phage")
        top = ranked.head(top_k)

        lines = [
            "=" * 70,
            "PHAGE THERAPY RECOMMENDATION REPORT",
            f"Model: {self.model_name}",
            f"Total phages evaluated: {len(ranked):,}",
            f"Feasibility threshold : {threshold}",
            "=" * 70,
            "",
        ]

        viable = 0
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            feasible = self.check_feasibility(row["ci_score"], threshold)
            viable += int(feasible)
            tag = "VIABLE" if feasible else "below threshold"
            lines.append(
                f"  {rank:>3}. {row['phage_id']:<20s}  "
                f"CI={row['ci_score']:.4f}  "
                f"Conc={row['concentration']:<10g}  "
                f"Morph={row['morphology']:<15s}  "
                f"[{tag}]"
            )

        lines.append("")
        lines.append(f"Viable candidates (>= {threshold}): {viable} / {top_k}")

        # Detailed concentrations for top phage
        best_phage = top.iloc[0]["phage_id"]
        conc_rows = (
            self.df[self.df["phage_id"] == best_phage]
            .sort_values("concentration")
        )
        lines.append("")
        lines.append(f"Concentration breakdown for top phage ({best_phage}):")
        lines.append(f"  {'Concentration':<15s}  {'CI Score':<10s}  Status")
        lines.append(f"  {'-'*15}  {'-'*10}  {'-'*15}")
        for _, r in conc_rows.iterrows():
            tag = "VIABLE" if r["ci_score"] >= threshold else "-"
            lines.append(
                f"  {r['concentration']:<15g}  {r['ci_score']:<10.4f}  {tag}"
            )

        lines.append("=" * 70)

        report = "\n".join(lines)
        logger.info(f"Generated report for top-{top_k} phages")
        return report

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(
        self, view: str = "phage", top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Return rankings as a list of dicts (for JSON serialisation).

        Args:
            view: ``"phage"`` or ``"interaction"``.
            top_k: Limit rows (None = all).

        Returns:
            List of dicts with keys: rank, phage_id, morphology,
            concentration, ci_score, feasible.
        """
        ranked = self.rank_phages(view=view)
        if top_k:
            ranked = ranked.head(top_k)

        records = []
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            records.append({
                "rank": rank,
                "phage_id": row["phage_id"],
                "morphology": row["morphology"],
                "concentration": row["concentration"],
                "ci_score": round(float(row["ci_score"]), 6),
                "feasible": self.check_feasibility(row["ci_score"]),
            })
        return records
