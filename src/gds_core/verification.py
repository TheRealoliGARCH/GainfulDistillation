"""Independent, auditable adjudication of claims produced within GDS.

Verification is based on distinct independent review groups, not on the
hierarchical position or popularity of the originating agent. Quorum values
are configurable governance parameters, not universal statistical constants.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class VerificationStatus(str, Enum):
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ReviewVerdict(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ABSTAINS = "abstains"


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Finding:
    """A provenance-bearing claim submitted for independent evaluation."""
    finding_id: str
    origin_agent: str
    claim: str
    evidence_ids: tuple[str, ...]
    method: str

    def __post_init__(self) -> None:
        for name in ("finding_id", "origin_agent", "claim", "method"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        evidence = tuple(_required_text(item, "evidence_id") for item in self.evidence_ids)
        if not evidence:
            raise ValueError("a finding must reference at least one evidence item")
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", evidence)


@dataclass(frozen=True)
class ValidatorReview:
    """One review by a validator belonging to an independence group."""
    finding_id: str
    validator_id: str
    independence_group: str
    verdict: ReviewVerdict
    rationale: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("finding_id", "validator_id", "independence_group", "rationale"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        try:
            verdict = ReviewVerdict(self.verdict)
        except ValueError as exc:
            raise ValueError("verdict must be supports, contradicts, or abstains") from exc
        object.__setattr__(self, "verdict", verdict)
        refs = tuple(_required_text(item, "review evidence_id") for item in self.evidence_ids)
        if len(set(refs)) != len(refs):
            raise ValueError("review evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", refs)


@dataclass(frozen=True)
class VerificationDecision:
    """Decision plus the complete review record used to reach it."""
    finding_id: str
    status: VerificationStatus
    supporting_reviews: int
    contradicting_reviews: int
    abstaining_reviews: int
    independent_review_groups: int
    reviews: tuple[ValidatorReview, ...]
    rationale: str


def adjudicate(
    finding: Finding,
    reviews: Sequence[ValidatorReview],
    *,
    minimum_independent_reviewers: int = 2,
    support_quorum: int = 2,
    reject_quorum: int = 2,
) -> VerificationDecision:
    """Adjudicate a finding using distinct validators and independence groups.

    The origin agent cannot review its own finding. Each independence group
    may contribute at most one review, preventing correlated reviewers from
    being counted as independent votes. Conflicting quorum-level evidence
    produces an inconclusive result rather than an arbitrary tie-break.
    """
    for name, value in (
        ("minimum_independent_reviewers", minimum_independent_reviewers),
        ("support_quorum", support_quorum),
        ("reject_quorum", reject_quorum),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if support_quorum > minimum_independent_reviewers or reject_quorum > minimum_independent_reviewers:
        raise ValueError("quorums cannot exceed minimum_independent_reviewers")
    checked = tuple(reviews)
    validator_ids: set[str] = set()
    groups: set[str] = set()
    for review in checked:
        if review.finding_id != finding.finding_id:
            raise ValueError("all reviews must refer to the adjudicated finding")
        if review.validator_id == finding.origin_agent:
            raise ValueError("the originating agent cannot validate its own finding")
        if review.validator_id in validator_ids:
            raise ValueError("a validator may submit only one review per finding")
        if review.independence_group in groups:
            raise ValueError("each independence group may submit only one review per finding")
        validator_ids.add(review.validator_id)
        groups.add(review.independence_group)

    supporting = sum(r.verdict is ReviewVerdict.SUPPORTS for r in checked)
    contradicting = sum(r.verdict is ReviewVerdict.CONTRADICTS for r in checked)
    abstaining = sum(r.verdict is ReviewVerdict.ABSTAINS for r in checked)
    independent = len(groups)

    if independent == 0:
        status = VerificationStatus.PROPOSED
        rationale = "No independent reviews have been submitted."
    elif independent < minimum_independent_reviewers:
        status = VerificationStatus.UNDER_REVIEW
        rationale = "The minimum independent-review threshold has not been reached."
    elif supporting >= support_quorum and contradicting < reject_quorum:
        status = VerificationStatus.VERIFIED
        rationale = "Independent supporting reviews reached quorum without a rejecting quorum."
    elif contradicting >= reject_quorum and supporting < support_quorum:
        status = VerificationStatus.REJECTED
        rationale = "Independent contradicting reviews reached quorum without a supporting quorum."
    else:
        status = VerificationStatus.INCONCLUSIVE
        rationale = "The evidence is mixed, insufficient, or dominated by abstentions."

    return VerificationDecision(
        finding.finding_id, status, supporting, contradicting, abstaining,
        independent, checked, rationale,
    )
