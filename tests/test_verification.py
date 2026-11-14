import pytest

from gds_core.verification import (
    Finding,
    ReviewVerdict,
    ValidatorReview,
    VerificationStatus,
    adjudicate,
)


def finding():
    return Finding("F-1", "S1", "A candidate claim", ("E-1",), "independent test")


def review(validator, group, verdict):
    return ValidatorReview(
        "F-1", validator, group, verdict, "Review rationale", ("E-2",)
    )


def test_no_reviews_remains_proposed():
    decision = adjudicate(finding(), [])
    assert decision.status is VerificationStatus.PROPOSED


def test_insufficient_independent_reviews_remains_under_review():
    decision = adjudicate(
        finding(), [review("V1", "Lab-A", ReviewVerdict.SUPPORTS)]
    )
    assert decision.status is VerificationStatus.UNDER_REVIEW


def test_independent_support_quorum_verifies_finding():
    decision = adjudicate(
        finding(),
        [
            review("V1", "Lab-A", ReviewVerdict.SUPPORTS),
            review("V2", "Lab-B", ReviewVerdict.SUPPORTS),
        ],
    )
    assert decision.status is VerificationStatus.VERIFIED
    assert decision.supporting_reviews == 2
    assert decision.independent_review_groups == 2


def test_independent_rejection_quorum_rejects_finding():
    decision = adjudicate(
        finding(),
        [
            review("V1", "Lab-A", ReviewVerdict.CONTRADICTS),
            review("V2", "Lab-B", ReviewVerdict.CONTRADICTS),
        ],
    )
    assert decision.status is VerificationStatus.REJECTED


def test_conflicting_quorums_are_inconclusive():
    decision = adjudicate(
        finding(),
        [
            review("V1", "Lab-A", ReviewVerdict.SUPPORTS),
            review("V2", "Lab-B", ReviewVerdict.CONTRADICTS),
            review("V3", "Lab-C", ReviewVerdict.ABSTAINS),
        ],
        minimum_independent_reviewers=3,
    )
    assert decision.status is VerificationStatus.INCONCLUSIVE


def test_origin_cannot_review_own_finding():
    with pytest.raises(ValueError):
        adjudicate(finding(), [review("S1", "Lab-A", ReviewVerdict.SUPPORTS)])


def test_same_validator_or_independence_group_cannot_vote_twice():
    with pytest.raises(ValueError):
        adjudicate(
            finding(),
            [
                review("V1", "Lab-A", ReviewVerdict.SUPPORTS),
                review("V1", "Lab-B", ReviewVerdict.SUPPORTS),
            ],
        )
    with pytest.raises(ValueError):
        adjudicate(
            finding(),
            [
                review("V1", "Lab-A", ReviewVerdict.SUPPORTS),
                review("V2", "Lab-A", ReviewVerdict.SUPPORTS),
            ],
        )


def test_wrong_finding_reference_and_bad_quorum_are_rejected():
    wrong = ValidatorReview("F-2", "V1", "Lab-A", ReviewVerdict.SUPPORTS, "No match")
    with pytest.raises(ValueError):
        adjudicate(finding(), [wrong])
    with pytest.raises(ValueError):
        adjudicate(finding(), [], minimum_independent_reviewers=2, support_quorum=3)


def test_finding_requires_evidence_reference():
    with pytest.raises(ValueError):
        Finding("F-2", "T", "Claim", (), "method")
