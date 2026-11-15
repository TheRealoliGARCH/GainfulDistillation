import pytest

from experiments.mixture_coverage import run_mixture_coverage


@pytest.fixture(scope="module")
def coverage():
    return run_mixture_coverage(scan_intervals=8192)


def test_convex_mixture_coverage_reference_value(coverage):
    assert coverage["coverage_percentage"] == pytest.approx(73.2565995308, abs=2e-8)
    assert coverage["covered_area"] == pytest.approx(1.465131990616, abs=2e-10)
    assert coverage["uncovered_percentage"] == pytest.approx(26.7434004692, abs=2e-8)


def test_coverage_uses_correct_rectangle_area_and_symmetry(coverage):
    assert coverage["rectangle_area"] == 2.0
    assert coverage["covered_area"] + coverage["uncovered_area"] == pytest.approx(2.0, abs=1e-12)
    assert coverage["coverage_fraction"] == pytest.approx(
        coverage["positive_half_envelope_span_integral"], abs=1e-12
    )


def test_pointwise_envelope_has_eight_segments(coverage):
    segments = coverage["envelope_segments_on_positive_half"]
    assert len(segments) == 8
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 1.0
    for first, second in zip(segments[:-1], segments[1:]):
        assert first["end"] == pytest.approx(second["start"], abs=1e-12)
    assert all(s["span_integral_on_positive_half"] >= 0.0 for s in segments)


def test_envelope_switches_are_stable_at_higher_scan_resolution(coverage):
    higher = run_mixture_coverage(scan_intervals=16384)
    assert higher["coverage_percentage"] == pytest.approx(coverage["coverage_percentage"], abs=2e-9)
    low = coverage["envelope_segments_on_positive_half"]
    high = higher["envelope_segments_on_positive_half"]
    assert len(low) == len(high) == 8
    for a, b in zip(low, high):
        assert a["upper"] == b["upper"]
        assert a["lower"] == b["lower"]
        assert a["start"] == pytest.approx(b["start"], abs=2e-10)
        assert a["end"] == pytest.approx(b["end"], abs=2e-10)


def test_invalid_root_scan_settings_are_rejected():
    with pytest.raises(ValueError):
        run_mixture_coverage(scan_intervals=100)
    with pytest.raises(ValueError):
        run_mixture_coverage(scan_intervals=True)
