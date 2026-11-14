import pytest

from experiments.six_density_verification import verify_six_densities


@pytest.fixture(scope="module")
def audit():
    return verify_six_densities()


def test_six_densities_normalise(audit):
    assert audit["all_integrals_within_1e-8_of_one"] is True
    assert len(audit["results"]) == 6
    for result in audit["results"].values():
        assert result["integral_numeric"] == pytest.approx(1.0, abs=1e-8)


def test_all_six_peak_values_are_strictly_below_one(audit):
    assert audit["all_peaks_strictly_below_one"] is True
    assert audit["cup_peak_is_strictly_below_one"] is True
    for result in audit["results"].values():
        assert result["peak_density"] < 1.0


def test_mahaloy_sinc_convention_and_normaliser(audit):
    assert audit["sinc_convention"] == "sinc(u) = sin(u)/u (unnormalised)"
    assert audit["results"]["Mahaloy"]["normalising_constant"] == pytest.approx(
        1.02985102642849, abs=1e-11
    )


def test_cup_bound_is_strict(audit):
    cup = audit["results"]["Cup"]
    assert audit["cup_normaliser_minus_e"] > 0.20
    assert cup["peak_density"] == pytest.approx(0.92923070582775, abs=1e-11)


def test_all_distributions_are_symmetric_with_finite_moments(audit):
    for result in audit["results"].values():
        assert result["mean"] == 0.0
        assert result["variance"] > 0.0
        assert result["fourth_moment"] > 0.0
        assert result["kurtosis"] > 0.0


def test_reference_normalisers_and_variances(audit):
    expected = {
        "Ghosh": (2.413671104813, 0.264917290354),
        "Mahaloy": (1.029851026428, 0.134795111147),
        "Madras": (1.009709188227, 0.433146815363),
        "Temple bell": (1.812804954111, 0.337989120034),
        "Rocket cone": (1.493648265625, 0.253704101804),
        "Cup": (2.925303491814, 0.429230705828),
    }
    for name, (normaliser, variance) in expected.items():
        result = audit["results"][name]
        assert result["normalising_constant"] == pytest.approx(
            normaliser, abs=1e-10
        )
        assert result["variance"] == pytest.approx(variance, abs=1e-10)
