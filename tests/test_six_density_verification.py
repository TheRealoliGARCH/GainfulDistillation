import pytest

from experiments.six_density_verification import verify_six_densities


@pytest.fixture(scope="module")
def audit():
    return verify_six_densities()


def test_all_six_normalisers_match_independent_scipy_quadrature(audit):
    assert audit["all_normalised_integrals_within_1e-10_of_one"] is True
    for result in audit["results"].values():
        assert result["normalised_density_integral"] == pytest.approx(1.0, abs=1e-10)
        assert result["raw_kernel_integral_quad_error"] < 1e-9


def test_all_peak_values_are_strictly_below_one(audit):
    assert audit["all_peak_values_strictly_below_one"] is True
    assert audit["cup_peak_strictly_below_one"] is True
    for result in audit["results"].values():
        assert result["peak_density"] < 1.0


def test_mahaloy_printed_normaliser_identifies_sinc_convention(audit):
    assert audit["sinc_convention"] == "unnormalised sinc(u)=sin(u)/u, sinc(0)=1"
    assert abs(audit["mahaloy_normaliser_minus_direct_integral"]) < 1e-10
    assert audit["results"]["Mahaloy"]["normalising_constant"] == pytest.approx(
        1.02985102642849, abs=1e-11
    )


def test_cup_upper_bound_can_be_strengthened_to_strict(audit):
    cup = audit["results"]["Cup"]
    assert audit["cup_normaliser_minus_e"] > 0.20
    assert cup["peak_density"] == pytest.approx(0.92923070582775, abs=1e-11)


def test_all_six_are_symmetric_with_finite_positive_moments(audit):
    for result in audit["results"].values():
        assert result["mean"] == 0.0
        assert result["variance"] > 0.0
        assert result["fourth_moment"] > 0.0
        assert result["kurtosis"] > 0.0


def test_reference_normalisers_variances_and_kurtoses(audit):
    expected = {
        "Ghosh": (2.413671104813002, 0.264917290353926, 2.24104596880736),
        "Mahaloy": (1.029851026428492, 0.134795111147494, 2.91469523146811),
        "Madras": (1.009709188227373, 0.433146815362601, 1.27525358792227),
        "Temple bell": (1.812804954110954, 0.337989120033642, 2.18843961522648),
        "Rocket cone": (1.493648265624854, 0.253704101803684, 2.08589521815590),
        "Cup": (2.925303491814363, 0.429230705827751, 1.54899152136945),
    }
    for name, (normaliser, variance, kurtosis) in expected.items():
        result = audit["results"][name]
        assert result["normalising_constant"] == pytest.approx(normaliser, abs=1e-11)
        assert result["variance"] == pytest.approx(variance, abs=1e-11)
        assert result["kurtosis"] == pytest.approx(kurtosis, abs=1e-10)
