# The Theory of Gainful Distillation

## The Science of Learning and Intelligence

This repository develops the **Gainful Distillation System (GDS)**: a reciprocal teacher–student learning architecture designed to increase verified collective capability rather than merely reproduce a teacher's outputs.

The initial architecture contains one teacher and five students. The students form a complete graph, K5, augmented by a distinct circumferential overlay. The teacher is connected to every student. The fixed structural topology is kept separate from directed, adaptive knowledge-transmission strengths.

## Central components

- **Adaptive knowledge transmission:** distinct teacher-to-student, student-to-teacher, and peer-to-peer channels.
- **Student Learning Dynamics Estimator (SLDE):** maintains student–domain–method-specific online estimates of learning response, with optional descriptive estimates for retention, transfer, Breadth, and Depth.
- **Specialist task allocation:** intended to route tasks to individual students or collaborative coalitions.
- **Independent learning:** protects each student's capacity to investigate beyond teacher and peer information.
- **Resource allocation:** intended to allocate compute according to expected marginal verified gain, covariance, cost, and risk.
- **Stability and diversity controls:** intended to constrain error propagation and reduce epistemic collapse.
- **External validation and governance:** intended to distinguish transmission, investigation, verification, and integration.

## Principal research conjecture

The Gainful Distillation Conjecture proposes that, under suitable conditions, an adaptive and independently validated teacher–student network can achieve greater net welfare than its original teacher under an equivalent resource budget.

This is a falsifiable research conjecture, not an established empirical result. The repository distinguishes formal definitions and invariants, proposed hypotheses, and experimentally validated findings.

## Repository structure

| Path | Description |
|---|---|
| `1. The Theory of Gainful Distillation.pdf` | Existing treatise (PDF manuscript) |
| `src/gds_core/graph.py` | Fixed six-agent structural topology |
| `src/gds_core/transmission.py` | Directed adaptive transmission-channel representation |
| `src/gds_core/slde.py` | Online Student Learning Dynamics Estimator |
| `tests/test_graph.py` | Structural topology tests |
| `tests/test_transmission.py` | Directed transmission tests |
| `tests/test_slde.py` | SLDE estimation and validation tests |
| `pyproject.toml` | Python package and test configuration |
| `.github/workflows/python-tests.yml` | Python test workflow |
| `LICENSE` | Repository license |

The PDF is retained as the manuscript artifact. This repository does not contain a LaTeX source file or a LaTeX build workflow.

## GDS-Core: structural topology

The node order is (T, S1, S2, S3, S4, S5). The student subgraph contains all ten edges of K5. The additional circumferential overlay adds one further unit of multiplicity to each of the five perimeter pairs.

Consequently:

- There are six vertices and fifteen distinct undirected vertex pairs.
- The five teacher–student pairs each have multiplicity one.
- Five student perimeter pairs have multiplicity two; five student diagonals have multiplicity one.
- Total undirected edge multiplicity is twenty.
- The teacher's weighted degree is five; each student's weighted degree is seven.

The structural adjacency matrix is symmetric and fixed. The transmission matrix is directed and dynamic. Its convention is `W[source][target]`; therefore teacher-to-student and student-to-teacher strengths are independently parameterised. Peer transmission is the structural multiplicity multiplied by a directed peer-channel multiplier.

## Student Learning Dynamics Estimator

The SLDE maintains a separate profile for each student–domain–method tuple. Its current learning-response estimator uses scalar recursive least squares (RLS) to model performance change as a function of positive instructional dose:

`performance\_after - performance\_before = learning\_rate × instructional\_dose + residual.`

The forgetting factor is configurable; a value of one gives ordinary RLS, while a value below one discounts older observations. Retention, transfer, Breadth, and Depth are tracked as optional bounded observations using online descriptive moments. Missing metric values are left missing rather than imputed as zero.

These estimates are implementation-level statistical summaries, not validated measures of general intelligence or universal learning ability. In particular, the scalar response model and its uncertainty estimates require further specification, simulation, calibration, and empirical evaluation before being used for consequential task or resource allocation.

## Development and testing

Python 3.10 or later is required. Install the development dependency and run the test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The Python workflow runs the test suite on pushes and pull requests that change source code, tests, or project configuration.

## Research discipline

1. Evaluate all architectures under comparable compute and data budgets.
2. Measure generalisation on held-out tasks and distributions.
3. Treat student learning profiles as estimates with uncertainty, not fixed labels.
4. Verify novelty independently; disagreement alone is not innovation.
5. Report capability, cost, energy, risk, coordination overhead, and distributional effects.
6. Preserve audit trails for data, models, interventions, validation, and updates.
7. Report negative and null results as well as positive findings.

## Implementation status

**Implemented:** GDS-Core structural graph, directed transmission-channel representation, online SLDE with scalar RLS and optional descriptive metric tracking, and unit tests for these components.

**Next:** specialist task routing, resource optimisation, external validation, stability and diversity controls, and reproducible end-to-end evaluation.

The treatise provides a theoretical specification and empirical research programme. Claims of performance superiority remain subject to implementation, replication, and independent validation.
