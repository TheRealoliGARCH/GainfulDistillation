# The Theory of Gainful Distillation

## The Science of Learning and Intelligence

This repository develops the **Gainful Distillation System (GDS)**: a reciprocal teacher–student learning architecture designed to increase verified collective capability rather than merely reproduce a teacher's outputs.

The initial architecture contains one teacher and five students. The students form a complete graph, K5, augmented by a distinct circumferential overlay. The teacher is connected to every student. The fixed structural topology is kept separate from directed, adaptive knowledge-transmission strengths.

## Central components

- **Adaptive knowledge transmission:** distinct teacher-to-student, student-to-teacher, and peer-to-peer channels.
- **Student Learning Dynamics Estimator (SLDE):** estimates student-, domain-, and method-specific learning responses, with separate tracking of retention, transfer, Breadth, Depth, and uncertainty.
- **Specialist task allocation:** routes tasks to individual students or collaborative coalitions.
- **Independent learning:** protects each student's capacity to investigate beyond teacher and peer information.
- **Resource allocation:** allocates compute according to expected marginal verified gain, covariance, cost, and risk.
- **Stability and diversity controls:** constrain error propagation and reduce epistemic collapse.
- **External validation and governance:** distinguish transmission, investigation, verification, and integration.

## Principal research conjecture

The Gainful Distillation Conjecture proposes that, under suitable conditions, an adaptive and independently validated teacher–student network can achieve greater net welfare than its original teacher under an equivalent resource budget.

This is a falsifiable research conjecture, not an established empirical result. The repository distinguishes formal definitions and invariants, proposed hypotheses, and experimentally validated findings.

## Repository structure

| Path | Description |
|---|---|
| `1. The Theory of Gainful Distillation.pdf` | Published theoretical treatise |
| `src/gds_core/graph.py` | Fixed six-agent structural topology |
| `src/gds_core/transmission.py` | Directed adaptive transmission channels |
| `src/gds_core/slde.py` | Online Student Learning Dynamics Estimator |
| `tests/` | Unit tests for graph, transmission, and SLDE |
| `pyproject.toml` | Python package and test configuration |
| `.github/workflows/python-tests.yml` | Automated Python test workflow |
| `LICENSE` | Repository license |

## GDS-Core: structural conventions

The node order is (T, S1, S2, S3, S4, S5). The student subgraph contains all ten edges of K5. The additional circumferential overlay adds one further unit of multiplicity to each of the five perimeter pairs.

Consequently:
- There are six vertices and fifteen distinct undirected vertex pairs.
- The five teacher–student pairs each have multiplicity one.
- Five student perimeter pairs have multiplicity two; five student diagonals have multiplicity one.
- Total undirected edge multiplicity is twenty.
- The teacher's weighted degree is five; each student's weighted degree is seven.

The structural adjacency matrix is symmetric and fixed. The transmission matrix is directed and dynamic. Its convention is W[source][target]; teacher-to-student and student-to-teacher strengths are independently parameterised. Peer transmission is the structural multiplicity multiplied by a directed peer-channel multiplier.

## SLDE: estimation conventions

SLDE maintains a separate profile for each student × domain × instructional-method combination. The initial response model is:

`performance_change = learning_rate × instructional_dose + innovation`

The learning-rate coefficient is updated by scalar recursive least squares (RLS). A forgetting factor of one gives ordinary RLS; values below one discount older observations. Retention, transfer, Breadth, and Depth are tracked as separate optional bounded measurements, using online means and sample variances. Missing measurements are not imputed as zero.

Performance and auxiliary metric scores are currently expected to be normalised to [0, 1], and instructional dose must be strictly positive. These are implementation conventions, not claims that every real-world learning measure is naturally bounded or directly comparable. Empirical work must establish valid measurement instruments, score comparability, and the causal interpretation of instructional dose.

## Development and testing

Python 3.10 or later is required:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The Python workflow runs tests on pushes and pull requests that change source code, tests, or project configuration.

## Research discipline

1. Evaluate all architectures under comparable compute and data budgets.
2. Measure generalisation on held-out tasks and distributions.
3. Treat student learning profiles as estimates with uncertainty, not fixed labels.
4. Verify novelty independently; disagreement alone is not innovation.
5. Report capability, cost, energy, risk, coordination overhead, and distributional effects.
6. Preserve audit trails for data, models, interventions, validation, and updates.
7. Report negative and null results as well as positive findings.

## Implementation status

**Implemented:** GDS-Core structural graph, directed transmission-channel representation, and the first SLDE estimator with unit tests.

**Next:** specialist task routing, resource optimisation, external validation, and reproducible evaluation.

The PDF treatise provides the theoretical specification and empirical research programme. Claims of performance superiority remain subject to implementation, replication, and independent validation.
