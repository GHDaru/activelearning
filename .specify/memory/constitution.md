<!--
Sync Impact Report:
- Version change: (new) -> 1.0.0
- Initial ratification for the activelearning library.
-->

# activelearning Constitution

Biblioteca de Aprendizado Ativo para classificação de texto — motor experimental da tese
de doutorado (FALCO) e núcleo de domínio reutilizado pelo FlowBuilder (backend/frontend).

## Core Principles

### I. Science First — Reproducibility is the Product (NON-NEGOTIABLE)
Every experiment that produces a number cited in the thesis MUST be reproducible from a
single entrypoint (`experiments/<id>/run_<id>.py` + config file). Every run MUST persist:
config snapshot, git SHA, random seeds, per-iteration metrics, and raw artifacts
(CSV/JSONL). **No number enters the thesis (`tesedaru`) without a traceable artifact in
this repository.**

### II. DDD + Hexagonal Architecture (NON-NEGOTIABLE)
The domain core (`src/activelearning/domain/`) MUST be pure Python: no imports of
sklearn, torch, transformers, openai, pandas, or any I/O. External capabilities enter
only through **ports** (`src/activelearning/ports/`) implemented by **adapters**
(`src/activelearning/adapters/`). Use cases live in `application/`. The Ubiquitous
Language is defined in [.specify/memory/ubiquitous-language.md](ubiquitous-language.md)
and MUST be kept synchronized with code and specs.

### III. Constrained Oracle Output (NON-NEGOTIABLE)
Every LLM oracle adapter MUST constrain the predicted label to the closed
`CategorySchema` of the task — via JSON Schema `enum` (structured output) when the
provider supports it, and via post-hoc exact-match validation with explicit
`invalid_label` accounting when it does not. Free-text label output is PROHIBITED: the
measured accuracy of an unconstrained oracle is not a valid measurement (lesson learned
from the legacy `activetextclassification`, where missing `enum` deflated oracle accuracy
by counting phrasing variants as errors).

### IV. Cost & Noise Observability
Every oracle call MUST record: model id, temperature, prompt version, input/output
tokens, latency, monetary cost estimate, and (in simulation) correctness vs. gold label.
Aggregated cost per 1k labels is a first-class experiment output — it feeds the thesis
cost analysis.

### V. Legacy is Read-Only
`activetextclassification` and the legacy `FlowBuilder` are frozen references. Code is
**ported** (rewritten to fit this architecture, with tests), never copied wholesale.
Legacy result artifacts (L0 sensitivity, GA optimization, DRI-SL logs) remain valid
evidence and MAY be re-analyzed here, but new executions happen only in this library.

### VI. Spec-Driven Development
Every feature starts as `specs/[###-name]/spec.md` (spec-kit). Plans and tasks derive
from the spec. A feature is done when its spec's acceptance criteria pass and the README
of the affected module is updated (final Polish task).

## Technical Standards

- **Language**: Python 3.12+.
- **Dependency Management**: `uv` with `pyproject.toml`. `pip`/`requirements.txt` are
  PROHIBITED. (✅ MANDATORY)
- **Testing**: `pytest`; TDD for domain code (write the failing test first). Domain
  tests MUST NOT require network, GPU, or API keys. (✅ MANDATORY)
- **Layout**: `src/` layout; package `activelearning`.
- **Paths**: `pathlib` only.
- **LLM Providers**: OpenAI, Google Gemini (`google-genai`), Ollama (local), Anthropic —
  all as interchangeable adapters of `OraclePort`.
- **Classifiers**: BERTimbau (HuggingFace `transformers`) and lightweight PVBin-style
  vectorizer classifiers — all as adapters of `ClassifierPort`.
- **Data**: experiment artifacts as CSV/JSONL under `experiments/<id>/results/`
  (git-ignored above a size threshold; summary tables are committed).
- **Config**: experiment configs are JSON, versioned in git next to the runner.
- **Lint/Format**: `ruff` (line length 100).

## Governance

- **Amendments**: version bump + note in the Sync Impact Report header.
- **Compliance**: every PR/feature plan MUST verify compliance with Principles I–VI.
- **Versioning**: Semantic Versioning for this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-07-16 | **Last Amended**: 2026-07-16
