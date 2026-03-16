# Repository Guidelines

## Project Structure & Module Organization
`anomalylab/` contains the published package. Core analysis code is split by responsibility: `core/` exposes the main panel API, `structure/` defines shared data containers, `preprocess/` handles cleaning and transforms, `empirical/` implements asset-pricing methods, `visualization/` formats Excel output, and `datasets/` ships sample CSV data. Keep new modules close to the feature area they extend. Tests belong in `tests/`. Release automation lives in `.github/workflows/python-publish.yml`.

## Build, Test, and Development Commands
Use Python 3.10+.

- `python -m pip install -r requirements.txt` installs runtime dependencies.
- `python -m pip install -e .` installs the package in editable mode for local development.
- `python -m build` creates source and wheel distributions, matching the release workflow.
- `python setup.py sdist bdist_wheel` is a workable fallback if `build` is unavailable.
- `python -m pytest` should be used for local verification once tests are added.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module-level imports grouped by standard library, third-party, then local packages, and type hints on public APIs. Use `snake_case` for functions, methods, and variables; `PascalCase` for classes; and short, explicit parameter names such as `time`, `id`, and `weight`. Keep docstrings concise and focused on behavior and arguments. Preserve backward compatibility for public API names unless a versioned change is intentional.

## Testing Guidelines
The repository currently only includes a placeholder `tests/__init__.py`, so new contributions should add automated coverage with `pytest`. Place tests under `tests/` and name files `test_<module>.py`; name test functions `test_<behavior>()`. Prefer small fixture-based DataFrame samples over large external files. Cover both expected outputs and failure cases for statistical routines, especially around missing data, grouping, and regression inputs.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit-style prefixes such as `feat(portfolio): ...`, `fix: ...`, `refactor(portfolio): ...`, and `chore(setup): ...`. Keep commit subjects imperative and scoped when useful. Pull requests should summarize the analytical or API impact, list the verification performed, and note any dataset or packaging changes. Include before/after examples when modifying tabular output or Excel formatting behavior.

## Release Notes & Packaging
Package metadata is defined in `setup.py`, and sample datasets are included through `package_data`. If you add files under `anomalylab/datasets/`, confirm they are packaged correctly and rebuild with `python -m build` before tagging a release.
