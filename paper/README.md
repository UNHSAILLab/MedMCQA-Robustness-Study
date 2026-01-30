# Paper: When Chain-of-Thought Backfires

This directory contains the LaTeX source for our paper evaluating prompt sensitivity in MedGemma.

## Structure

```
paper/
├── main.tex              # Main paper source
├── references.bib        # Bibliography
├── generate_figures.py   # Script to generate figures from results
├── figures/              # Generated figures (PDF and PNG)
└── README.md             # This file
```

## Building the Paper

### Prerequisites

- LaTeX distribution (TeX Live recommended)
- Python 3.8+ with matplotlib, seaborn, numpy

### Generate Figures

First, ensure experiments have been run:

```bash
# From project root
python scripts/run_parallel.py --gpu-ids 1,2,3,4,5,6,7
```

Then generate figures:

```bash
cd paper
python generate_figures.py
```

### Compile PDF

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:

```bash
latexmk -pdf main.tex
```

## Status

- [x] Abstract drafted
- [x] Introduction drafted
- [x] Methods section complete
- [x] Figure generation script
- [ ] Full experimental results (50-sample pilot complete)
- [ ] Results section with full data
- [ ] Discussion updates
- [ ] Final proofreading

## Key Findings (Pilot, n=50)

| Finding | Result |
|---------|--------|
| CoT Gain | -8% (hurts performance) |
| Few-shot Gain | -12% (hurts performance) |
| Position Bias | Model predicts 'A' 52% vs 42% actual |
| Context Gain | +10% (full vs question-only) |
| Truncation Effect | 25% truncation worse than no context |
