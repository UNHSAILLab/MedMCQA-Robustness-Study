# Paper: When Chain-of-Thought Backfires

LaTeX source and camera-ready PDF for our paper evaluating prompt sensitivity in MedGemma.

**Camera-ready PDF:** [`2AI_CRC_183.pdf`](2AI_CRC_183.pdf)
**Authors:** Binesh Sadanandan, Vahid Behzadan (SAIL Lab, University of New Haven)

## Files

```
paper/
├── 2AI_CRC_183.pdf       # Camera-ready submission (final)
├── main.tex              # LaTeX source
├── main.pdf              # Latest local build (same as camera-ready)
├── references.bib        # Bibliography
├── generate_figures.py   # Generates figures from experiment results
├── figures/              # Generated figures (PDF and PNG)
├── springer/             # Springer LNCS template files
└── README.md             # This file
```

## Building from Source

### Prerequisites

- LaTeX distribution (TeX Live 2023 or later recommended)
- Python 3.8+ with matplotlib, seaborn, numpy (for figure generation)

### Generate Figures

If you want to regenerate figures from raw experiment outputs:

```bash
# From project root, run experiments first
python scripts/run_parallel.py --gpu-ids 1,2,3,4,5,6,7

# Then generate figures
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

The output `main.pdf` should match `2AI_CRC_183.pdf`.

## Bibliography

All 22 citations were verified against [scite.ai](https://scite.ai/) metadata:
- Author names match published versions
- Titles match published versions
- DOIs included where available

## Code Availability

A footnote on the title page links to this repository: <https://github.com/UNHSAILLab/MedMCQA-Robustness-Study>. Appendix H also lists it under Reproducibility.
