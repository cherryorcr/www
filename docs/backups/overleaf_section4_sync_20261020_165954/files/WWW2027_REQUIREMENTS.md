# WWW 2027 Research Track checklist

Source: https://www2027.thewebconf.org/research-track-papers/ (checked 2026-10-16).

- English, ACM double-column format.
- Double-blind review; omit author names, affiliations, acknowledgements, and identifying self-citations.
- Use `\documentclass[sigconf,anonymous,review]{acmart}` for the submission.
- Long paper: at most 8 pages of main content.
- References and optional appendix are additional, but the complete PDF is at most 12 pages.
- The first 8 pages must be self-contained because reviewers are not required to read beyond them.
- Submit one PDF file.

The local draft follows the ACM class. MiKTeX is installed for the current
Windows user. From the paper_www directory, run:

```powershell
./tools/compile.ps1
```

The script first uses pdflatex (three passes for references), then Tectonic,
then latexmk if the earlier compilers are unavailable. It detects the
user-installed MiKTeX even when it is not on PATH. The PDF is build/main.pdf;
build products remain in build and are not source files for Overleaf.

The draft keeps the ACM font sizes and margins. An unsupported `balance=false`
option was removed after the installed class reported it as unused.

## Verified draft

- Verified on 2026-10-17 with the installed MiKTeX/pdflatex compiler.
- Main text, including the conclusion, ends on page 7; the full PDF has 11 pages.
- References occupy pages 7--8; supplementary evidence occupies pages 8--11.
- The current verified PDF is `build/graphclip_revision/main.pdf`, compiled
  from the existing `main.tex` while `build/main.pdf` was locked by another app.
- No overfull boxes, unresolved references, or unresolved citations in the
  final local compile. CCS descriptions are included. The ACM class still
  warns about the disabled ACM reference block; submission metadata should be checked
  against the venue template before the final upload.
- The three main figures are vector PDFs: the problem/interface comparison,
  the construction pipeline, and paired effects on the primary/additional
  masks. Their scripts and PNG previews are included in the source repository.
  The appendix retains the nine-panel no-GM loss-trajectory raster figure.
- Tables use a consistent booktabs format. Primary replication, broader
  single-construction results, core ablations, and full-history retention are
  separate tables, with detailed controls in the appendix.
- The built-in standalone preview compiler cannot resolve this project's
  external `sections/` inputs in its temporary directory. Use the local
  project compiler above to check the complete manuscript; this limitation
  is separate from the successful local PDF build.
