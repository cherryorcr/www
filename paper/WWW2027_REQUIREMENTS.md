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

The draft disables automatic end-of-document column balancing with the
official `balance=false` class option. This avoids a spurious final page
when appendix floats remain pending; the ACM font sizes and margins are
unchanged.

## Verified draft

- Verified on 2026-10-26 with the installed MiKTeX/pdflatex compiler.
- Main text, including the conclusion, ends on page 8; the full submission PDF
  has 12 pages, with references and a selected appendix after the main text.
  The full PDF remains within the 12-page limit.
- The first 8 pages are self-contained. The appendix source files remain in the
  repository for reproducibility. The submission PDF includes a compact
  selection of protocol, archived comparison, resource, and convergence
  evidence; the longer exploratory appendix remains in the repository.
- No overfull boxes, unresolved references, or unresolved citations were found
  in the final local compile. The ACM class still warns about the disabled ACM
  reference block; submission metadata should be checked against the venue
  template before the final upload.
- The three active figures are vector PDFs: the construction diagram, paired
  baseline effects with full-history retention, and core ablation effects.
  Their scripts and PNG previews are included in the source repository.
- The built-in standalone preview compiler cannot resolve this project's
  external `sections/` inputs in its temporary directory. Use the local
  project compiler above to check the complete manuscript; this limitation
  is separate from the successful local PDF build.
