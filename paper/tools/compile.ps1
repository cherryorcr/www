param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$paperDir = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $paperDir 'build'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

if ($Clean -and (Test-Path $buildDir)) {
    Get-ChildItem -LiteralPath $buildDir -Force | Remove-Item -Recurse -Force
}

Push-Location $paperDir
try {
    # Prefer the user-installed MiKTeX when it is not exposed in PATH.
    $miktexBin = Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64'
    $hasMiKTeX = Test-Path (Join-Path $miktexBin 'pdflatex.exe')
    if ($hasMiKTeX) {
        $env:Path = "$miktexBin;$env:Path"
    }

    $compiler = $null
    if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
        $compiler = 'pdflatex'
        $latexArgs = @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-synctex=1')
        if ($hasMiKTeX) { $latexArgs += '--enable-installer' }
        $latexArgs += @("-output-directory=$buildDir", 'main.tex')
        & pdflatex @latexArgs
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on the bibliography preparation pass with exit code $LASTEXITCODE" }
        if (-not (Get-Command bibtex -ErrorAction SilentlyContinue)) {
            throw 'BibTeX is required for the references.bib bibliography.'
        }
        Push-Location $buildDir
        try {
            # BibTeX reads the auxiliary file from the build directory, while
            # the bibliography database remains next to main.tex.
            $oldBibInputs = $env:BIBINPUTS
            if ([string]::IsNullOrWhiteSpace($oldBibInputs)) {
                $env:BIBINPUTS = "$paperDir;"
            } else {
                $env:BIBINPUTS = "$paperDir;$oldBibInputs"
            }
            & bibtex 'main'
            if ($LASTEXITCODE -ne 0) { throw "bibtex failed with exit code $LASTEXITCODE" }
            $env:BIBINPUTS = $oldBibInputs
        } finally {
            if ($null -ne $oldBibInputs) {
                $env:BIBINPUTS = $oldBibInputs
            } else {
                Remove-Item Env:BIBINPUTS -ErrorAction SilentlyContinue
            }
            Pop-Location
        }
        for ($pass = 2; $pass -le 3; $pass++) {
            & pdflatex @latexArgs
            if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on pass $pass with exit code $LASTEXITCODE" }
        }
    } elseif (Get-Command tectonic -ErrorAction SilentlyContinue) {
        $compiler = 'tectonic'
        & tectonic --keep-logs --synctex --outdir $buildDir main.tex
        if ($LASTEXITCODE -ne 0) { throw "tectonic failed with exit code $LASTEXITCODE" }
    } elseif (Get-Command latexmk -ErrorAction SilentlyContinue) {
        $compiler = 'latexmk'
        & latexmk -pdf -interaction=nonstopmode -halt-on-error "-outdir=$buildDir" main.tex
        if ($LASTEXITCODE -ne 0) { throw "latexmk failed with exit code $LASTEXITCODE" }
    } else {
        throw 'No LaTeX compiler found. Install tectonic, latexmk, or MiKTeX/pdflatex.'
    }

    $pdf = Join-Path $buildDir 'main.pdf'
    if (-not (Test-Path $pdf)) { throw "Compilation with $compiler did not produce $pdf" }

    $pdfInfo = Get-Command pdfinfo -ErrorAction SilentlyContinue
    if ($pdfInfo) {
        & pdfinfo $pdf | Select-String 'Pages|Page size|File size'
    } else {
        Write-Host "Compiled with ${compiler}: $pdf"
    }
} finally {
    Pop-Location
}
