<#
.SYNOPSIS
    Windows replacement for the Makefile. Every project command, one script.

.DESCRIPTION
    There is no 'make' on Windows and PowerShell does not understand 'export'
    or bash for-loops, so the commands in the README do not work as written.
    This script wraps all of them.

    By default every command runs with the OFFLINE providers (a stub LLM and a
    hashing embedder). That means no Ollama, no API key and no network are
    needed — the system runs end to end so you can see it work. Add -Real to
    use the provider configured in config/settings.yaml instead.

.EXAMPLE
    .\run.ps1 setup      # create .venv and install dependencies (do this first)
    .\run.ps1 test       # 86 offline tests
    .\run.ps1 ingest     # load policy versions 1, 2 and 3 into ChromaDB
    .\run.ps1 eval       # run the 10-question golden dataset
    .\run.ps1 api        # FastAPI on :8000   (leave running)
    .\run.ps1 ui         # Streamlit on :8501 (leave running, separate window)
    .\run.ps1 api -Real  # same, but using Ollama / your configured provider
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'check', 'test', 'ingest', 'eval', 'api', 'ui',
                 'watcher', 'versions', 'reset', 'help')]
    [string]$Command = 'help',

    # Use the real provider from config/settings.yaml instead of offline stubs.
    [switch]$Real
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# ---------------------------------------------------------------------------
# Find the right Python. A project virtualenv wins over whatever is on PATH,
# because the venv is the only one guaranteed to have this project's pinned
# dependency versions.
# ---------------------------------------------------------------------------
function Get-ProjectPython {
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return 'python'
}

function Show-Banner {
    param([string]$Message)
    Write-Host ''
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host ('  ' + ('-' * $Message.Length)) -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Provider selection. Setting these environment variables is what 'export ...'
# does on Linux; in PowerShell the equivalent is the $env: drive, and it lasts
# only for this process.
# ---------------------------------------------------------------------------
function Set-Providers {
    if ($Real) {
        Remove-Item Env:HLR__LLM__PROVIDER        -ErrorAction SilentlyContinue
        Remove-Item Env:HLR__EMBEDDINGS__PROVIDER -ErrorAction SilentlyContinue
        Write-Host '  providers: from config/settings.yaml (real)' -ForegroundColor Yellow
    }
    else {
        $env:HLR__LLM__PROVIDER        = 'deterministic'
        $env:HLR__EMBEDDINGS__PROVIDER = 'deterministic'
        Write-Host '  providers: offline stubs (no Ollama or key needed)' -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------------------------
# OneDrive holds a lock on files while it syncs them. ChromaDB keeps its index
# in a SQLite file that is written continuously, and the two combine into
# "database is locked" errors and occasionally a corrupt index. Warn once
# rather than let it be debugged from scratch.
# ---------------------------------------------------------------------------
function Test-OneDriveLocation {
    if ($PSScriptRoot -match 'OneDrive') {
        Write-Host ''
        Write-Host '  WARNING: this project is inside OneDrive.' -ForegroundColor Yellow
        Write-Host '  ChromaDB writes a SQLite file continuously and OneDrive syncing it' -ForegroundColor Yellow
        Write-Host '  mid-write causes "database is locked" errors.' -ForegroundColor Yellow
        Write-Host '  If that happens, move this folder outside OneDrive.' -ForegroundColor Yellow
    }
}

$python = Get-ProjectPython

switch ($Command) {

    'setup' {
        Show-Banner 'Creating virtual environment and installing dependencies'
        Test-OneDriveLocation

        if (-not (Test-Path '.venv')) {
            Write-Host '  creating .venv ...'
            python -m venv .venv
        }
        else {
            Write-Host '  .venv already exists, reusing it'
        }

        $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
        & $venvPython -m pip install --upgrade pip --quiet
        & $venvPython -m pip install -r requirements.txt

        if (-not (Test-Path '.env')) {
            Copy-Item '.env.example' '.env'
            Write-Host '  created .env from .env.example'
        }

        Write-Host ''
        Write-Host '  Done. Next:  .\run.ps1 test' -ForegroundColor Green
    }

    'check' {
        Show-Banner 'Environment check'
        Set-Providers
        & $python scripts/check_environment.py
    }

    'test' {
        Show-Banner 'Test suite (fully offline)'
        & $python -m pytest
    }

    'ingest' {
        Show-Banner 'Ingesting policy versions 1, 2 and 3'
        Set-Providers
        Test-OneDriveLocation

        # Each version must be promoted and ingested in order. The golden
        # dataset grades against version 3 being active while versions 1 and 2
        # remain queryable, so running only version 3 is not equivalent.
        foreach ($version in 1, 2, 3) {
            Write-Host ''
            Write-Host "  --- version $version ---" -ForegroundColor DarkGray
            & $python scripts/simulate_policy_update.py --version $version
            if ($LASTEXITCODE -ne 0) { throw "promoting version $version failed" }
            & $python scripts/ingest.py
            if ($LASTEXITCODE -ne 0) { throw "ingesting version $version failed" }
        }

        Write-Host ''
        Write-Host '  Done. Next:  .\run.ps1 eval' -ForegroundColor Green
    }

    'eval' {
        Show-Banner 'Golden dataset — 10 questions, 3 repeats each'
        Set-Providers
        & $python evaluation/run_evaluation.py
    }

    'api' {
        Show-Banner 'FastAPI backend on http://localhost:8000'
        Set-Providers
        Write-Host '  docs at http://localhost:8000/docs — Ctrl+C to stop'
        & $python run_backend.py
    }

    'ui' {
        Show-Banner 'Streamlit frontend on http://localhost:8501'
        Set-Providers
        Write-Host '  the backend must already be running in another window'
        & $python -m streamlit run frontend/streamlit_app.py
    }

    'watcher' {
        Show-Banner 'Document update watcher'
        Set-Providers
        & $python scripts/run_watcher.py
    }

    'versions' {
        Show-Banner 'Policy version history'
        & $python scripts/simulate_policy_update.py --status
    }

    'reset' {
        Show-Banner 'Deleting the index and version history'
        & $python scripts/reset_data.py --yes
        Write-Host '  Code and config are untouched. Re-run:  .\run.ps1 ingest'
    }

    default {
        Write-Host ''
        Write-Host '  Home Loan RAG — Windows commands' -ForegroundColor Cyan
        Write-Host '  --------------------------------' -ForegroundColor DarkGray
        Write-Host ''
        Write-Host '  First time, in order:'
        Write-Host '    .\run.ps1 setup      create .venv, install dependencies, make .env'
        Write-Host '    .\run.ps1 test       86 offline tests — proves the install works'
        Write-Host '    .\run.ps1 ingest     load policy versions 1, 2 and 3 into ChromaDB'
        Write-Host '    .\run.ps1 eval       run the 10-question golden dataset'
        Write-Host ''
        Write-Host '  Running the app (each needs its own window):'
        Write-Host '    .\run.ps1 api        FastAPI on :8000'
        Write-Host '    .\run.ps1 ui         Streamlit on :8501'
        Write-Host '    .\run.ps1 watcher    live document update process'
        Write-Host ''
        Write-Host '  Other:'
        Write-Host '    .\run.ps1 check      diagnose the environment'
        Write-Host '    .\run.ps1 versions   show policy version history'
        Write-Host '    .\run.ps1 reset      delete the index and start over'
        Write-Host ''
        Write-Host '  Add -Real to any command to use Ollama or your configured'
        Write-Host '  provider instead of the offline stubs:'
        Write-Host '    .\run.ps1 api -Real'
        Write-Host ''
    }
}
