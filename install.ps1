param(
    [string]$Version = $(if ($env:ZEROGRAVITY_VERSION) { $env:ZEROGRAVITY_VERSION } else { "latest" }),
    [string]$InstallDir = $env:ZEROGRAVITY_INSTALL_DIR,
    [string]$Repository = $env:ZEROGRAVITY_REPO,
    [string]$ReleaseBaseUrl = $env:ZEROGRAVITY_RELEASE_BASE_URL
)

$ErrorActionPreference = "Stop"

if (-not $InstallDir) {
    if (-not $HOME) { throw "HOME is unavailable; provide -InstallDir." }
    $InstallDir = Join-Path $HOME ".local\bin"
}

$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
switch ($Arch.ToString()) {
    "X64"   { $Asset = "zerogravity-windows-x86_64.exe" }
    "Arm64" { $Asset = "zerogravity-windows-aarch64.exe" }
    default { throw "Unsupported Windows architecture: $Arch (supported: x86_64, ARM64)" }
}

if (-not $ReleaseBaseUrl) {
    if (-not $Repository) {
        throw "Specify -Repository OWNER/zerogravity (after publishing) or -ReleaseBaseUrl. No repository is assumed."
    }

    if ($Version -eq "latest") {
        $ReleaseBaseUrl = "https://github.com/$Repository/releases/latest/download"
    } else {
        $Tag = if ($Version.StartsWith("v")) { $Version } else { "v$Version" }
        $ReleaseBaseUrl = "https://github.com/$Repository/releases/download/$Tag"
    }
}

$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) ("zerogravity-install-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $WorkDir | Out-Null

try {
    $File = Join-Path $WorkDir $Asset
    $Sidecar = "$File.sha256"
    Write-Host "Downloading zerogravity bootstrap $Asset..."
    Invoke-WebRequest -UseBasicParsing -Uri "$($ReleaseBaseUrl.TrimEnd('/'))/$Asset" -OutFile $File
    Invoke-WebRequest -UseBasicParsing -Uri "$($ReleaseBaseUrl.TrimEnd('/'))/$Asset.sha256" -OutFile $Sidecar

    $Expected = ((Get-Content -LiteralPath $Sidecar -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
    if ($Expected -notmatch '^[0-9a-f]{64}$') { throw "Invalid SHA-256 sidecar." }
    $Actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Expected -ne $Actual) { throw "SHA-256 mismatch; refusing to install." }

    $Line = (& $File --version | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or $Line -notmatch '^zerogravity v(.+)$') {
        throw "Downloaded binary does not identify itself as zerogravity."
    }
    $ReportedVersion = $Matches[1].Trim()
    if ($Version -ne "latest") {
        $Requested = $Version.TrimStart("v")
        if ($Requested -ne $ReportedVersion) {
            throw "Downloaded binary version differs from requested version."
        }
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $Destination = Join-Path $InstallDir "zerogravity.exe"
    $ExistingMatches = (Test-Path -LiteralPath $Destination) -and
        ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Expected)

    if ($ExistingMatches) {
        Write-Host "zerogravity v$ReportedVersion is already installed at $Destination"
    } else {
        $TempDestination = Join-Path $InstallDir (".zerogravity.exe.tmp." + [guid]::NewGuid().ToString("N"))
        $BackupDestination = Join-Path $InstallDir (".zerogravity.exe.backup." + [guid]::NewGuid().ToString("N"))
        try {
            Copy-Item -LiteralPath $File -Destination $TempDestination
            if (Test-Path -LiteralPath $Destination) {
                [System.IO.File]::Replace($TempDestination, $Destination, $BackupDestination)
            } else {
                [System.IO.File]::Move($TempDestination, $Destination)
            }
        } finally {
            Remove-Item -LiteralPath $TempDestination -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $BackupDestination -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Installed zerogravity v$ReportedVersion at $Destination (bootstrap binary only)."
    }

    if (-not (($env:PATH -split ';') -contains $InstallDir)) {
        Write-Host "Add $InstallDir to PATH to run zerogravity."
    }
} finally {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force -ErrorAction SilentlyContinue
}
