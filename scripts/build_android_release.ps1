param(
    [string]$ApiBaseUrl = 'https://fabricare1-markkennethkenets-projects.vercel.app/api'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
# A short socket directory avoids Java's Windows Unix-domain loopback error.
$socketDirectory = Join-Path (Split-Path $projectRoot -Parent) 'java-tmp'
New-Item -ItemType Directory -Force -Path $socketDirectory | Out-Null
$previousJavaOptions = $env:JAVA_TOOL_OPTIONS
$socketPath = $socketDirectory.Replace('\', '/')
try {
    $env:JAVA_TOOL_OPTIONS = "$previousJavaOptions -Djdk.net.unixdomain.tmpdir=$socketPath -Djava.io.tmpdir=$socketPath".Trim()
    Push-Location $projectRoot
    try {
        & flutter build apk --release "--dart-define=API_BASE_URL=$ApiBaseUrl"
        if ($LASTEXITCODE -ne 0) {
            throw "Android release build failed (exit $LASTEXITCODE)."
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:JAVA_TOOL_OPTIONS = $previousJavaOptions
}
