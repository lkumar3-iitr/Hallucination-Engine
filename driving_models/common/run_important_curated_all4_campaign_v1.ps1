param(
    [string]$PythonExe = "C:\Users\Admin\anaconda3\envs\he_neat\python.exe",
    [string]$AssetRoot = "D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production",
    [string]$OutputRoot = "paper_results\important_curated_all4_v1",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 2000,
    [string]$Device = "cuda",
    [int]$MaxFrames = -1
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Runner = Join-Path $RepoRoot "driving_models\common\run_four_model_benchmark_v1.py"
$LogRoot = Join-Path $RepoRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$Suites = @(
    @{
        Name = "left_cutin_stop_resume"
        Manifest = "important_curated_scenarios\suite_manifest.json"
        Output = "left_cutin_stop_resume"
    },
    @{
        Name = "overtake_static_targeted"
        Manifest = "important_curated_scenarios\overtake_static_targeted_suite_manifest_v1.json"
        Output = "overtake_static_targeted"
    },
    @{
        Name = "multi_actor_safety_critical"
        Manifest = "important_curated_scenarios\multi_actor_safety_critical_suite_manifest_v1.json"
        Output = "multi_actor_safety_critical"
    }
)

Set-Location $RepoRoot
$env:HE_DEBUG_ANCHOR = ""

foreach ($Suite in $Suites) {
    $ManifestPath = Join-Path $RepoRoot $Suite.Manifest
    $SuiteOutput = Join-Path $LogRoot $Suite.Output
    $SuiteLog = Join-Path $SuiteOutput "campaign_stdout.log"
    New-Item -ItemType Directory -Force -Path $SuiteOutput | Out-Null

    $ArgsList = @(
        $Runner,
        "--suite-manifest", $ManifestPath,
        "--asset-root", $AssetRoot,
        "--output-root", $SuiteOutput,
        "--models", "tcp", "neat", "cilpp", "aimmt",
        "--conditions", "carla", "he",
        "--device", $Device,
        "--host", $HostName,
        "--port", "$Port",
        "--he-renderer-version", "v2",
        "--resume",
        "--continue-on-error"
    )

    if ($MaxFrames -ge 0) {
        $ArgsList += @("--max-frames", "$MaxFrames")
    }

    Write-Host ""
    Write-Host "===================================================================================================="
    Write-Host ("Running curated suite: {0}" -f $Suite.Name)
    Write-Host ("Manifest: {0}" -f $ManifestPath)
    Write-Host ("Output:   {0}" -f $SuiteOutput)
    Write-Host ("Log:      {0}" -f $SuiteLog)
    Write-Host "===================================================================================================="

    & $PythonExe @ArgsList 2>&1 | Tee-Object -FilePath $SuiteLog
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        Write-Host ("Suite {0} exited with code {1}; continuing because --continue-on-error is enabled." -f $Suite.Name, $ExitCode)
    }
}

Write-Host ""
Write-Host "Curated all-four-model campaign launcher finished."
Write-Host ("Output root: {0}" -f (Resolve-Path $LogRoot))
