param(
  [int]$WaitPid = 0,
  [string[]]$Generators = @("midjourney", "adm", "biggan"),
  [string]$ProjectRoot = "C:\Users\okkah\Code Repository\School Projects\AI Information Detection"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

$logPath = Join-Path $ProjectRoot "reports\genimage_queue.log"
"[$(Get-Date -Format s)] Queue starting. WaitPid=$WaitPid Generators=$($Generators -join ',')" | Out-File -FilePath $logPath -Append -Encoding utf8

if ($WaitPid -gt 0) {
  while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
  }
  "[$(Get-Date -Format s)] WaitPid $WaitPid finished." | Out-File -FilePath $logPath -Append -Encoding utf8
}

foreach ($generator in $Generators) {
  $datasetDir = Join-Path $ProjectRoot ("Dataset\prepared\external_eval\genimage_{0}_val" -f (
      switch ($generator) {
        "sd14" { "stable_diffusion_v1_4" }
        default { $generator }
      }
    ))

  if (Test-Path $datasetDir) {
    "[$(Get-Date -Format s)] Skipping $generator because $datasetDir already exists." | Out-File -FilePath $logPath -Append -Encoding utf8
    continue
  }

  "[$(Get-Date -Format s)] Starting $generator download/prep." | Out-File -FilePath $logPath -Append -Encoding utf8
  try {
    python src\cnn\download_genimage_subset.py --generator $generator *>> $logPath
    "[$(Get-Date -Format s)] Completed $generator." | Out-File -FilePath $logPath -Append -Encoding utf8
  } catch {
    "[$(Get-Date -Format s)] Failed $generator : $($_.Exception.Message)" | Out-File -FilePath $logPath -Append -Encoding utf8
    break
  }
}

"[$(Get-Date -Format s)] Queue finished." | Out-File -FilePath $logPath -Append -Encoding utf8
