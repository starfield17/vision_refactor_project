param(
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WebPort = if ($env:CONTROL_PLANE_WEB_PORT) { [int]$env:CONTROL_PLANE_WEB_PORT } else { 5173 }
$RunDir = Join-Path $ProjectDir "work-dir\tmp\quickstart"

function Write-Info($Message) {
    if (!$Quiet) {
        Write-Host "[INFO] $Message"
    }
}

function Write-Warn($Message) {
    if (!$Quiet) {
        Write-Host "[WARN] $Message" -ForegroundColor Yellow
    }
}

function Get-ServiceNames {
    @(
        "control-plane",
        "statistics",
        "edge-agent",
        "control-plane-web",
        "remote-worker"
    )
}

function Get-ServicePort($Name) {
    switch ($Name) {
        "control-plane" { 7800 }
        "statistics" { 7803 }
        "edge-agent" { 7813 }
        "remote-worker" { 60051 }
        "control-plane-web" { $WebPort }
        default { throw "unknown service: $Name" }
    }
}

function Get-PidPath($Name) {
    Join-Path $RunDir "$Name.pid"
}

function Stop-ProcessTree($ProcessId, [bool]$Force = $false) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId -Force $Force
    }
    if ($Force) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
    else {
        Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    }
}

function Test-QuickstartProcess($Process) {
    if ($null -eq $Process -or !$Process.CommandLine) {
        return $false
    }
    $command = [string]$Process.CommandLine
    return (
        $command.Contains(" -m control_plane.api") -or
        $command.Contains(" -m stats_service.api") -or
        $command.Contains(" -m edge_agent.service") -or
        $command.Contains(" -m remote_worker.api") -or
        $command.Contains("npm --prefix control_plane/web run dev") -or
        $command.Contains("npm run dev -- --port $WebPort --strictPort") -or
        $command.Contains("vite --host 0.0.0.0 --port $WebPort") -or
        ($command.Contains("node_modules\.bin\vite") -and $command.Contains("--port $WebPort")) -or
        ($command.Contains("node_modules\vite\bin\vite.js") -and $command.Contains("--port $WebPort"))
    )
}

function Stop-QuickstartPid($ProcessId, $Label) {
    if (!$ProcessId) {
        return
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        return
    }
    Write-Info "Terminating $Label (pid $ProcessId)"
    Stop-ProcessTree -ProcessId ([int]$ProcessId)
    Start-Sleep -Milliseconds 500
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Stop-ProcessTree -ProcessId ([int]$ProcessId) -Force $true
    }
}

function Stop-PidFiles {
    if (!(Test-Path $RunDir)) {
        return
    }
    foreach ($name in Get-ServiceNames) {
        $pidPath = Get-PidPath $name
        if (!(Test-Path $pidPath)) {
            continue
        }
        $rawPid = Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
        Stop-QuickstartPid $rawPid $name
        Remove-Item -Force $pidPath -ErrorAction SilentlyContinue
    }
}

function Stop-Ports {
    foreach ($name in Get-ServiceNames) {
        $port = Get-ServicePort $name
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($connection in $connections) {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue
            if (Test-QuickstartProcess $process) {
                Stop-QuickstartPid $connection.OwningProcess "$name listener on port $port"
            }
            elseif ($connection.OwningProcess) {
                Write-Warn "Skipping pid $($connection.OwningProcess) on port $port; command does not match quickstart services"
            }
        }
    }
}

function Stop-KnownProcesses {
    $processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        Test-QuickstartProcess $_
    }
    foreach ($process in $processes) {
        Stop-QuickstartPid $process.ProcessId "quickstart process"
    }
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Stop-PidFiles
Stop-Ports
Stop-KnownProcesses
Write-Info "Terminate complete"
