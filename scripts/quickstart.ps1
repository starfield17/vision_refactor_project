param(
    [string]$Command = "up",
    [string]$Service = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$WebPort = if ($env:CONTROL_PLANE_WEB_PORT) { $env:CONTROL_PLANE_WEB_PORT } else { "5173" }
$LogTail = if ($env:QUICKSTART_LOG_TAIL) { [int]$env:QUICKSTART_LOG_TAIL } else { 120 }
$RunDir = Join-Path $ProjectDir "work-dir\tmp\quickstart"

function Write-Info($Message) {
    Write-Host "[INFO] $Message"
}

function Write-Warn($Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function New-QuickstartDirs {
    @(
        "artifacts",
        "state",
        "stats",
        "tmp",
        "models",
        "datasets",
        "runs",
        "outputs",
        "tmp\quickstart"
    ) | ForEach-Object {
        New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "work-dir\$_") | Out-Null
    }
}

function Test-CommandAvailable($File, $Argument) {
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    try {
        & $File $Argument > $stdoutPath 2> $stderrPath
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
    finally {
        Remove-Item -Force $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
    }
}

function Assert-LocalPrerequisites {
    if (!(Test-CommandAvailable $PythonBin "--version")) {
        throw "Python is not runnable as '$PythonBin'. Set PYTHON to a Windows Python with project dependencies, or use bash scripts/quickstart.sh inside WSL."
    }
    if (!(Test-CommandAvailable "npm" "--version")) {
        throw "npm is not runnable. Install Node.js/npm on Windows, or use bash scripts/quickstart.sh inside WSL."
    }
    $vitePath = Join-Path $ProjectDir "control_plane\web\node_modules\.bin\vite.cmd"
    if (!(Test-Path $vitePath)) {
        throw "Control Plane Web dependencies are not installed. Run 'pushd control_plane\web && npm install && popd' before scripts\quickstart.bat up."
    }
}

function Get-ServiceNames {
    $names = @(
        "control-plane",
        "statistics",
        "train-worker",
        "autolabel-worker",
        "edge-agent",
        "control-plane-web"
    )
    if ($env:QUICKSTART_REMOTE -eq "1") {
        $names += "remote-worker"
    }
    return $names
}

function Get-PidPath($Name) {
    Join-Path $RunDir "$Name.pid"
}

function Get-LogPaths($Name) {
    @(
        (Join-Path $RunDir "$Name.out.log"),
        (Join-Path $RunDir "$Name.err.log")
    )
}

function Get-HealthUrl($Name) {
    switch ($Name) {
        "control-plane" { "http://127.0.0.1:7800/health" }
        "statistics" { "http://127.0.0.1:7803/health" }
        "train-worker" { "http://127.0.0.1:7811/health" }
        "autolabel-worker" { "http://127.0.0.1:7812/health" }
        "edge-agent" { "http://127.0.0.1:7813/health" }
        "remote-worker" { "http://127.0.0.1:60051/health" }
        "control-plane-web" { "http://127.0.0.1:$WebPort" }
        default { throw "unknown service: $Name" }
    }
}

function Get-ServicePort($Name) {
    switch ($Name) {
        "control-plane" { 7800 }
        "statistics" { 7803 }
        "train-worker" { 7811 }
        "autolabel-worker" { 7812 }
        "edge-agent" { 7813 }
        "remote-worker" { 60051 }
        "control-plane-web" { [int]$WebPort }
        default { throw "unknown service: $Name" }
    }
}

function Assert-PortAvailable($Name) {
    $port = Get-ServicePort $Name
    $address = if ($Name -eq "control-plane-web") { "0.0.0.0" } else { "127.0.0.1" }
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($address), $port)
        $listener.Start()
    }
    catch {
        throw "$Name port $port is already in use"
    }
    finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Get-ServiceSpec($Name) {
    switch ($Name) {
        "control-plane" {
            @{
                File = $PythonBin
                Args = @("-m", "control_plane.api", "--config", "control_plane/config/config.example.toml")
            }
        }
        "statistics" {
            @{
                File = $PythonBin
                Args = @(
                    "-m", "stats_service.api",
                    "--config", "stats_service/config/config.example.toml",
                    "--set", "control_plane.url=http://127.0.0.1:7800",
                    "--set", "server.advertise_url=http://127.0.0.1:7803"
                )
            }
        }
        "train-worker" {
            @{
                File = $PythonBin
                Args = @(
                    "-m", "train_worker.service",
                    "--config", "train_worker/config/config.example.toml",
                    "--set", "control_plane.url=http://127.0.0.1:7800",
                    "--set", "server.advertise_url=http://127.0.0.1:7811"
                )
            }
        }
        "autolabel-worker" {
            @{
                File = $PythonBin
                Args = @(
                    "-m", "autolabel_worker.service",
                    "--config", "autolabel_worker/config/config.example.toml",
                    "--set", "control_plane.url=http://127.0.0.1:7800",
                    "--set", "server.advertise_url=http://127.0.0.1:7812"
                )
            }
        }
        "edge-agent" {
            @{
                File = $PythonBin
                Args = @(
                    "-m", "edge_agent.service",
                    "--config", "edge_agent/config/config.example.toml",
                    "--set", "control_plane.url=http://127.0.0.1:7800",
                    "--set", "server.advertise_url=http://127.0.0.1:7813"
                )
            }
        }
        "remote-worker" {
            @{
                File = $PythonBin
                Args = @(
                    "-m", "remote_worker.api",
                    "--config", "remote_worker/config/config.example.toml",
                    "--set", "control_plane.url=http://127.0.0.1:7800",
                    "--set", "node.endpoint=http://127.0.0.1:60051"
                )
            }
        }
        "control-plane-web" {
            @{
                File = "powershell.exe"
                Args = @(
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "Set-Location control_plane/web; npm run dev -- --port $WebPort --strictPort"
                )
                Env = @{ VITE_CONTROL_PLANE_API_URL = "http://127.0.0.1:7800" }
            }
        }
        default {
            throw "unknown service: $Name"
        }
    }
}

function Test-ProcessRunning($Name) {
    $pidPath = Get-PidPath $Name
    if (!(Test-Path $pidPath)) {
        return $false
    }
    $rawPid = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (!$rawPid) {
        return $false
    }
    $proc = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
    return $null -ne $proc
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

function Test-Health($Name) {
    $response = $null
    try {
        $request = [System.Net.WebRequest]::Create((Get-HealthUrl $Name))
        $request.Method = "GET"
        $request.Proxy = $null
        $request.Timeout = 2000
        $response = $request.GetResponse()
        return [int]$response.StatusCode -lt 500
    }
    catch {
        return $false
    }
    finally {
        if ($response) {
            $response.Dispose()
        }
    }
}

function Wait-Health($Name) {
    for ($i = 0; $i -lt 30; $i++) {
        if (Test-Health $Name) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Start-QuickstartService($Name) {
    if (Test-ProcessRunning $Name) {
        $runningPid = Get-Content (Get-PidPath $Name) | Select-Object -First 1
        Write-Info "$Name already running (pid $runningPid)"
        return
    }
    Assert-PortAvailable $Name

    $spec = Get-ServiceSpec $Name
    $logPaths = @(Get-LogPaths $Name)
    Write-Info "Starting $Name"
    foreach ($logPath in $logPaths) {
        New-Item -ItemType File -Force -Path $logPath | Out-Null
    }

    $previousEnv = @{}
    $processEnv = @{
        NO_PROXY = "127.0.0.1,localhost,::1"
    }
    if ($env:NO_PROXY) {
        $processEnv.NO_PROXY = "$($processEnv.NO_PROXY),$env:NO_PROXY"
    }
    if ($env:no_proxy) {
        $processEnv.NO_PROXY = "$($processEnv.NO_PROXY),$env:no_proxy"
    }
    if ($spec.ContainsKey("Env")) {
        foreach ($entry in $spec.Env.GetEnumerator()) {
            $processEnv[$entry.Key] = $entry.Value
        }
    }
    if ($processEnv.Count -gt 0) {
        foreach ($entry in $processEnv.GetEnumerator()) {
            $previousEnv[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
    try {
        $proc = Start-Process `
            -FilePath $spec.File `
            -ArgumentList $spec.Args `
            -WorkingDirectory $ProjectDir `
            -RedirectStandardOutput $logPaths[0] `
            -RedirectStandardError $logPaths[1] `
            -PassThru `
            -WindowStyle Hidden
    }
    finally {
        foreach ($entry in $previousEnv.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }

    Set-Content -Path (Get-PidPath $Name) -Value $proc.Id -Encoding ascii
    Start-Sleep -Milliseconds 300
    if ($proc.HasExited) {
        Write-Warn "$Name exited during startup; logs: $($logPaths -join ', ')"
        foreach ($logPath in $logPaths) {
            if (Test-Path $logPath) {
                Get-Content $logPath -Tail 40
            }
        }
        throw "$Name failed to start"
    }
    if (Wait-Health $Name) {
        Write-Info "$Name ready at $(Get-HealthUrl $Name)"
    }
    else {
        $proc = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($null -eq $proc) {
            Remove-Item -Force (Get-PidPath $Name) -ErrorAction SilentlyContinue
            Write-Warn "$Name exited before health check passed; logs: $($logPaths -join ', ')"
            foreach ($logPath in $logPaths) {
                if (Test-Path $logPath) {
                    Get-Content $logPath -Tail 40
                }
            }
            throw "$Name failed to start"
        }
        Write-Warn "$Name started but health check did not pass yet; logs: $($logPaths -join ', ')"
    }
}

function Stop-QuickstartService($Name) {
    $pidPath = Get-PidPath $Name
    if (!(Test-Path $pidPath)) {
        Write-Info "$Name not started by quickstart"
        return
    }
    $rawPid = Get-Content $pidPath | Select-Object -First 1
    $proc = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        Remove-Item -Force $pidPath
        Write-Info "$Name not running"
        return
    }
    Write-Info "Stopping $Name (pid $rawPid)"
    Stop-ProcessTree -ProcessId $proc.Id
    Start-Sleep -Milliseconds 500
    $proc = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Stop-ProcessTree -ProcessId $proc.Id -Force $true
    }
    Remove-Item -Force $pidPath
}

function Start-All {
    New-QuickstartDirs
    Assert-LocalPrerequisites
    foreach ($name in Get-ServiceNames) {
        Start-QuickstartService $name
    }
    Write-Host "[OK] Quickstart services are running."
    Write-Host "Control Plane API: http://127.0.0.1:7800"
    Write-Host "Control Plane Web: http://127.0.0.1:$WebPort"
    Write-Host "Logs: scripts\quickstart.bat logs <service>"
    Write-Host "Stop: scripts\quickstart.bat down"
}

function Stop-All {
    New-QuickstartDirs
    $names = @(Get-ServiceNames)
    [array]::Reverse($names)
    foreach ($name in $names) {
        Stop-QuickstartService $name
    }
}

function Show-Status {
    New-QuickstartDirs
    "{0,-20} {1,-8} {2,-8} {3}" -f "SERVICE", "PID", "HEALTH", "URL"
    foreach ($name in Get-ServiceNames) {
        $pidValue = "-"
        if (Test-ProcessRunning $name) {
            $pidValue = Get-Content (Get-PidPath $name) | Select-Object -First 1
        }
        $health = if (Test-Health $name) { "ok" } else { "down" }
        "{0,-20} {1,-8} {2,-8} {3}" -f $name, $pidValue, $health, (Get-HealthUrl $name)
    }
}

function Show-Logs($Name) {
    New-QuickstartDirs
    if (!$Name) {
        Get-ServiceNames
        return
    }
    $logPaths = @(Get-LogPaths $Name | Where-Object { Test-Path $_ })
    if ($logPaths.Count -eq 0) {
        throw "no log files for ${Name}"
    }
    Get-Content -Path $logPaths -Tail $LogTail -Wait
}

function Invoke-PodmanProxy($Subcommand, $Profile) {
    if (!$Subcommand) { $Subcommand = "up" }
    if (!$Profile) { $Profile = "all-in-one" }
    & bash (Join-Path $ProjectDir "deployments/install.sh") $Subcommand $Profile
    exit $LASTEXITCODE
}

switch ($Command) {
    "up" { Start-All }
    "down" { Stop-All }
    "restart" { Stop-All; Start-All }
    "status" { Show-Status }
    "logs" { Show-Logs $Service }
    "podman" { Invoke-PodmanProxy $Service $args[0] }
    "help" {
        Write-Host "Usage: scripts\quickstart.bat [up|down|restart|status|logs <service>|podman <command> <profile>]"
    }
    default {
        throw "unknown command: $Command"
    }
}
