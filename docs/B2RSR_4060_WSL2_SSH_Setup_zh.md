# 4060 笔记本（WSL2）SSH 接入配置指南

> 更新日期：2026-07-28
> 目的：让 Mac（及 AI 助手）能免密 SSH 直连 4060 笔记本的 WSL2 Ubuntu，
> 作为本地免费 GPU 筛查机（替代云端 3060 跑探针/小实验），
> 形成「4060 本地筛查 → 方向确认 → 云端 4090 出正式数字」的两段式工作流。
> 配置完成后，环境搭建、数据准备、实验执行均由 AI 助手通过 SSH 自动完成。

---

## 0. 网络结构与原理（为什么不能"直连 WSL2"）

```text
Mac ──WiFi──> 路由器 ──WiFi──> Windows（局域网 IP，如 192.168.1.23）
                                 └─ 内部虚拟网卡 ─> WSL2 Ubuntu（172.x.x.x，仅 Windows 可见）
```

WSL2 默认运行在 Windows 内部的 NAT 网段，路由器和局域网内其他设备
看不到它。因此接入方案只有两类：

- **路线 A（mirrored 模式）**：让 WSL2 共享 Windows 的 IP，效果等同直连
  Ubuntu。要求 Windows 11 22H2+ 且 WSL ≥ 2.0。
- **路线 B（Windows 跳板）**：SSH 连 Windows 的 sshd，再用 `wsl` 命令进入
  Ubuntu。任何 Windows 版本都可用，最稳。

---

## 1. 第 0 步：分流检查（笔记本上，1 分钟）

1. `Win + R` → 输入 `winver` → 回车，记下版本号；
2. PowerShell 执行：

```powershell
wsl --version
# WSL 版本 >= 2.0.x 才支持 mirrored；显示不出来先执行 wsl --update
```

判定：

| 条件 | 走哪条路线 |
|---|---|
| Win11 ≥ 22H2 且 WSL ≥ 2.0 | **路线 A**（推荐，最优雅） |
| Win10 或旧版 Win11 | **路线 B**（跳板，最稳） |

---

## 2. 路线 A：mirrored 模式（Win11 22H2+）

### A1. 配置 mirrored 网络

记事本打开（没有就新建）`C:\Users\<你的用户名>\.wslconfig`，写入：

```ini
[wsl2]
networkingMode=mirrored
```

保存后 PowerShell 执行：

```powershell
wsl --shutdown
```

等 10 秒，重新打开 Ubuntu（开始菜单点 Ubuntu 图标）。

### A2. WSL2 Ubuntu 里安装 SSH 服务

Ubuntu 终端逐行执行：

```bash
sudo apt update && sudo apt install -y openssh-server

# 改端口为 2222（避开 Windows 可能占用的 22）
sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config
# 首次允许密码登录（之后配公钥免密）
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config

sudo service ssh restart
sudo service ssh status   # 看到 running 即可
```

### A3. Windows 防火墙放行 2222（管理员 PowerShell）

```powershell
New-NetFirewallRule -DisplayName "WSL2 SSH" -Direction Inbound -LocalPort 2222 -Protocol TCP -Action Allow
```

### A4. 查 IP 并从 Mac 测试

```powershell
ipconfig
# 记下「无线局域网适配器 WLAN」的 IPv4 地址，例如 192.168.1.23
```

Mac 终端测试（用户名是 **Ubuntu 里的用户名**，不是 Windows 的）：

```bash
ssh -p 2222 <ubuntu用户名>@192.168.1.23
# 输 Ubuntu 密码，能进即成功
```

### A5. sshd 开机自启（否则每次重启需手动启动）

Ubuntu 里：

```bash
echo "%sudo ALL=(ALL) NOPASSWD: /usr/sbin/service ssh start" | sudo tee /etc/sudoers.d/ssh-autostart
```

Windows 管理员 PowerShell（登录时静默拉起 sshd）：

```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-e sudo service ssh start"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "WSL-SSHD" -Action $action -Trigger $trigger
```

---

## 3. 路线 B：Windows 跳板（Win10 / 旧 Win11）

### B1. 安装 Windows OpenSSH Server（管理员 PowerShell）

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
```

### B2. 确认防火墙已放行

```powershell
Get-NetFirewallRule -Name *OpenSSH* | Select DisplayName, Enabled
# "OpenSSH SSH Server (sshd)" 应为 True；若不是：
# Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
```

### B3. 查 IP 并从 Mac 测试

```powershell
ipconfig   # 记 WLAN 的 IPv4
```

Mac 终端（这里用 **Windows 用户名 + Windows 密码**）：

```bash
ssh <windows用户名>@192.168.1.23
# 连上后输 wsl 回车，能进 Ubuntu 即全通
```

> ⚠️ 微软账户登录的 Windows，SSH 密码是微软账户密码。若密码始终不对，
> 直接跳到第 4 节配公钥（公钥方式绕开密码问题）。

路线 B 下 AI 执行命令的形态（供了解）：

```bash
ssh 4060 "wsl -e bash -lc 'cd ~/B2R-SR && python scripts/eval/run_cascade_oracle.py'"
```

---

## 4. 最后一步（两条路线通用）：配免密 + 交接

AI 助手执行命令时无法交互式输密码，**必须配公钥免密**。
Mac 终端（成功 ssh 过一次之后）：

```bash
# 路线 A：
ssh-copy-id -p 2222 <ubuntu用户名>@<IP>
# 路线 B：
ssh-copy-id <windows用户名>@<IP>
```

输最后一次密码，然后**再连一次确认已免密**。

完成后把三样信息交给 AI 助手：

```text
1. IP 地址
2. 用户名（A = Ubuntu 用户名 / B = Windows 用户名）
3. 走的哪条路线（A 或 B）
```

---

## 5. 交接后由 AI 完成的事（无需人工参与）

```text
① Mac 上写 ~/.ssh/config 别名（此后用 ssh 4060 直连）
② 验证 WSL2 内 nvidia-smi 能看到 RTX 4060（CUDA 直通检查）
③ clone 仓库 + 安装 PyTorch / OpenCV 等依赖
④ 传 checkpoint（120000_G.pth）+ 运行 prepare_large_benchmarks.py 准备数据
⑤ 跑通冒烟 → 全量实验 → 直接给出 Gate 判定
```

---

## 6. 常见坑速查

| 症状 | 原因 / 解法 |
|---|---|
| `Connection refused` | sshd 没启动，或防火墙没放行对应端口 |
| `Connection timed out` | 两台机器不在同一 WiFi；或路由器开了 AP 隔离（访客网络常见），换主网络 |
| 密码一直错（路线 B） | 微软账户问题，改用公钥（第 4 节）绕开 |
| `ipconfig` 出现多个 IPv4 | 认准「无线局域网适配器 WLAN」一节，忽略 vEthernet / 虚拟网卡 |
| mirrored 配置后 WSL 起不来 | 删除 `.wslconfig` 回退，改走路线 B |
| WSL 里 `nvidia-smi` 找不到 | Windows 侧装最新 NVIDIA 驱动即可（WSL 内不需要装驱动，只需 CUDA toolkit 由 pip torch 自带） |
| 重启后连不上（路线 A） | sshd 未自启，补做 A5；临时手动 `sudo service ssh start` |

---

## 7. 延伸：跨网络访问（可选，暂不配置）

若笔记本与 Mac 经常不在同一局域网（如实习期间笔记本在家），可两台设备
安装 Tailscale（免费个人版）组虚拟局域网，SSH 配置不变、IP 换成
Tailscale 分配的 100.x.x.x 即可。属于一次性配置，需要时再做。

---

## 8. 工作流定位备忘

| 设备 | 角色 | 何时使用 |
|---|---|---|
| 4060 笔记本（本地，免费） | 探针 / kill-check / 小训练筛查 | 所有 Stop/Go 判定实验 |
| 云端 4090（按量付费） | 论文正式数字 | 方向确认后的完整实验矩阵 |
| 云端 3060 | 历史延迟基线所在设备 | 需要与既有延迟数据对齐时 |

注意：**延迟判定用同设备内的比值（speedup）**，4060 筛查有效；
论文的正式延迟表必须在同一台目标设备上一次性完整重测。
4060 笔记本计时前确认插电 + 最高性能模式，
`nvidia-smi -q -d PERFORMANCE` 检查无 throttling。
