#!/bin/bash
# Hyperliquid HFT VPS Setup Script (Hetzner NBG1, Ubuntu 24.04)
set -euo pipefail

echo " Starting VPS setup for Hyperliquid HFT..."

if [ "$EUID" -ne 0 ]; then
  echo " Run as root (sudo $0)"
  exit 1
fi

read -p "Enter your static IP for SSH allow-list (required): " STATIC_IP
if [ -z "$STATIC_IP" ]; then
  echo " Static IP required to lock SSH"
  exit 1
fi

echo " Kernel tuning..."
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_congestion_control=bbr
sysctl -w vm.swappiness=0

cat >/etc/sysctl.d/99-hft.conf <<'EOF_CONF'
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.core.rmem_default=16777216
net.core.wmem_default=16777216
net.ipv4.tcp_congestion_control=bbr
net.ipv4.tcp_rmem=4096 87380 16777216
net.ipv4.tcp_wmem=4096 65536 16777216
net.ipv4.tcp_slow_start_after_idle=0
vm.swappiness=0
EOF_CONF
sysctl --system

echo " Firewall (ufw)..."
apt-get update -y
apt-get install -y ufw
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow from "$STATIC_IP" to any port 22 proto tcp
ufw allow 51820/udp
ufw allow from 10.100.0.0/24 to any port 3000 proto tcp
ufw allow from 10.100.0.0/24 to any port 9090 proto tcp
ufw --force enable
ufw status verbose

echo " WireGuard..."
apt-get install -y wireguard wireguard-tools
if [ -f /etc/wireguard/wg0.conf ]; then
  systemctl enable wg-quick@wg0
  systemctl start wg-quick@wg0 || true
else
  echo " /etc/wireguard/wg0.conf not found. Add config then: systemctl enable --now wg-quick@wg0"
fi

echo " System limits..."
cat >>/etc/security/limits.conf <<'EOF_LIM'
* soft nofile 1048576
* hard nofile 1048576
root soft nofile 1048576
root hard nofile 1048576
EOF_LIM

echo " systemd service template..."
cat >/etc/systemd/system/hyperliquid-hft.service <<'EOF_SVC'
[Unit]
Description=Hyperliquid HFT Trading System
After=network.target

[Service]
Type=simple
User=hyperliquid
WorkingDirectory=/opt/hyperliquid-hft
ExecStart=/opt/hyperliquid-hft/target/release/hyperliquid-hft
Restart=always
RestartSec=5
Environment="BATCH_WINDOW_MS=2"
Environment="RUST_LOG=hyperliquid_hft=info"
CPUAffinity=2-15
Nice=-10
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF_SVC
systemctl daemon-reload

echo " Setup complete"
echo "Next: place wg0.conf, reboot, deploy binary to /opt/hyperliquid-hft, enable service."
