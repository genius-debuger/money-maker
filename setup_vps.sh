#!/bin/bash
# Hyperliquid HFT VPS Setup Script
# Configures kernel, WireGuard, and firewall for production deployment

set -e

echo "🚀 Starting VPS setup for Hyperliquid HFT..."

# Check for root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (sudo $0)"
    exit 1
fi

# Get user's static IP for SSH restriction
read -p "Enter your static IP address for SSH access (or press Enter to skip): " STATIC_IP

# ============================================================================
# Phase 1: Kernel Tuning for HFT
# ============================================================================
echo ""
echo "📝 Phase 1: Kernel Tuning..."

# TCP BBR Congestion Control
cat >> /etc/sysctl.conf <<EOF

# Hyperliquid HFT Kernel Tuning
net.core.default_qdisc=fq
net.ipv4.tcp_congestion_control=bbr

# Network buffer sizes (16MB for bursty snapshots)
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.core.rmem_default=16777216
net.core.wmem_default=16777216
net.ipv4.tcp_rmem=4096 87380 16777216
net.ipv4.tcp_wmem=4096 65536 16777216

# TCP optimizations
net.ipv4.tcp_tw_reuse=1
net.ipv4.tcp_fin_timeout=30
net.ipv4.tcp_max_syn_backlog=8192
net.core.netdev_max_backlog=16384
net.ipv4.tcp_fastopen=3
net.ipv4.tcp_slow_start_after_idle=0

# Memory and swap
vm.swappiness=0
vm.overcommit_memory=1

# Scheduler tuning
kernel.sched_migration_cost_ns=5000000
kernel.sched_autogroup_enabled=0
EOF

sysctl -p

echo "✅ Kernel tuning applied"

# ============================================================================
# Phase 2: WireGuard Installation and Configuration
# ============================================================================
echo ""
echo "📝 Phase 2: WireGuard Setup..."

if ! command -v wg &> /dev/null; then
    echo "Installing WireGuard..."
    apt-get update
    apt-get install -y wireguard wireguard-tools
else
    echo "WireGuard already installed"
fi

# Check if wg0.conf exists
if [ ! -f /etc/wireguard/wg0.conf ]; then
    echo "⚠️  WireGuard config not found at /etc/wireguard/wg0.conf"
    echo "   Please create it manually or copy from the repository"
else
    echo "✅ WireGuard config found"
    
    # Enable WireGuard
    systemctl enable wg-quick@wg0
    systemctl start wg-quick@wg0 2>/dev/null || true
    
    echo "✅ WireGuard started"
fi

# ============================================================================
# Phase 3: UFW Firewall Configuration
# ============================================================================
echo ""
echo "📝 Phase 3: Firewall Configuration..."

# Install UFW if not present
if ! command -v ufw &> /dev/null; then
    apt-get install -y ufw
fi

# Reset UFW to defaults
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (restricted to static IP if provided)
if [ -n "$STATIC_IP" ]; then
    echo "🔒 Restricting SSH to IP: $STATIC_IP"
    ufw allow from $STATIC_IP to any port 22 proto tcp
    echo "⚠️  WARNING: SSH is restricted to $STATIC_IP"
    echo "   Make sure you have another way to access the server if this IP changes!"
else
    echo "⚠️  WARNING: SSH is open to all IPs (not recommended for production)"
    ufw allow 22/tcp
fi

# Allow WireGuard
ufw allow 51820/udp

# Allow Grafana only from WireGuard network (10.100.0.0/24)
ufw allow from 10.100.0.0/24 to any port 3000 proto tcp

# Allow Prometheus only from WireGuard network
ufw allow from 10.100.0.0/24 to any port 9090 proto tcp
ufw allow from 10.100.0.0/24 to any port 9091 proto tcp

# Allow localhost for metrics (Rust bot)
ufw allow from 127.0.0.1 to any port 9090 proto tcp

# Enable UFW
ufw --force enable

echo "✅ Firewall configured"
ufw status verbose

# ============================================================================
# Phase 4: Disable Unnecessary Services
# ============================================================================
echo ""
echo "📝 Phase 4: Service Optimization..."

# Disable services that might cause latency
systemctl stop apparmor 2>/dev/null || true
systemctl disable apparmor 2>/dev/null || true

systemctl stop snapd 2>/dev/null || true
systemctl disable snapd 2>/dev/null || true

systemctl stop systemd-resolved 2>/dev/null || true
systemctl disable systemd-resolved 2>/dev/null || true

# Disable irqbalance (already done in optimize_kernel.sh, but double-check)
systemctl stop irqbalance 2>/dev/null || true
systemctl disable irqbalance 2>/dev/null || true

echo "✅ Services optimized"

# ============================================================================
# Phase 5: File Descriptor Limits
# ============================================================================
echo ""
echo "📝 Phase 5: System Limits..."

cat >> /etc/security/limits.conf <<EOF

# Hyperliquid HFT Limits
* soft nofile 1048576
* hard nofile 1048576
root soft nofile 1048576
root hard nofile 1048576
EOF

echo "✅ File descriptor limits increased"

# ============================================================================
# Phase 6: Create Systemd Service for Hyperliquid HFT
# ============================================================================
echo ""
echo "📝 Phase 6: Creating systemd service..."

cat > /etc/systemd/system/hyperliquid-hft.service <<EOF
[Unit]
Description=Hyperliquid HFT Trading System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/hyperliquid-hft
ExecStart=/opt/hyperliquid-hft/target/release/hyperliquid-hft
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables
Environment="BATCH_WINDOW_MS=2"
Environment="RUST_LOG=hyperliquid_hft=info"

# CPU affinity (isolated cores 2-15)
CPUAffinity=2-15

# Nice priority
Nice=-10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo "✅ Systemd service created (not enabled - enable with: systemctl enable hyperliquid-hft)"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "✅ VPS setup complete!"
echo ""
echo "📋 Summary:"
echo "  • Kernel tuning applied"
echo "  • WireGuard installed and configured"
echo "  • UFW firewall configured"
echo "  • System limits increased"
echo "  • Systemd service created"
echo ""
echo "⚠️  Next steps:"
echo "  1. Configure WireGuard keys in /etc/wireguard/wg0.conf"
echo "  2. Reboot the system for all kernel changes to take effect"
echo "  3. Test WireGuard connection"
echo "  4. Deploy your trading bot to /opt/hyperliquid-hft"
echo "  5. Enable the service: systemctl enable --now hyperliquid-hft"
echo ""
echo "🔒 Security notes:"
if [ -n "$STATIC_IP" ]; then
    echo "  • SSH is restricted to: $STATIC_IP"
else
    echo "  • ⚠️  SSH is open to all IPs - consider restricting it!"
fi
echo "  • Grafana/Prometheus only accessible via WireGuard VPN"
echo "  • Monitoring endpoints bound to localhost or VPN network only"
echo ""

