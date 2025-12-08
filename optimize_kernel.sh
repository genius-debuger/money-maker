#!/bin/bash
# Hyperliquid HFT Kernel Optimization Script
# For Hetzner NBG1-DC3 (Ubuntu 24.04 LTS)
# Target: AMD Ryzen 9 7950X

set -e

echo "🔧 Starting HFT Kernel Optimization..."

# Check for root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root"
    exit 1
fi

# Backup original grub configuration
cp /etc/default/grub /etc/default/grub.backup.$(date +%Y%m%d_%H%M%S)

echo "📝 Configuring CPU isolation..."
# Update GRUB to isolate CPUs 2-15 (reserve 0-1 for OS)
GRUB_CMDLINE_LINUX=$(grep GRUB_CMDLINE_LINUX /etc/default/grub | sed 's/.*GRUB_CMDLINE_LINUX="//;s/".*//')
if [[ ! "$GRUB_CMDLINE_LINUX" =~ "isolcpus=2-15" ]]; then
    GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX isolcpus=2-15 nohz_full=2-15 rcu_nocbs=2-15"
    sed -i "s|GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX=\"$GRUB_CMDLINE_LINUX\"|" /etc/default/grub
fi

# CPU Governor to Performance
echo "⚡ Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null || true
done

# Disable CPU frequency scaling
systemctl disable ondemand || true
systemctl stop ondemand || true

echo "🌐 Optimizing network stack..."
# TCP BBR Congestion Control
echo 'net.core.default_qdisc=fq' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_congestion_control=bbr' >> /etc/sysctl.conf

# Disable TCP slow start
echo 'net.ipv4.tcp_slow_start_after_idle=0' >> /etc/sysctl.conf

# Socket buffer sizes for micro-bursts
echo 'net.core.rmem_max=16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max=16777216' >> /etc/sysctl.conf
echo 'net.core.rmem_default=16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_default=16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem=4096 87380 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem=4096 65536 16777216' >> /etc/sysctl.conf

# TCP optimizations
echo 'net.ipv4.tcp_tw_reuse=1' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_fin_timeout=30' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog=8192' >> /etc/sysctl.conf
echo 'net.core.netdev_max_backlog=16384' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_fastopen=3' >> /etc/sysctl.conf

# Disable swap
echo 'vm.swappiness=1' >> /etc/sysctl.conf

# Apply sysctl changes
sysctl -p

echo "🔌 Configuring IRQ affinity..."
# Disable irqbalance
systemctl stop irqbalance
systemctl disable irqbalance

# Get network interface (assuming primary interface)
INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
if [ -z "$INTERFACE" ]; then
    INTERFACE=$(ls /sys/class/net | grep -v lo | head -1)
fi

echo "📡 Binding network interrupts to Core 0 for interface: $INTERFACE"

# Bind all IRQs for the network interface to CPU 0
for irq in $(grep "$INTERFACE" /proc/interrupts | awk -F: '{print $1}'); do
    echo 1 > /proc/irq/$irq/smp_affinity
done

# Also bind other common network IRQs
for irq in /proc/irq/*/smp_affinity; do
    if [ -f "$irq" ]; then
        echo 1 > "$irq" 2>/dev/null || true
    fi
done

echo "🚫 Disabling unnecessary services..."
# Disable services that might cause latency
systemctl stop apparmor || true
systemctl disable apparmor || true
systemctl stop snapd || true
systemctl disable snapd || true
systemctl stop systemd-resolved || true
systemctl disable systemd-resolved || true

# Disable transparent hugepages
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
echo 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' >> /etc/rc.local
echo 'echo never > /sys/kernel/mm/transparent_hugepage/defrag' >> /etc/rc.local

# Increase file descriptor limits
echo '* soft nofile 1048576' >> /etc/security/limits.conf
echo '* hard nofile 1048576' >> /etc/security/limits.conf
echo 'root soft nofile 1048576' >> /etc/security/limits.conf
echo 'root hard nofile 1048576' >> /etc/security/limits.conf

# Tune scheduler for low latency
echo 'kernel.sched_migration_cost_ns=5000000' >> /etc/sysctl.conf
echo 'kernel.sched_autogroup_enabled=0' >> /etc/sysctl.conf

# Update GRUB
update-grub

echo "✅ Kernel optimization complete!"
echo ""
echo "⚠️  IMPORTANT: Reboot the system for changes to take effect:"
echo "   sudo reboot"
echo ""
echo "After reboot, verify settings with:"
echo "   cat /proc/cmdline | grep isolcpus"
echo "   sysctl net.ipv4.tcp_congestion_control"
echo "   cat /sys/kernel/mm/transparent_hugepage/enabled"

