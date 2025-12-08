# Hyperliquid L1 HFT System

High-Frequency Trading (HFT) Market Making System for Hyperliquid Layer 1 Blockchain, optimized for Hetzner Dedicated Server (Germany) deployment.

## Architecture Overview

The system follows a **Hybrid Actor Model** with three distinct phases:

1. **Phase 1: Bare Metal Infrastructure** - Kernel optimizations for ultra-low latency
2. **Phase 2: Execution Engine (Rust)** - WebSocket handling, order book reconstruction, order execution
3. **Phase 3: Alpha Engine (Python + AI)** - LiT Transformer + PPO for predictive alpha

## System Components

### Rust Execution Engine
- **Order Book**: Lock-free order book reconstruction using `DashMap` and `BTreeMap`
- **Order Management**: Batching logic to minimize rate-limit usage (1 unit for N orders)
- **Strategy Processor**: Circuit breakers, funding arbitrage, rebate capture
- **IPC**: ZeroMQ communication with Python AI engine

### Python Alpha Engine
- **LiT Transformer**: Limit Order Book Transformer predicting micro-price movements
- **Avellaneda-Stoikov Strategy**: Extended with PPO for dynamic risk aversion
- **Inference Loop**: High-performance async loop using `uvloop`

### Monitoring Stack
- **Prometheus**: Metrics scraping (1s intervals)
- **Grafana**: Real-time dashboard with PnL, latency heatmaps, inventory gauges
- **WireGuard**: Secure VPN-only access to monitoring endpoints

## Quick Start

### 1. Kernel Optimization

```bash
sudo chmod +x optimize_kernel.sh
sudo ./optimize_kernel.sh
sudo reboot
```

### 2. Build Rust Components

```bash
cd hyperliquid-hft
cargo build --release
```

### 3. Setup Python Environment

```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure WireGuard

Edit `wg0.conf` with your WireGuard keys and deploy:

```bash
sudo cp wg0.conf /etc/wireguard/
sudo wg-quick up wg0
```

### 5. Start Services

**Rust Execution Engine:**
```bash
./target/release/hyperliquid-hft
```

**Python Alpha Engine:**
```bash
cd python
python main.py
```

**Prometheus:**
```bash
prometheus --config.file=prometheus.yml
```

**Grafana:**
```bash
# Import dashboard.json into Grafana
```

## Configuration

### Hyperliquid Node

Edit `visor.json` with your node configuration. Key settings:
- Static peer: `157.90.207.92` (Imperator node)
- Disable output file buffering for lower latency
- CPU pinning to isolated cores (2-15)

### Strategy Parameters

Edit `src/strategy.rs` for:
- Circuit breaker thresholds (latency: 150ms, drawdown: 3%)
- Funding threshold: 0.002% hourly
- Risk aversion parameters

### Python Model

Edit `python/model.py` for:
- Sequence length (default: 100)
- Transformer dimensions
- LSTM hidden size

## Circuit Breakers

The system includes hard-coded circuit breakers that trigger **before** AI logic:

1. **Latency Circuit Breaker**: Cancels all orders if `tick_to_trade > 150ms`
2. **Drawdown Circuit Breaker**: Panic closes (market sell) if `drawdown > 3%` in 1 hour

## Performance Targets

- **Tick-to-Trade Latency**: < 50ms (p95), < 150ms (p99)
- **Order Book Update**: < 10ms
- **Inference Time**: < 20ms (GPU), < 50ms (CPU)

## Monitoring

Access Grafana dashboard at `http://10.100.0.1:3000` (WireGuard VPN only).

Key metrics:
- Real-time PnL & Maker Rebates
- Inventory Skew (-100% to +100%)
- Tick-to-Trade Latency Heatmap
- Circuit Breaker Status

## Security

- All monitoring endpoints restricted to WireGuard VPN (`10.100.0.0/24`)
- Circuit breakers hard-coded in Rust (cannot be bypassed by AI)
- Non-validator node (read-only blockchain access)

## Dependencies

### Rust
- `tokio` 1.32
- `hyperliquid-rust-sdk` 0.4
- `zeromq` 0.4
- `dashmap` 5.5

### Python
- `torch` >= 2.0.0
- `pyzmq` >= 25.0.0
- `uvloop` >= 0.19.0

## File Structure

```
.
├── optimize_kernel.sh      # Kernel optimization script
├── visor.json              # Hyperliquid node configuration
├── Cargo.toml              # Rust dependencies
├── src/
│   ├── main.rs             # Entry point
│   ├── orderbook.rs        # Lock-free order book
│   ├── execution.rs        # Order manager with batching
│   ├── ipc.rs              # ZeroMQ IPC
│   ├── strategy.rs         # Strategy processor + circuit breakers
│   ├── metrics.rs          # Prometheus metrics endpoint
│   └── types.rs            # Shared types
├── python/
│   ├── model.py            # LiT Transformer model
│   ├── strategy.py         # Avellaneda-Stoikov + PPO
│   ├── main.py             # Inference loop
│   └── requirements.txt    # Python dependencies
├── wg0.conf                # WireGuard configuration
├── prometheus.yml          # Prometheus scrape config
└── dashboard.json          # Grafana dashboard
```

## Notes

- The Rust execution engine uses CPU isolation (cores 2-15)
- Python AI engine should run on GPU if available
- Network interrupts pinned to Core 0
- All orders are Post-Only by default for rebate capture

## License

Proprietary - HFT Trading System

