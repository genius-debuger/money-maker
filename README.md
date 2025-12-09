# Hyperliquid L1 HFT System

High-Frequency Trading (HFT) Market Making System for Hyperliquid Layer 1 Blockchain, optimized for Hetzner Dedicated Server (Germany) deployment.

## Architecture Overview

The system follows a **Hybrid Actor Model** with three distinct phases:

1. **Phase 1: Bare Metal Infrastructure** - Kernel optimizations for ultra-low latency
2. **Phase 2: Execution Engine (Rust)** - WebSocket handling, order book reconstruction, order execution
3. **Phase 3: Alpha Engine (Python + AI)** - LiT Transformer + PPO for predictive alpha

## System Components

### Rust Execution Engine
- **WebSocket Manager**: Finite State Machine with watchdog timer and backoff logic
- **Order Book**: Double-buffered zero-allocation order book for snapshot updates
- **Order Management**: Latency-aware batching (configurable via `BATCH_WINDOW_MS` env var)
- **Strategy Processor**: Circuit breakers, funding arbitrage, rebate capture
- **IPC**: ZeroMQ communication with Python AI engine
- **Backtester**: Event-driven backtesting with priority queue and latency simulation

### Python Alpha Engine
- **Data Miner**: Downloads historical L2 order book data from Hyperliquid's S3 archive
- **LiT Transformer**: Limit Order Book Transformer predicting micro-price movements
- **Training Script**: Triple-Barrier labeling method for model training
- **Avellaneda-Stoikov Strategy**: Extended with PPO for dynamic risk aversion
- **Inference Loop**: High-performance async loop using `uvloop`

### Monitoring Stack
- **Prometheus**: Metrics scraping (1s intervals)
- **Grafana**: Real-time dashboard with PnL, latency heatmaps, inventory gauges
- **WireGuard**: Secure VPN-only access to monitoring endpoints

## Quick Start

### 1. VPS Setup (One-time)

```bash
sudo chmod +x setup_vps.sh optimize_kernel.sh
sudo ./setup_vps.sh
sudo ./optimize_kernel.sh
sudo reboot
```

### 2. Download Training Data

```bash
cd python
python data_miner.py --start-date 2024-01-01 --end-date 2024-01-31 --coins BTC ETH
```

### 3. Train LiT Model

```bash
python train.py --data-dir data/parquet --epochs 50
```

### 4. Run Backtest

```bash
cd ../..
cargo build --release
cargo run --bin backtest data/parquet/2024-01-01/BTC.parquet 15
```

### 5. Build and Deploy

```bash
cargo build --release
sudo cp target/release/hyperliquid-hft /opt/hyperliquid-hft/
sudo systemctl enable --now hyperliquid-hft
```

### 6. Start Python Alpha Engine

```bash
cd python
python main.py
```

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

### Environment Variables

- `BATCH_WINDOW_MS`: Order batching window in milliseconds (default: 2ms)
- `RUST_LOG`: Logging level (e.g., `hyperliquid_hft=info`)

### Hyperliquid Node

Edit `config/visor.json` with your node configuration. Key settings:
- Static peer: `157.90.207.92` (Imperator validator in Germany)
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
├── setup_vps.sh            # VPS setup script (firewall, WireGuard, services)
├── Cargo.toml              # Rust dependencies
├── src/
│   ├── main.rs             # Entry point
│   ├── websocket.rs        # WebSocket FSM with watchdog
│   ├── orderbook.rs        # Double-buffered order book
│   ├── execution.rs        # Order manager with latency-aware batching
│   ├── ipc.rs              # ZeroMQ IPC
│   ├── strategy.rs         # Strategy processor + circuit breakers
│   ├── metrics.rs          # Prometheus metrics endpoint
│   ├── types.rs            # Shared types
│   └── bin/
│       └── backtest.rs     # Event-driven backtesting engine
├── python/
│   ├── model.py            # LiT Transformer model
│   ├── strategy.py         # Avellaneda-Stoikov + PPO
│   ├── main.py             # Inference loop
│   ├── data_miner.py       # S3 data downloader
│   ├── train.py            # Model training script
│   └── requirements.txt    # Python dependencies
├── config/
│   └── visor.json          # Hyperliquid node configuration
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

