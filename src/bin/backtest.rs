use anyhow::{Context, Result};
use polars::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

/// Event types for the backtesting engine
#[derive(Debug, Clone, PartialEq)]
enum EventType {
    MarketUpdate {
        timestamp: u64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    },
    OrderArrival {
        timestamp: u64,
        order_id: String,
        side: OrderSide,
        price: f64,
        size: f64,
    },
    Fill {
        timestamp: u64,
        order_id: String,
        price: f64,
        size: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum OrderSide {
    Buy,
    Sell,
}

/// Event for priority queue (min-heap by timestamp)
#[derive(Debug, Clone)]
struct Event {
    timestamp: u64,
    event_type: EventType,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap (earliest timestamp first)
        other.timestamp.partial_cmp(&self.timestamp)
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.timestamp.cmp(&self.timestamp)
    }
}

/// Order book state
struct OrderBookState {
    bids: Vec<(f64, f64)>, // (price, size)
    asks: Vec<(f64, f64)>,
}

impl OrderBookState {
    fn new() -> Self {
        Self {
            bids: Vec::new(),
            asks: Vec::new(),
        }
    }

    fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        self.bids = bids.to_vec();
        self.asks = asks.to_vec();
        
        // Sort bids descending, asks ascending
        self.bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        self.asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    }

    fn get_mid_price(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }
        let best_bid = self.bids[0].0;
        let best_ask = self.asks[0].0;
        Some((best_bid + best_ask) / 2.0)
    }

    fn would_fill(&self, side: OrderSide, price: f64, size: f64) -> Option<(f64, f64)> {
        // Conservative fill: only fill if price crosses our limit
        match side {
            OrderSide::Buy => {
                // Buy order: fill only if ask price is <= our limit
                if !self.asks.is_empty() && self.asks[0].0 <= price {
                    let fill_price = self.asks[0].0;
                    let fill_size = size.min(self.asks[0].1);
                    return Some((fill_price, fill_size));
                }
            }
            OrderSide::Sell => {
                // Sell order: fill only if bid price is >= our limit
                if !self.bids.is_empty() && self.bids[0].0 >= price {
                    let fill_price = self.bids[0].0;
                    let fill_size = size.min(self.bids[0].1);
                    return Some((fill_price, fill_size));
                }
            }
        }
        None
    }
}

/// Pending order
#[derive(Clone)]
struct PendingOrder {
    order_id: String,
    side: OrderSide,
    price: f64,
    size: f64,
}

/// Backtesting engine
struct BacktestEngine {
    events: BinaryHeap<Event>,
    order_book: OrderBookState,
    pending_orders: HashMap<String, PendingOrder>,
    position: f64, // Current position (positive = long, negative = short)
    realized_pnl: f64,
    trades: Vec<(u64, f64, f64)>, // (timestamp, price, size)
    latency_ms: u64, // Simulated latency from NBG1 to validator
}

impl BacktestEngine {
    fn new(latency_ms: u64) -> Self {
        Self {
            events: BinaryHeap::new(),
            order_book: OrderBookState::new(),
            pending_orders: std::collections::HashMap::new(),
            position: 0.0,
            realized_pnl: 0.0,
            trades: Vec::new(),
            latency_ms,
        }
    }

    fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    fn submit_order(&mut self, timestamp: u64, side: OrderSide, price: f64, size: f64) {
        let order_id = format!("order_{}", timestamp);
        
        // Add order arrival event with latency
        let arrival_time = timestamp + (self.latency_ms * 1_000_000); // Convert ms to nanoseconds
        
        self.add_event(Event {
            timestamp: arrival_time,
            event_type: EventType::OrderArrival {
                timestamp: arrival_time,
                order_id: order_id.clone(),
                side,
                price,
                size,
            },
        });
    }

    fn process_event(&mut self, event: Event) {
        match event.event_type {
            EventType::MarketUpdate { bids, asks, .. } => {
                self.order_book.update(&bids, &asks);
                
                // Check if any pending orders would fill
                let mut filled_orders = Vec::new();
                
                for (order_id, order) in &self.pending_orders {
                    if let Some((fill_price, fill_size)) = 
                        self.order_book.would_fill(order.side, order.price, order.size) 
                    {
                        filled_orders.push((
                            order_id.clone(),
                            fill_price,
                            fill_size,
                            order.side,
                        ));
                    }
                }
                
                // Process fills
                for (order_id, fill_price, fill_size, side) in filled_orders {
                    self.pending_orders.remove(&order_id);
                    
                    // Update position and PnL
                    let position_change = match side {
                        OrderSide::Buy => fill_size,
                        OrderSide::Sell => -fill_size,
                    };
                    self.position += position_change;
                    
                    // Track trade
                    self.trades.push((event.timestamp, fill_price, fill_size));
                }
            }
            EventType::OrderArrival { order_id, side, price, size, .. } => {
                // Check immediate fill
                if let Some((fill_price, fill_size)) = 
                    self.order_book.would_fill(side, price, size) 
                {
                    // Immediate fill
                    let position_change = match side {
                        OrderSide::Buy => fill_size,
                        OrderSide::Sell => -fill_size,
                    };
                    self.position += position_change;
                    self.trades.push((event.timestamp, fill_price, fill_size));
                } else {
                    // Add to pending orders
                    self.pending_orders.insert(
                        order_id.clone(),
                        PendingOrder {
                            order_id,
                            side,
                            price,
                            size,
                        },
                    );
                }
            }
            EventType::Fill { .. } => {
                // Already handled in MarketUpdate
            }
        }
    }

    fn run(&mut self) -> BacktestResults {
        while let Some(event) = self.events.pop() {
            self.process_event(event);
        }

        // Close any remaining position at final price
        if let Some(final_price) = self.order_book.get_mid_price() {
            if self.position != 0.0 {
                let close_value = -self.position * final_price;
                self.realized_pnl += close_value;
            }
        }

        BacktestResults {
            total_trades: self.trades.len(),
            realized_pnl: self.realized_pnl,
            final_position: self.position,
            trades: self.trades.clone(),
        }
    }
}

/// Backtest results
struct BacktestResults {
    total_trades: usize,
    realized_pnl: f64,
    final_position: f64,
    trades: Vec<(u64, f64, f64)>,
}

impl BacktestResults {
    fn calculate_metrics(&self) -> (f64, f64, f64) {
        // Sharpe Ratio (simplified - assumes 0 risk-free rate)
        if self.trades.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let returns: Vec<f64> = self.trades
            .windows(2)
            .map(|w| {
                let (_, price1, _) = w[0];
                let (_, price2, _) = w[1];
                (price2 - price1) / price1
            })
            .collect();

        if returns.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        let sharpe = if std_dev > 0.0 {
            mean_return / std_dev * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };

        // Max Drawdown
        let mut peak = 0.0;
        let mut max_drawdown = 0.0;
        let mut cumulative_pnl = 0.0;

        for (_, price, size) in &self.trades {
            cumulative_pnl += price * size;
            if cumulative_pnl > peak {
                peak = cumulative_pnl;
            }
            let drawdown = peak - cumulative_pnl;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // ROI (assuming initial capital)
        let roi = self.realized_pnl / 100000.0 * 100.0; // Assume $100k initial

        (sharpe, max_drawdown, roi)
    }
}

fn load_parquet_data(file_path: &Path) -> Result<Vec<(u64, Vec<(f64, f64)>, Vec<(f64, f64)>)>> {
    let df = LazyFrame::scan_parquet(file_path, ScanArgsParquet::default())?
        .collect()
        .context("Failed to collect DataFrame")?;

    // Expect wide columns: bid_px_0..19, bid_sz_0..19, ask_px_0..19, ask_sz_0..19
    let mut snapshots = Vec::new();
    for row in df.iter() {
        let ts = row
            .0
            .get(0)
            .and_then(|v| v.try_extract::<u64>().ok())
            .unwrap_or(0);

        let mut bids = Vec::new();
        let mut asks = Vec::new();
        for level in 0..20 {
            let bp = row
                .0
                .get(df.find_idx(&format!("bid_px_{}", level)).unwrap_or(0))
                .and_then(|v| v.try_extract::<f64>().ok())
                .unwrap_or(0.0);
            let bs = row
                .0
                .get(df.find_idx(&format!("bid_sz_{}", level)).unwrap_or(0))
                .and_then(|v| v.try_extract::<f64>().ok())
                .unwrap_or(0.0);
            if bp > 0.0 && bs > 0.0 {
                bids.push((bp, bs));
            }

            let ap = row
                .0
                .get(df.find_idx(&format!("ask_px_{}", level)).unwrap_or(0))
                .and_then(|v| v.try_extract::<f64>().ok())
                .unwrap_or(0.0);
            let az = row
                .0
                .get(df.find_idx(&format!("ask_sz_{}", level)).unwrap_or(0))
                .and_then(|v| v.try_extract::<f64>().ok())
                .unwrap_or(0.0);
            if ap > 0.0 && az > 0.0 {
                asks.push((ap, az));
            }
        }
        snapshots.push((ts, bids, asks));
    }
    Ok(snapshots)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: backtest <parquet_file> [latency_ms]");
        std::process::exit(1);
    }

    let file_path = Path::new(&args[1]);
    let latency_ms = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15); // Default 15ms latency

    println!("🔬 Starting backtest...");
    println!("File: {}", file_path.display());
    println!("Simulated latency: {}ms", latency_ms);

    // Load data
    let snapshots = load_parquet_data(file_path)
        .context("Failed to load Parquet data")?;

    if snapshots.is_empty() {
        eprintln!("❌ No data found in file");
        return Ok(());
    }

    // Create backtest engine
    let mut engine = BacktestEngine::new(latency_ms);

    // Load market updates
    for (timestamp, bids, asks) in snapshots {
        engine.add_event(Event {
            timestamp,
            event_type: EventType::MarketUpdate {
                timestamp,
                bids,
                asks,
            },
        });
    }

    // Example strategy: Simple market making
    // In production, this would come from the actual strategy
    for i in 0..snapshots.len().min(1000) {
        if let Some((timestamp, _, _)) = snapshots.get(i) {
            if let Some(mid_price) = engine.order_book.get_mid_price() {
                // Place bid and ask around mid price
                let spread = mid_price * 0.0001; // 1bp spread
                engine.submit_order(*timestamp, OrderSide::Buy, mid_price - spread, 0.1);
                engine.submit_order(*timestamp, OrderSide::Sell, mid_price + spread, 0.1);
            }
        }
    }

    // Run backtest
    let results = engine.run();

    // Calculate metrics
    let (sharpe, max_drawdown, roi) = results.calculate_metrics();

    // Print results
    println!("\n📊 Backtest Results:");
    println!("  Total Trades: {}", results.total_trades);
    println!("  Realized PnL: ${:.2}", results.realized_pnl);
    println!("  Final Position: {:.6}", results.final_position);
    println!("  Sharpe Ratio: {:.4}", sharpe);
    println!("  Max Drawdown: ${:.2}", max_drawdown);
    println!("  ROI: {:.2}%", roi);

    Ok(())
}

