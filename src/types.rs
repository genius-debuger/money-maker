use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookUpdate {
    pub symbol: String,
    pub sequence: u64,
    pub bids: Vec<(f64, f64)>, // (price, size)
    pub asks: Vec<(f64, f64)>, // (price, size)
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderInstruction {
    pub symbol: String,
    pub side: OrderSide,
    pub price: f64,
    pub size: f64,
    pub order_type: OrderType,
    pub reduce_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    PostOnly,
    Market,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevel {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub mid_price: f64,
    pub spread: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub price: f64,
    pub size: f64,
    pub filled: f64,
    pub status: OrderStatus,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct PnLState {
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub maker_rebates: f64,
    pub inventory: f64,
    pub drawdown: f64,
    pub peak_equity: f64,
    pub last_reset: Instant,
}

#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    pub tick_to_trade: Duration,
    pub book_update_latency: Duration,
    pub signal_processing: Duration,
}

