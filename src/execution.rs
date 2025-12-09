use crate::types::{Order, OrderInstruction, OrderSide, OrderStatus, OrderType};
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use hyperliquid_rust_sdk as hlsdk;

pub struct OrderManager {
    orders: Arc<RwLock<HashMap<String, Order>>>,
    pending_batch: Arc<RwLock<Vec<OrderInstruction>>>,
    batch_tx: mpsc::UnboundedSender<Vec<OrderInstruction>>,
    last_batch_time: Arc<RwLock<Instant>>,
    batch_window: Duration,
    max_batch: usize,
}

impl OrderManager {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<Vec<OrderInstruction>>) {
        // Get batch window from environment variable, default to 2ms
        let batch_window_ms = env::var("BATCH_WINDOW_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2);
        let max_batch = 10;
        tracing::info!("OrderManager initialized with batch window: {}ms", batch_window_ms);
        
        let (tx, rx) = mpsc::unbounded_channel();
        let manager = Self {
            orders: Arc::new(RwLock::new(HashMap::new())),
            pending_batch: Arc::new(RwLock::new(Vec::new())),
            batch_tx: tx,
            last_batch_time: Arc::new(RwLock::new(Instant::now())),
            batch_window: Duration::from_millis(batch_window_ms),
            max_batch,
        };
        (manager, rx)
    }

    pub fn get_batch_window(&self) -> Duration {
        self.batch_window
    }

    pub fn submit_order(&self, instruction: OrderInstruction) -> Result<String> {
        let order_id = format!("{}_{}", instruction.symbol, Instant::now().as_nanos());
        
        let mut should_flush = false;
        {
            let mut pending = self.pending_batch.write();
            pending.push(instruction);
            if pending.len() >= self.max_batch {
                should_flush = true;
            }
        }

        // Check if batch window expired
        if !should_flush {
            let should = {
                let last_time = self.last_batch_time.read();
                Instant::now().duration_since(*last_time) >= self.batch_window
            };
            should_flush = should;
        }

        if should_flush {
            self.flush_batch()?;
        }

        Ok(order_id)
    }

    pub fn flush_batch(&self) -> Result<()> {
        let mut pending = self.pending_batch.write();
        if pending.is_empty() {
            return Ok(());
        }

        let batch = pending.clone();
        pending.clear();
        
        *self.last_batch_time.write() = Instant::now();
        
        self.batch_tx.send(batch).map_err(|e| anyhow::anyhow!("Failed to send batch: {}", e))?;
        
        Ok(())
    }

    pub fn cancel_order(&self, order_id: &str) -> Result<()> {
        let mut orders = self.orders.write();
        if let Some(order) = orders.get_mut(order_id) {
            if order.status == OrderStatus::Open || order.status == OrderStatus::PartiallyFilled {
                order.status = OrderStatus::Cancelled;
            }
        }
        Ok(())
    }

    pub fn cancel_all(&self) -> Result<usize> {
        let mut orders = self.orders.write();
        let mut cancelled = 0;
        
        for order in orders.values_mut() {
            if order.status == OrderStatus::Open || order.status == OrderStatus::PartiallyFilled {
                order.status = OrderStatus::Cancelled;
                cancelled += 1;
            }
        }
        
        Ok(cancelled)
    }

    pub fn update_order_status(&self, order_id: &str, status: OrderStatus, filled: f64) {
        let mut orders = self.orders.write();
        if let Some(order) = orders.get_mut(order_id) {
            order.status = status.clone();
            order.filled = filled;
            
            if status == OrderStatus::Filled || status == OrderStatus::Cancelled {
                orders.remove(order_id);
            }
        }
    }

    pub fn get_order(&self, order_id: &str) -> Option<Order> {
        self.orders.read().get(order_id).cloned()
    }

    pub fn get_open_orders(&self) -> Vec<Order> {
        self.orders
            .read()
            .values()
            .filter(|o| o.status == OrderStatus::Open || o.status == OrderStatus::PartiallyFilled)
            .cloned()
            .collect()
    }

    pub fn get_open_orders_for_symbol(&self, symbol: &str) -> Vec<Order> {
        self.orders
            .read()
            .values()
            .filter(|o| {
                o.symbol == symbol
                    && (o.status == OrderStatus::Open || o.status == OrderStatus::PartiallyFilled)
            })
            .cloned()
            .collect()
    }
}

pub async fn batch_processor(
    mut batch_rx: mpsc::UnboundedReceiver<Vec<OrderInstruction>>,
) -> Result<()> {
    let client = hlsdk::Client::new_default();

    while let Some(batch) = batch_rx.recv().await {
        if batch.is_empty() {
            continue;
        }

        // Convert to batchModify format for Hyperliquid
        // This consumes only 1 rate-limit unit for N orders
        let orders: Vec<hlsdk::types::BatchOrder> = batch
            .iter()
            .map(|instr| hlsdk::types::BatchOrder {
                symbol: instr.symbol.clone(),
                size: instr.size,
                price: instr.price,
                reduce_only: instr.reduce_only,
                side: match instr.side {
                    OrderSide::Buy => hlsdk::types::Side::Bid,
                    OrderSide::Sell => hlsdk::types::Side::Ask,
                },
                order_type: match instr.order_type {
                    OrderType::Limit | OrderType::PostOnly => hlsdk::types::OrderType::Limit,
                    OrderType::Market => hlsdk::types::OrderType::Market,
                },
                post_only: matches!(instr.order_type, OrderType::PostOnly),
            })
            .collect();

        match client.batch_modify(orders).await {
            Ok(_) => {
                tracing::info!("Sent batch of {} orders", batch.len());
            }
            Err(e) => {
                tracing::error!("batch_modify failed: {}", e);
            }
        }
    }

    Ok(())
}

