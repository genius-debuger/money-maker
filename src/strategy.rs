use crate::execution::OrderManager;
use crate::orderbook::{BookManager, BookError};
use crate::types::{MarketData, OrderInstruction, OrderType, PnLState, LatencyMetrics};
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const CIRCUIT_BREAKER_LATENCY_MS: u64 = 150;
const CIRCUIT_BREAKER_DRAWDOWN_PCT: f64 = 3.0;
const FUNDING_THRESHOLD: f64 = 0.00002; // 0.002% hourly

pub struct StrategyProcessor {
    order_manager: Arc<OrderManager>,
    book_manager: Arc<BookManager>,
    pnl_state: Arc<RwLock<PnLState>>,
    latency_metrics: Arc<RwLock<LatencyMetrics>>,
    funding_rates: Arc<RwLock<HashMap<String, f64>>>,
    signal_rx: mpsc::UnboundedReceiver<OrderInstruction>,
    circuit_breaker_active: Arc<RwLock<bool>>,
}

impl StrategyProcessor {
    pub fn new(
        order_manager: Arc<OrderManager>,
        book_manager: Arc<BookManager>,
        signal_rx: mpsc::UnboundedReceiver<OrderInstruction>,
    ) -> Self {
        Self {
            order_manager,
            book_manager,
            pnl_state: Arc::new(RwLock::new(PnLState {
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                maker_rebates: 0.0,
                inventory: 0.0,
                drawdown: 0.0,
                peak_equity: 100000.0, // Initial equity
                last_reset: Instant::now(),
            })),
            latency_metrics: Arc::new(RwLock::new(LatencyMetrics {
                tick_to_trade: Duration::from_millis(0),
                book_update_latency: Duration::from_millis(0),
                signal_processing: Duration::from_millis(0),
            })),
            funding_rates: Arc::new(RwLock::new(HashMap::new())),
            signal_rx,
            circuit_breaker_active: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn process_signal(&mut self, instruction: OrderInstruction) -> Result<()> {
        let start = Instant::now();

        // Check circuit breaker FIRST (before any AI logic)
        if *self.circuit_breaker_active.read() {
            tracing::warn!("Circuit breaker active, rejecting signal");
            return Ok(());
        }

        // Check latency circuit breaker
        {
            let latency = self.latency_metrics.read();
            if latency.tick_to_trade.as_millis() > CIRCUIT_BREAKER_LATENCY_MS as u128 {
                tracing::error!("Latency circuit breaker triggered: {:?}", latency.tick_to_trade);
                self.trigger_circuit_breaker().await?;
                return Ok(());
            }
        }

        // Check drawdown circuit breaker
        {
            let pnl = self.pnl_state.read();
            if pnl.drawdown > CIRCUIT_BREAKER_DRAWDOWN_PCT {
                tracing::error!("Drawdown circuit breaker triggered: {:.2}%", pnl.drawdown);
                self.trigger_panic_close().await?;
                return Ok(());
            }
        }

        // Apply funding arbitrage bias
        let mut adjusted_instruction = instruction.clone();
        if let Some(funding) = self.funding_rates.read().get(&instruction.symbol) {
            if *funding > FUNDING_THRESHOLD {
                // Bias to short when funding is high
                if matches!(adjusted_instruction.side, crate::types::OrderSide::Buy) {
                    // Reduce buy size or convert to sell
                    adjusted_instruction.size *= 0.5;
                } else {
                    // Increase sell size
                    adjusted_instruction.size *= 1.2;
                }
            }
        }

        // Ensure Post-Only for rebate capture
        adjusted_instruction.order_type = OrderType::PostOnly;

        // Submit order
        let order_id = self.order_manager.submit_order(adjusted_instruction)?;

        // Update latency metrics
        let processing_time = start.elapsed();
        {
            let mut latency = self.latency_metrics.write();
            latency.signal_processing = processing_time;
        }

        tracing::debug!("Processed signal: order_id={}, latency={:?}", order_id, processing_time);

        Ok(())
    }

    pub fn update_funding_rate(&self, symbol: String, funding: f64) {
        self.funding_rates.write().insert(symbol, funding);
    }

    pub fn update_pnl(&self, realized: f64, unrealized: f64, rebates: f64, inventory: f64) {
        let mut pnl = self.pnl_state.write();
        pnl.realized_pnl += realized;
        pnl.unrealized_pnl = unrealized;
        pnl.maker_rebates += rebates;
        pnl.inventory = inventory;

        let current_equity = pnl.realized_pnl + pnl.unrealized_pnl + pnl.maker_rebates + 100000.0; // Base equity
        if current_equity > pnl.peak_equity {
            pnl.peak_equity = current_equity;
        }

        pnl.drawdown = if pnl.peak_equity > 0.0 {
            ((pnl.peak_equity - current_equity) / pnl.peak_equity) * 100.0
        } else {
            0.0
        };
    }

    pub fn update_latency(&self, tick_to_trade: Duration, book_update: Duration) {
        let mut latency = self.latency_metrics.write();
        latency.tick_to_trade = tick_to_trade;
        latency.book_update_latency = book_update;
    }

    async fn trigger_circuit_breaker(&self) -> Result<()> {
        *self.circuit_breaker_active.write() = true;
        tracing::error!("🚨 CIRCUIT BREAKER ACTIVATED - Cancelling all orders");
        
        let cancelled = self.order_manager.cancel_all()?;
        tracing::info!("Cancelled {} orders", cancelled);
        
        Ok(())
    }

    async fn trigger_panic_close(&self) -> Result<()> {
        tracing::error!("🚨 PANIC CLOSE TRIGGERED - Market selling all positions");
        
        // Cancel all orders first
        self.order_manager.cancel_all()?;
        
        // TODO: Execute market orders to close positions
        // This would need to query current positions and market sell them
        
        *self.circuit_breaker_active.write() = true;
        
        Ok(())
    }

    pub fn is_circuit_breaker_active(&self) -> bool {
        *self.circuit_breaker_active.read()
    }

    pub fn reset_circuit_breaker(&self) {
        *self.circuit_breaker_active.write() = false;
        tracing::info!("Circuit breaker reset");
    }

    pub fn get_pnl_state(&self) -> PnLState {
        self.pnl_state.read().clone()
    }

    pub fn get_latency_metrics(&self) -> LatencyMetrics {
        self.latency_metrics.read().clone()
    }
}

pub async fn strategy_loop(mut processor: StrategyProcessor) -> Result<()> {
    use tokio_stream::wrappers::UnboundedReceiverStream;
    use futures::StreamExt;

    let mut stream = UnboundedReceiverStream::new(processor.signal_rx);

    while let Some(instruction) = stream.next().await {
        if let Err(e) = processor.process_signal(instruction).await {
            tracing::error!("Error processing signal: {}", e);
        }
    }

    Ok(())
}

