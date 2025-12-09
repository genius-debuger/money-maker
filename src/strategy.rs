use crate::execution::OrderManager;
use crate::orderbook::{BookManager, BookError};
use crate::types::{MarketData, OrderInstruction, OrderType, PnLState, LatencyMetrics};
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// --- Risk Engine & Guardrail Constants ---
const CIRCUIT_BREAKER_LATENCY_MS: u64 = 150;
const FUNDING_THRESHOLD: f64 = 0.00002; // 0.002% hourly

// Task 1: Proactive "Fade-Out" Circuit Breaker Thresholds
const DRAWDOWN_TIER1_PCT: f64 = 1.5; // Inventory skew control
const DRAWDOWN_TIER2_PCT: f64 = 2.0; // Reduce-only mode
const DRAWDOWN_TIER3_PCT: f64 = 3.0; // Panic close

// Task 2: Anti-Toxic Flow Guardrails
const MIN_SPREAD_BPS: f64 = 2.0;
const DYNAMIC_DELAY_LATENCY_MS: u128 = 100;

// Task 3: Rebate Harvesting Size Bias
const AGGRESSIVE_SIZE_MULTIPLIER: f64 = 1.1; // Increase size by 10% for aggressive signals
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

        // 1. Check master circuit breaker
        if *self.circuit_breaker_active.read() {
            tracing::warn!("Circuit breaker active, rejecting signal");
            return Ok(());
        }

        // 2. Latency Guardrails (Task 2 Dynamic Delay & existing latency breaker)
        {
            let latency = self.latency_metrics.read();
            // Task 2: Dynamic Delay on stale book. Prevents trading on an old market view.
            if latency.book_update_latency.as_millis() > DYNAMIC_DELAY_LATENCY_MS {
                tracing::warn!(
                    "Stale order book ({}ms > {}ms), skipping signal.",
                    latency.book_update_latency.as_millis(), DYNAMIC_DELAY_LATENCY_MS
                );
                return Ok(());
            }
            // Existing latency breaker for tick-to-trade
            if latency.tick_to_trade.as_millis() > CIRCUIT_BREAKER_LATENCY_MS as u128 {
                tracing::error!("Latency circuit breaker triggered: {:?}", latency.tick_to_trade);
                self.trigger_circuit_breaker().await?; // This cancels all open orders
                return Ok(());
            }
        }

        let mut adjusted_instruction = instruction.clone();

        // 3. Task 1: Proactive "Fade-Out" Tiered Drawdown Engine
        {
            let pnl = self.pnl_state.read();
            if pnl.drawdown > DRAWDOWN_TIER3_PCT {
                tracing::error!("Drawdown Tier 3 (> {:.2}%) triggered: PANIC CLOSE", DRAWDOWN_TIER3_PCT);
                self.trigger_panic_close().await?;
                return Ok(());
            } else if pnl.drawdown > DRAWDOWN_TIER2_PCT {
                tracing::warn!("Drawdown Tier 2 (> {:.2}%) triggered: Forcing Reduce-Only.", DRAWDOWN_TIER2_PCT);
                adjusted_instruction.reduce_only = true;
            } else if pnl.drawdown > DRAWDOWN_TIER1_PCT {
                // Stop quoting on the losing side to control inventory skew.
                if pnl.inventory > f64::EPSILON && matches!(instruction.side, crate::types::OrderSide::Buy) {
                    tracing::warn!("Drawdown Tier 1 (> {:.2}%): Rejecting BUY signal due to positive inventory.", DRAWDOWN_TIER1_PCT);
                    return Ok(());
                }
                if pnl.inventory < -f64::EPSILON && matches!(instruction.side, crate::types::OrderSide::Sell) {
                    tracing::warn!("Drawdown Tier 1 (> {:.2}%): Rejecting SELL signal due to negative inventory.", DRAWDOWN_TIER1_PCT);
                    return Ok(());
                }
            }
        }

        // NOTE: The following logic assumes `book_manager` has a method `get_best_bid_ask`
        // which returns `Result<Option<(best_bid, best_ask)>, BookError>`. This is a standard
        // feature for an order book and necessary for the requested guardrails.
        if let Ok(Some((best_bid, best_ask))) = self.book_manager.get_best_bid_ask(&instruction.symbol) {
            // Ensure book is valid before proceeding
            if best_bid > 0.0 && best_ask > best_bid {
                let mid_price = (best_bid + best_ask) / 2.0;

                // 4. Task 2: Anti-Toxic Flow Guardrail (Gamma Clamping)
                let required_spread = mid_price * (MIN_SPREAD_BPS / 10000.0);
                match adjusted_instruction.side {
                    crate::types::OrderSide::Buy => {
                        if (best_ask - adjusted_instruction.price) < required_spread {
                            let old_price = adjusted_instruction.price;
                            adjusted_instruction.price = best_ask - required_spread;
                            tracing::warn!(
                                "Gamma Clamp: Buy price {} too aggressive. Overriding to {} to meet min spread of {} bps.",
                                old_price, adjusted_instruction.price, MIN_SPREAD_BPS
                            );
                        }
                    },
                    crate::types::OrderSide::Sell => {
                        if (adjusted_instruction.price - best_bid) < required_spread {
                            let old_price = adjusted_instruction.price;
                            adjusted_instruction.price = best_bid + required_spread;
                            tracing::warn!(
                                "Gamma Clamp: Sell price {} too aggressive. Overriding to {} to meet min spread of {} bps.",
                                old_price, adjusted_instruction.price, MIN_SPREAD_BPS
                            );
                        }
                    }
                }

                // 5. Task 3: Rebate Harvesting Bias (Increase Size on Strong Signals)
                // A "strong" signal is inferred if the original AI price was very aggressive.
                // Instead of taking, we increase our passive size.
                let is_aggressive = match instruction.side {
                    crate::types::OrderSide::Buy => (best_ask - instruction.price) < (mid_price * 0.0001), // e.g., within 1 bps
                    crate::types::OrderSide::Sell => (instruction.price - best_bid) < (mid_price * 0.0001),
                };
                if is_aggressive {
                    tracing::info!("Aggressive signal detected. Increasing size by {}% for rebate harvesting.", (AGGRESSIVE_SIZE_MULTIPLIER - 1.0) * 100.0);
                    adjusted_instruction.size *= AGGRESSIVE_SIZE_MULTIPLIER;
                }
            }
        } else {
            tracing::error!("Could not get valid book for {}. Skipping signal for safety.", instruction.symbol);
            return Ok(());
        }

        // 6. Apply funding arbitrage bias
        if let Some(funding) = self.funding_rates.read().get(&instruction.symbol) {
            if *funding > FUNDING_THRESHOLD {
                if matches!(adjusted_instruction.side, crate::types::OrderSide::Buy) {
                    adjusted_instruction.size *= 0.5;
                } else {
                    adjusted_instruction.size *= 1.2;
                }
            }
        }

        // 7. Task 3: Strictly enforce Post-Only to guarantee maker rebates.
        adjusted_instruction.order_type = OrderType::PostOnly;

        // 8. Submit final order
        let order_id = self.order_manager.submit_order(adjusted_instruction)?;

        // 9. Update latency metrics
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
