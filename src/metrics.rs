use crate::strategy::StrategyProcessor;
use axum::{
    extract::State,
    http::{Response, StatusCode},
    response::Html,
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct MetricsState {
    pub strategy: Arc<Mutex<StrategyProcessor>>,
}

pub fn create_metrics_router(state: MetricsState) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

async fn metrics_handler(State(state): State<MetricsState>) -> Response<axum::body::Body> {
    let strategy = state.strategy.lock().await;
    
    let pnl = strategy.get_pnl_state();
    let latency = strategy.get_latency_metrics();
    let circuit_breaker = if strategy.is_circuit_breaker_active() { 1 } else { 0 };

    // Format as Prometheus metrics
    let metrics = format!(
        r#"# HELP hyperliquid_hft_pnl_realized Realized PnL in USD
# TYPE hyperliquid_hft_pnl_realized gauge
hyperliquid_hft_pnl_realized {}

# HELP hyperliquid_hft_pnl_unrealized Unrealized PnL in USD
# TYPE hyperliquid_hft_pnl_unrealized gauge
hyperliquid_hft_pnl_unrealized {}

# HELP hyperliquid_hft_pnl_total Total PnL (realized + unrealized) in USD
# TYPE hyperliquid_hft_pnl_total gauge
hyperliquid_hft_pnl_total {}

# HELP hyperliquid_hft_maker_rebates Maker rebates collected in USD
# TYPE hyperliquid_hft_maker_rebates counter
hyperliquid_hft_maker_rebates {}

# HELP hyperliquid_hft_inventory Current inventory position
# TYPE hyperliquid_hft_inventory gauge
hyperliquid_hft_inventory {}

# HELP hyperliquid_hft_inventory_skew Inventory skew as percentage (-100 to 100)
# TYPE hyperliquid_hft_inventory_skew gauge
hyperliquid_hft_inventory_skew {}

# HELP hyperliquid_hft_drawdown_pct Current drawdown percentage
# TYPE hyperliquid_hft_drawdown_pct gauge
hyperliquid_hft_drawdown_pct {}

# HELP hyperliquid_hft_peak_equity Peak equity value in USD
# TYPE hyperliquid_hft_peak_equity gauge
hyperliquid_hft_peak_equity {}

# HELP hyperliquid_hft_latency_tick_to_trade_ms Tick-to-trade latency in milliseconds
# TYPE hyperliquid_hft_latency_tick_to_trade_ms histogram
hyperliquid_hft_latency_tick_to_trade_ms_bucket{{le="10"}} {}
hyperliquid_hft_latency_tick_to_trade_ms_bucket{{le="50"}} {}
hyperliquid_hft_latency_tick_to_trade_ms_bucket{{le="100"}} {}
hyperliquid_hft_latency_tick_to_trade_ms_bucket{{le="150"}} {}
hyperliquid_hft_latency_tick_to_trade_ms_bucket{{le="200"}} {}
hyperliquid_hft_latency_tick_to_trade_ms_bucket{{le="+Inf"}} {}
hyperliquid_hft_latency_tick_to_trade_ms_sum {}
hyperliquid_hft_latency_tick_to_trade_ms_count 1

# HELP hyperliquid_hft_latency_book_update_ms Book update latency in milliseconds
# TYPE hyperliquid_hft_latency_book_update_ms gauge
hyperliquid_hft_latency_book_update_ms {}

# HELP hyperliquid_hft_latency_signal_processing_ms Signal processing latency in milliseconds
# TYPE hyperliquid_hft_latency_signal_processing_ms gauge
hyperliquid_hft_latency_signal_processing_ms {}

# HELP hyperliquid_hft_circuit_breaker_active Circuit breaker status (1=active, 0=inactive)
# TYPE hyperliquid_hft_circuit_breaker_active gauge
hyperliquid_hft_circuit_breaker_active {}
"#,
        pnl.realized_pnl,
        pnl.unrealized_pnl,
        pnl.realized_pnl + pnl.unrealized_pnl,
        pnl.maker_rebates,
        pnl.inventory,
        // Inventory skew as percentage (assuming max 10.0)
        (pnl.inventory / 10.0) * 100.0,
        pnl.drawdown,
        pnl.peak_equity,
        // Latency buckets (simple implementation)
        if latency.tick_to_trade.as_millis() <= 10 { 1 } else { 0 },
        if latency.tick_to_trade.as_millis() <= 50 { 1 } else { 0 },
        if latency.tick_to_trade.as_millis() <= 100 { 1 } else { 0 },
        if latency.tick_to_trade.as_millis() <= 150 { 1 } else { 0 },
        if latency.tick_to_trade.as_millis() <= 200 { 1 } else { 0 },
        1,
        latency.tick_to_trade.as_millis() as f64,
        latency.book_update_latency.as_millis(),
        latency.signal_processing.as_millis(),
        circuit_breaker,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(axum::body::Body::from(metrics))
        .unwrap()
}

async fn health_handler() -> (StatusCode, Html<&'static str>) {
    (StatusCode::OK, Html("<html><body>OK</body></html>"))
}

