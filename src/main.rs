mod execution;
mod ipc;
mod metrics;
mod orderbook;
mod strategy;
mod types;
mod websocket;

use anyhow::Result;
use execution::{batch_processor, OrderManager};
use ipc::ZmqPublisher;
use orderbook::BookManager;
use std::sync::Arc;
use std::time::Instant;
use strategy::StrategyProcessor;
use tokio::sync::mpsc;
use tracing::{info, warn};
use websocket::WebSocketManager;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hyperliquid_hft=info,hyperliquid_rust_sdk=info".into()),
        )
        .init();

    info!("🚀 Starting Hyperliquid HFT System...");

    // Initialize components
    let book_manager = Arc::new(BookManager::new());
    let (order_manager, batch_rx) = OrderManager::new(); // Batch window from env var
    let order_manager = Arc::new(order_manager);
    
    let strategy_processor = StrategyProcessor::new(
        order_manager.clone(),
        book_manager.clone(),
        mpsc::unbounded_channel().1, // Dummy receiver (not used)
    );
    let strategy_processor = Arc::new(tokio::sync::Mutex::new(strategy_processor));

    // Spawn batch processor
    let batch_handle = tokio::spawn(async move {
        if let Err(e) = batch_processor(batch_rx).await {
            warn!("Batch processor error: {}", e);
        }
    });

    // Initialize ZeroMQ publisher for market data
    let mut zmq_publisher = ZmqPublisher::new().await?;
    info!("✅ ZeroMQ publisher initialized");

    // Spawn WebSocket reader (this would connect to Hyperliquid WS)
    let ws_handle = tokio::spawn({
        let book_manager = book_manager.clone();
        let zmq_publisher_ws = Arc::new(tokio::sync::Mutex::new(zmq_publisher));
        
        async move {
            websocket_reader(book_manager, zmq_publisher_ws).await
        }
    });

    // Spawn Python signal receiver
    let python_receiver_handle = tokio::spawn({
        let strategy = strategy_processor.clone();
        
        async move {
            python_signal_receiver(strategy).await
        }
    });

    // Start metrics server
    let metrics_state = metrics::MetricsState {
        strategy: strategy_processor.clone(),
    };
    let metrics_router = metrics::create_metrics_router(metrics_state);
    
    let metrics_server = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:9090").await?;
        info!("📊 Metrics server listening on 127.0.0.1:9090");
        axum::serve(listener, metrics_router).await?;
        Ok::<(), anyhow::Error>(())
    });

    // Wait for shutdown signal
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("🛑 Shutdown signal received");
        }
        result = ws_handle => {
            if let Err(e) = result {
                warn!("WebSocket reader error: {:?}", e);
            }
        }
        result = python_receiver_handle => {
            if let Err(e) = result {
                warn!("Python receiver error: {:?}", e);
            }
        }
    }

    Ok(())
}

async fn websocket_reader(
    book_manager: Arc<BookManager>,
    zmq_publisher: Arc<tokio::sync::Mutex<ZmqPublisher>>,
) -> Result<()> {
    // TODO: Implement actual Hyperliquid WebSocket connection
    // This is a placeholder structure
    
    info!("📡 WebSocket reader started");
    
    // Example structure:
    // 1. Connect to Hyperliquid WS endpoint
    // 2. Subscribe to L2 orderbook updates
    // 3. Parse messages into BookUpdate
    // 4. Update LocalBook
    // 5. Publish to ZeroMQ for Python
    
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        // Placeholder: In real implementation, this would process WebSocket messages
        // For now, this is the structure where WebSocket handling would go
    }
}

async fn python_signal_receiver(
    strategy: Arc<tokio::sync::Mutex<StrategyProcessor>>,
) -> Result<()> {
    use ipc::ZmqSubscriber;
    
    let mut subscriber = ZmqSubscriber::new().await?;
    info!("📥 Python signal receiver started");
    
    loop {
        match subscriber.receive_signal().await {
            Ok(Some(instruction)) => {
                let tick_start = Instant::now();
                
                // Process signal directly through strategy processor
                {
                    let mut proc = strategy.lock().await;
                    if let Err(e) = proc.process_signal(instruction.clone()).await {
                        warn!("Failed to process signal: {}", e);
                        continue;
                    }
                }
                
                // Update latency metrics
                let tick_to_trade = tick_start.elapsed();
                {
                    let mut proc = strategy.lock().await;
                    proc.update_latency(tick_to_trade, std::time::Duration::from_millis(0));
                }
            }
            Ok(None) => continue,
            Err(e) => {
                warn!("Error receiving signal: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

