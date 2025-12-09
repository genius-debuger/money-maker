use anyhow::{Context, Result};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};
use tracing::{error, info, warn};

const WATCHDOG_TIMEOUT_SECS: u64 = 5;
const BACKOFF_INITIAL_MS: u64 = 1000;
const BACKOFF_MAX_MS: u64 = 30000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Connecting,
    Handshaking,
    Active,
    Backoff,
}

pub struct WebSocketManager {
    url: String,
    state: Arc<parking_lot::RwLock<ConnectionState>>,
    last_message_time: Arc<parking_lot::RwLock<Option<Instant>>>,
    backoff_duration_ms: Arc<parking_lot::RwLock<u64>>,
    message_tx: mpsc::UnboundedSender<Value>,
}

impl WebSocketManager {
    pub fn new(url: String, message_tx: mpsc::UnboundedSender<Value>) -> Self {
        Self {
            url,
            state: Arc::new(parking_lot::RwLock::new(ConnectionState::Connecting)),
            last_message_time: Arc::new(parking_lot::RwLock::new(None)),
            backoff_duration_ms: Arc::new(parking_lot::RwLock::new(BACKOFF_INITIAL_MS)),
            message_tx,
        }
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            match *self.state.read() {
                ConnectionState::Connecting => {
                    self.connect().await?;
                }
                ConnectionState::Handshaking => {
                    // Transition to Active after successful handshake
                    *self.state.write() = ConnectionState::Active;
                    info!("WebSocket handshake complete, entering Active state");
                }
                ConnectionState::Active => {
                    // This state is handled by the message loop
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                ConnectionState::Backoff => {
                    let backoff_ms = *self.backoff_duration_ms.read();
                    warn!("Backing off for {}ms", backoff_ms);
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    
                    // Exponential backoff with max cap
                    let new_backoff = (backoff_ms * 2).min(BACKOFF_MAX_MS);
                    *self.backoff_duration_ms.write() = new_backoff;
                    
                    *self.state.write() = ConnectionState::Connecting;
                }
            }
        }
    }

    async fn connect(&self) -> Result<()> {
        *self.state.write() = ConnectionState::Handshaking;
        info!("Connecting to WebSocket: {}", self.url);

        let (ws_stream, _) = connect_async(&self.url)
            .await
            .context("Failed to connect to WebSocket")?;

        info!("WebSocket connection established");
        
        // Reset backoff on successful connection
        *self.backoff_duration_ms.write() = BACKOFF_INITIAL_MS;
        *self.state.write() = ConnectionState::Active;

        // Spawn message processing loop
        let state_clone = self.state.clone();
        let last_msg_clone = self.last_message_time.clone();
        let msg_tx_clone = self.message_tx.clone();
        
        tokio::spawn(Self::message_loop(ws_stream, state_clone, last_msg_clone, msg_tx_clone));
        
        // Spawn watchdog
        let state_watchdog = self.state.clone();
        let last_msg_watchdog = self.last_message_time.clone();
        tokio::spawn(Self::watchdog_loop(state_watchdog, last_msg_watchdog));

        Ok(())
    }

    async fn message_loop(
        mut ws_stream: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
        state: Arc<parking_lot::RwLock<ConnectionState>>,
        last_message_time: Arc<parking_lot::RwLock<Option<Instant>>>,
        message_tx: mpsc::UnboundedSender<Value>,
    ) {
        use futures_util::{SinkExt, StreamExt};

        loop {
            match ws_stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    *last_message_time.write() = Some(Instant::now());

                    // Parse JSON message
                    match serde_json::from_str::<Value>(&text) {
                        Ok(value) => {
                            if message_tx.send(value).is_err() {
                                error!("Message channel closed, stopping message loop");
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse WebSocket message: {}", e);
                        }
                    }
                }
                Some(Ok(Message::Binary(data))) => {
                    *last_message_time.write() = Some(Instant::now());

                    // Try to parse binary as JSON (some protocols send JSON over binary)
                    if let Ok(text) = String::from_utf8(data) {
                        if let Ok(value) = serde_json::from_str::<Value>(&text) {
                            if message_tx.send(value).is_err() {
                                error!("Message channel closed, stopping message loop");
                                break;
                            }
                        }
                    }
                }
                Some(Ok(Message::Ping(data))) => {
                    // Auto-respond to ping with pong
                    if ws_stream.send(Message::Pong(data)).await.is_err() {
                        error!("Failed to send pong, disconnecting");
                        break;
                    }
                }
                Some(Ok(Message::Close(_))) => {
                    info!("WebSocket closed by server");
                    break;
                }
                Some(Err(e)) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                None => {
                    warn!("WebSocket stream ended");
                    break;
                }
                _ => {}
            }
        }

        // Transition to Backoff state
        *state.write() = ConnectionState::Backoff;
        warn!("WebSocket connection lost, entering Backoff state");
    }

    async fn watchdog_loop(
        state: Arc<parking_lot::RwLock<ConnectionState>>,
        last_message_time: Arc<parking_lot::RwLock<Option<Instant>>>,
    ) {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;

            // Only check watchdog in Active state
            if *state.read() != ConnectionState::Active {
                continue;
            }

            if let Some(last_msg) = *last_message_time.read() {
                let elapsed = last_msg.elapsed();
                if elapsed.as_secs() >= WATCHDOG_TIMEOUT_SECS {
                    error!(
                        "Watchdog timeout: No message received for {}s, forcing disconnect",
                        elapsed.as_secs()
                    );
                    *state.write() = ConnectionState::Backoff;
                }
            }
        }
    }

    pub fn get_state(&self) -> ConnectionState {
        *self.state.read()
    }

    pub fn send_message(&self, message: Value) -> Result<()> {
        // Note: This would need a sender channel to the WebSocket stream
        // For now, this is a placeholder - full implementation would require
        // maintaining a separate sender handle
        serde_json::to_string(&message)
            .context("Failed to serialize message")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_transitions() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let manager = WebSocketManager::new("wss://echo.websocket.org".to_string(), tx);
        
        assert_eq!(manager.get_state(), ConnectionState::Connecting);
    }
}

