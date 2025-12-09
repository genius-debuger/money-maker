use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};
use tracing::{error, info, warn};

const WATCHDOG_TIMEOUT_SECS: u64 = 5;
const BACKOFF_INITIAL_MS: u64 = 1_000;
const BACKOFF_MAX_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Subscribing,
    Streaming,
    Backoff,
}

pub struct WebSocketManager {
    url: String,
    subscription_msg: Value,
    state: Arc<parking_lot::RwLock<ConnectionState>>,
    last_message_time: Arc<parking_lot::RwLock<Option<Instant>>>,
    backoff_ms: Arc<parking_lot::RwLock<u64>>,
    msg_tx: mpsc::UnboundedSender<Value>,
}

impl WebSocketManager {
    pub fn new(url: String, subscription_msg: Value, msg_tx: mpsc::UnboundedSender<Value>) -> Self {
        Self {
            url,
            subscription_msg,
            state: Arc::new(parking_lot::RwLock::new(ConnectionState::Disconnected)),
            last_message_time: Arc::new(parking_lot::RwLock::new(None)),
            backoff_ms: Arc::new(parking_lot::RwLock::new(BACKOFF_INITIAL_MS)),
            msg_tx,
        }
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            match *self.state.read() {
                ConnectionState::Disconnected | ConnectionState::Connecting => {
                    self.connect_and_subscribe().await?;
                }
                ConnectionState::Subscribing | ConnectionState::Streaming => {
                    // message_loop keeps running; just wait
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                ConnectionState::Backoff => {
                    let delay = *self.backoff_ms.read();
                    warn!("WebSocket backoff: {}ms", delay);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    let next = (delay.saturating_mul(2)).min(BACKOFF_MAX_MS);
                    *self.backoff_ms.write() = next;
                    *self.state.write() = ConnectionState::Connecting;
                }
            }
        }
    }

    async fn connect_and_subscribe(&self) -> Result<()> {
        *self.state.write() = ConnectionState::Connecting;
        info!("Connecting WS: {}", self.url);

        let (mut ws, _) = connect_async(&self.url)
            .await
            .context("connect_async failed")?;

        *self.backoff_ms.write() = BACKOFF_INITIAL_MS;
        *self.state.write() = ConnectionState::Subscribing;

        // Send subscription
        let sub_txt = serde_json::to_string(&self.subscription_msg)?;
        ws.send(Message::Text(sub_txt)).await.context("send subscribe")?;
        info!("Subscription sent");

        *self.last_message_time.write() = Some(Instant::now());
        *self.state.write() = ConnectionState::Streaming;

        let state = self.state.clone();
        let last = self.last_message_time.clone();
        let tx = self.msg_tx.clone();
        tokio::spawn(async move {
            WebSocketManager::message_loop(ws, state, last, tx).await;
        });

        let state_watch = self.state.clone();
        let last_watch = self.last_message_time.clone();
        tokio::spawn(async move {
            WebSocketManager::watchdog(state_watch, last_watch).await;
        });

        Ok(())
    }

    async fn message_loop(
        mut ws: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
        state: Arc<parking_lot::RwLock<ConnectionState>>,
        last_message_time: Arc<parking_lot::RwLock<Option<Instant>>>,
        tx: mpsc::UnboundedSender<Value>,
    ) {
        while let Some(msg) = ws.next().await {
            match msg {
                Ok(Message::Text(txt)) => {
                    *last_message_time.write() = Some(Instant::now());
                    match serde_json::from_str::<Value>(&txt) {
                        Ok(v) => {
                            if tx.send(v).is_err() {
                                error!("Receiver dropped; exiting message_loop");
                                break;
                            }
                        }
                        Err(e) => warn!("Bad JSON message: {}", e),
                    }
                }
                Ok(Message::Binary(bin)) => {
                    *last_message_time.write() = Some(Instant::now());
                    if let Ok(txt) = String::from_utf8(bin) {
                        if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                            if tx.send(v).is_err() {
                                error!("Receiver dropped; exiting message_loop");
                                break;
                            }
                        }
                    }
                }
                Ok(Message::Ping(data)) => {
                    if ws.send(Message::Pong(data)).await.is_err() {
                        warn!("Failed to send pong; closing");
                        break;
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("WS closed by server");
                    break;
                }
                Err(e) => {
                    error!("WS error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        *state.write() = ConnectionState::Backoff;
        warn!("WS loop ended -> Backoff");
    }

    async fn watchdog(
        state: Arc<parking_lot::RwLock<ConnectionState>>,
        last_message_time: Arc<parking_lot::RwLock<Option<Instant>>>,
    ) {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if *state.read() != ConnectionState::Streaming {
                continue;
            }
            if let Some(last) = *last_message_time.read() {
                if last.elapsed().as_secs() >= WATCHDOG_TIMEOUT_SECS {
                    error!("Watchdog: no WS msg for {}s -> Backoff", WATCHDOG_TIMEOUT_SECS);
                    *state.write() = ConnectionState::Backoff;
                }
            }
        }
    }

    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }
}

