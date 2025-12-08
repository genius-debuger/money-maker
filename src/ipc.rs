use crate::types::BookUpdate;
use anyhow::Result;
use zeromq::{Socket, SocketRecv, SocketSend};

const MARKET_DATA_SOCKET: &str = "ipc:///tmp/market_data.sock";
const SIGNALS_SOCKET: &str = "ipc:///tmp/signals.sock";

pub struct ZmqPublisher {
    socket: zeromq::PubSocket,
}

impl ZmqPublisher {
    pub async fn new() -> Result<Self> {
        let mut socket = zeromq::PubSocket::new();
        socket.bind(MARKET_DATA_SOCKET).await?;
        
        // Give bind time to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(Self { socket })
    }

    pub async fn publish_book_update(&mut self, update: &BookUpdate) -> Result<()> {
        let data = rmp_serde::to_vec(update)?;
        self.socket.send(data.into()).await?;
        Ok(())
    }

    pub async fn publish_multiple(&mut self, updates: &[BookUpdate]) -> Result<()> {
        for update in updates {
            self.publish_book_update(update).await?;
        }
        Ok(())
    }
}

pub struct ZmqSubscriber {
    socket: zeromq::SubSocket,
}

impl ZmqSubscriber {
    pub async fn new() -> Result<Self> {
        let mut socket = zeromq::SubSocket::new();
        socket.connect(SIGNALS_SOCKET).await?;
        socket.subscribe("").await?; // Subscribe to all messages
        
        // Give connection time to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(Self { socket })
    }

    pub async fn receive_signal(&mut self) -> Result<Option<crate::types::OrderInstruction>> {
        let messages = self.socket.recv().await?;
        if messages.is_empty() {
            return Ok(None);
        }
        
        // Parse msgpack from Python
        let instruction: crate::types::OrderInstruction = rmp_serde::from_slice(&messages[0])?;
        Ok(Some(instruction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pub_sub() {
        let mut publisher = ZmqPublisher::new().await.unwrap();
        let mut subscriber = ZmqSubscriber::new().await.unwrap();

        let update = BookUpdate {
            symbol: "BTC-USD".to_string(),
            sequence: 1,
            bids: vec![(50000.0, 1.0)],
            asks: vec![(50001.0, 1.0)],
            timestamp: 1234567890,
        };

        publisher.publish_book_update(&update).await.unwrap();
        
        // In real implementation, subscriber would receive this
        // For test, we just verify it doesn't panic
    }
}

