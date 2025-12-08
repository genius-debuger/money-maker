use crate::types::{BookLevel, BookUpdate, MarketData};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

pub struct LocalBook {
    symbol: String,
    bids: Arc<RwLock<BTreeMap<OrderedFloat, f64>>>,
    asks: Arc<RwLock<BTreeMap<OrderedFloat, f64>>>,
    last_sequence: Arc<RwLock<u64>>,
    depth: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderedFloat(f64);

impl LocalBook {
    pub fn new(symbol: String, depth: usize) -> Self {
        Self {
            symbol,
            bids: Arc::new(RwLock::new(BTreeMap::new())),
            asks: Arc::new(RwLock::new(BTreeMap::new())),
            last_sequence: Arc::new(RwLock::new(0)),
            depth,
        }
    }

    pub fn update(&self, update: &BookUpdate) -> Result<Option<MarketData>, BookError> {
        let mut last_seq = self.last_sequence.write();
        
        // Check for sequence gap
        if *last_seq > 0 && update.sequence != *last_seq + 1 {
            return Err(BookError::SequenceGap {
                expected: *last_seq + 1,
                received: update.sequence,
            });
        }
        
        *last_seq = update.sequence;

        // Update bids (descending order)
        {
            let mut bids = self.bids.write();
            bids.clear();
            for (price, size) in &update.bids {
                if *size > 0.0 {
                    bids.insert(OrderedFloat(*price), *size);
                }
            }
        }

        // Update asks (ascending order)
        {
            let mut asks = self.asks.write();
            asks.clear();
            for (price, size) in &update.asks {
                if *size > 0.0 {
                    asks.insert(OrderedFloat(*price), *size);
                }
            }
        }

        Ok(self.get_market_data())
    }

    pub fn snapshot(&self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, sequence: u64) {
        {
            let mut bids_map = self.bids.write();
            bids_map.clear();
            for (price, size) in bids {
                if size > 0.0 {
                    bids_map.insert(OrderedFloat(price), size);
                }
            }
        }

        {
            let mut asks_map = self.asks.write();
            asks_map.clear();
            for (price, size) in asks {
                if size > 0.0 {
                    asks_map.insert(OrderedFloat(price), size);
                }
            }
        }

        *self.last_sequence.write() = sequence;
    }

    pub fn get_market_data(&self) -> Option<MarketData> {
        let bids = self.bids.read();
        let asks = self.asks.read();

        let best_bid = bids.iter().rev().next()?; // Highest bid
        let best_ask = asks.iter().next()?; // Lowest ask

        let mid_price = (best_bid.0 .0 + best_ask.0 .0) / 2.0;
        let spread = best_ask.0 .0 - best_bid.0 .0;

        let bid_size: f64 = bids.iter().rev().take(self.depth).map(|(_, size)| size).sum();
        let ask_size: f64 = asks.iter().take(self.depth).map(|(_, size)| size).sum();

        Some(MarketData {
            symbol: self.symbol.clone(),
            mid_price,
            spread,
            bid_size,
            ask_size,
            timestamp: Instant::now(),
        })
    }

    pub fn get_levels(&self, levels: usize) -> (Vec<BookLevel>, Vec<BookLevel>) {
        let bids = self.bids.read();
        let asks = self.asks.read();

        let bid_levels: Vec<BookLevel> = bids
            .iter()
            .rev()
            .take(levels)
            .map(|(price, size)| BookLevel {
                price: price.0,
                size: *size,
            })
            .collect();

        let ask_levels: Vec<BookLevel> = asks
            .iter()
            .take(levels)
            .map(|(price, size)| BookLevel {
                price: price.0,
                size: *size,
            })
            .collect();

        (bid_levels, ask_levels)
    }

    pub fn get_last_sequence(&self) -> u64 {
        *self.last_sequence.read()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BookError {
    #[error("Sequence gap: expected {expected}, received {received}")]
    SequenceGap { expected: u64, received: u64 },
}

pub struct BookManager {
    books: Arc<DashMap<String, Arc<LocalBook>>>,
}

impl BookManager {
    pub fn new() -> Self {
        Self {
            books: Arc::new(DashMap::new()),
        }
    }

    pub fn get_or_create(&self, symbol: String, depth: usize) -> Arc<LocalBook> {
        self.books
            .entry(symbol.clone())
            .or_insert_with(|| Arc::new(LocalBook::new(symbol, depth)))
            .clone()
    }

    pub fn get(&self, symbol: &str) -> Option<Arc<LocalBook>> {
        self.books.get(symbol).map(|entry| entry.clone())
    }
}

impl Default for BookManager {
    fn default() -> Self {
        Self::new()
    }
}

