use crate::types::{BookLevel, MarketData};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

// Double-buffering is implemented directly in DoubleBufferedBook using arrays

/// Order book level (price, size)
#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub price: f64,
    pub size: f64,
}

impl Level {
    pub fn new(price: f64, size: f64) -> Self {
        Self { price, size }
    }
}

/// Lock-free double-buffered order book for snapshot-only feeds.
pub struct DoubleBufferedBook {
    symbol: String,
    bids: [Vec<Level>; 2],
    asks: [Vec<Level>; 2],
    read_index: AtomicUsize,
    last_sequence: AtomicU64,
    last_update_time: AtomicU64,
    depth: usize,
}

impl DoubleBufferedBook {
    pub fn new(symbol: String, depth: usize) -> Self {
        Self {
            symbol,
            bids: [
                Vec::with_capacity(depth),
                Vec::with_capacity(depth),
            ],
            asks: [
                Vec::with_capacity(depth),
                Vec::with_capacity(depth),
            ],
            read_index: AtomicUsize::new(0),
            last_sequence: std::sync::atomic::AtomicU64::new(0),
            last_update_time: std::sync::atomic::AtomicU64::new(0),
            depth,
        }
    }

    /// Apply full snapshot without allocating.
    pub fn apply_snapshot(&self, bids: &[(f64, f64)], asks: &[(f64, f64)], sequence: u64, timestamp: u64) {
        let write_idx = 1 - self.read_index.load(Ordering::Acquire);

        let write_bids = unsafe { &mut *(&self.bids[write_idx] as *const _ as *mut Vec<Level>) };
        write_bids.clear();
        write_bids.extend(
            bids.iter()
                .take(self.depth)
                .filter(|(_, sz)| *sz > 0.0)
                .map(|(p, sz)| Level::new(*p, *sz)),
        );
        write_bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));

        let write_asks = unsafe { &mut *(&self.asks[write_idx] as *const _ as *mut Vec<Level>) };
        write_asks.clear();
        write_asks.extend(
            asks.iter()
                .take(self.depth)
                .filter(|(_, sz)| *sz > 0.0)
                .map(|(p, sz)| Level::new(*p, *sz)),
        );
        write_asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));

        self.read_index.store(write_idx, Ordering::Release);
        self.last_sequence.store(sequence, Ordering::Release);
        self.last_update_time.store(timestamp, Ordering::Release);
    }

    /// Get current market data (reads from active buffer)
    pub fn get_market_data(&self) -> Option<MarketData> {
        let read_idx = self.read_index.load(Ordering::Acquire);
        let bids = &self.bids[read_idx];
        let asks = &self.asks[read_idx];

        let best_bid = bids.first()?; // Highest bid (first after sort)
        let best_ask = asks.first()?; // Lowest ask (first after sort)

        let mid_price = (best_bid.price + best_ask.price) / 2.0;
        let spread = best_ask.price - best_bid.price;

        let bid_size: f64 = bids.iter().take(self.depth).map(|l| l.size).sum();
        let ask_size: f64 = asks.iter().take(self.depth).map(|l| l.size).sum();

        Some(MarketData {
            symbol: self.symbol.clone(),
            mid_price,
            spread,
            bid_size,
            ask_size,
            timestamp: Instant::now(),
        })
    }

    /// Get top N levels for bids and asks
    pub fn get_levels(&self, levels: usize) -> (Vec<BookLevel>, Vec<BookLevel>) {
        let read_idx = self.read_index.load(Ordering::Acquire);
        let bids = &self.bids[read_idx];
        let asks = &self.asks[read_idx];

        let bid_levels: Vec<BookLevel> = bids
            .iter()
            .take(levels)
            .map(|l| BookLevel {
                price: l.price,
                size: l.size,
            })
            .collect();

        let ask_levels: Vec<BookLevel> = asks
            .iter()
            .take(levels)
            .map(|l| BookLevel {
                price: l.price,
                size: l.size,
            })
            .collect();

        (bid_levels, ask_levels)
    }

    pub fn get_last_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::Acquire)
    }
}


#[derive(Clone)]
pub struct SafeDoubleBufferedBook {
    inner: Arc<DoubleBufferedBook>,
}

impl SafeDoubleBufferedBook {
    pub fn new(symbol: String, depth: usize) -> Self {
        Self {
            inner: Arc::new(DoubleBufferedBook::new(symbol, depth)),
        }
    }

    pub fn apply_snapshot(&self, bids: &[(f64, f64)], asks: &[(f64, f64)], sequence: u64, timestamp: u64) {
        self.inner.apply_snapshot(bids, asks, sequence, timestamp);
    }

    pub fn market_data(&self) -> Option<MarketData> {
        self.inner.get_market_data()
    }

    pub fn get_levels(&self, levels: usize) -> (Vec<BookLevel>, Vec<BookLevel>) {
        self.inner.get_levels(levels)
    }

    pub fn get_last_sequence(&self) -> u64 {
        self.inner.get_last_sequence()
    }
}

/// Legacy LocalBook for compatibility (deprecated, use DoubleBufferedBook)
use parking_lot::RwLock;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderedFloat(f64);

pub struct LocalBook {
    symbol: String,
    bids: Arc<RwLock<BTreeMap<OrderedFloat, f64>>>,
    asks: Arc<RwLock<BTreeMap<OrderedFloat, f64>>>,
    last_sequence: Arc<RwLock<u64>>,
    depth: usize,
}

impl LocalBook {
    pub fn new(symbol: String, depth: usize) -> Self {
        Self {
            symbol,
            bids: Arc::new(RwLock::new(BTreeMap::new()),
            asks: Arc::new(RwLock::new(BTreeMap::new())),
            last_sequence: Arc::new(RwLock::new(0)),
            depth,
        }
    }

    pub fn update(&self, update: &BookUpdate) -> Result<Option<MarketData>, BookError> {
        let mut last_seq = self.last_sequence.write();
        
        if *last_seq > 0 && update.sequence != *last_seq + 1 {
            return Err(BookError::SequenceGap {
                expected: *last_seq + 1,
                received: update.sequence,
            });
        }
        
        *last_seq = update.sequence;

        {
            let mut bids = self.bids.write();
            bids.clear();
            for (price, size) in &update.bids {
                if *size > 0.0 {
                    bids.insert(OrderedFloat(*price), *size);
                }
            }
        }

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

        let best_bid = bids.iter().rev().next()?;
        let best_ask = asks.iter().next()?;

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
    books: Arc<DashMap<String, Arc<SafeDoubleBufferedBook>>>,
}

impl BookManager {
    pub fn new() -> Self {
        Self {
            books: Arc::new(DashMap::new()),
        }
    }

    pub fn get_or_create(&self, symbol: String, depth: usize) -> Arc<SafeDoubleBufferedBook> {
        self.books
            .entry(symbol.clone())
            .or_insert_with(|| Arc::new(SafeDoubleBufferedBook::new(symbol, depth)))
            .clone()
    }

    pub fn get(&self, symbol: &str) -> Option<Arc<SafeDoubleBufferedBook>> {
        self.books.get(symbol).map(|entry| entry.clone())
    }
}

impl Default for BookManager {
    fn default() -> Self {
        Self::new()
    }
}
