"""
Hyperliquid HFT Alpha Engine
Main inference loop using uvloop for high performance
Reads market data from ZeroMQ, runs LiT inference, sends signals to Rust
"""

import asyncio
import uvloop
import zmq
import msgpack
import numpy as np
import torch
from typing import Optional, Dict
import time
from datetime import datetime
import signal
import sys

from model import LiTTransformer, create_lit_model
from strategy import (
    AvellanedaStoikovStrategy,
    StrategyState,
    InventoryManager,
    OrderQuote,
)

# Configuration
MARKET_DATA_SOCKET = "ipc:///tmp/market_data.sock"
SIGNALS_SOCKET = "ipc:///tmp/signals.sock"
SEQUENCE_LENGTH = 100
NUM_LEVELS = 20  # Top 20 levels of order book
INFERENCE_BATCH_SIZE = 1

# Global state
order_book_buffer: Dict[str, np.ndarray] = {}  # symbol -> (seq_len, 40, 2)
current_prices: Dict[str, float] = {}
inventory_manager = InventoryManager(max_inventory=10.0)
strategy = AvellanedaStoikovStrategy(use_ppo=True)
model: Optional[LiTTransformer] = None


def initialize_model(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> LiTTransformer:
    """Initialize and load LiT model"""
    global model
    
    print(f"🚀 Initializing LiT model on {device}...")
    model = create_lit_model(
        seq_len=SEQUENCE_LENGTH,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        lstm_hidden=256,
        dropout=0.1,
    )
    model = model.to(device)
    model.eval()
    
    # In production, load trained weights:
    # model.load_state_dict(torch.load('lit_model.pt', map_location=device))
    
    print("✅ Model initialized")
    return model


def parse_book_update(data: bytes) -> Optional[Dict]:
    """Parse book update from msgpack (rmp-serde format from Rust)"""
    try:
        # Rust's rmp-serde uses msgpack format
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as e:
        print(f"Error parsing book update: {e}")
        return None


def update_order_book_buffer(symbol: str, bids: list, asks: list, timestamp: int):
    """Update order book buffer with new data"""
    global order_book_buffer, current_prices
    
    if symbol not in order_book_buffer:
        # Initialize buffer: (seq_len, 40, 2)
        # 40 = 20 bid levels + 20 ask levels, 2 = (price, volume)
        order_book_buffer[symbol] = np.zeros((SEQUENCE_LENGTH, 40, 2), dtype=np.float32)
    
    buffer = order_book_buffer[symbol]
    
    # Format: First 20 levels are bids (price descending), next 20 are asks (price ascending)
    formatted = np.zeros((40, 2), dtype=np.float32)
    
    # Fill bids (top 20)
    for i, (price, size) in enumerate(bids[:20]):
        formatted[i, 0] = float(price)
        formatted[i, 1] = float(size)
    
    # Fill asks (top 20)
    for i, (price, size) in enumerate(asks[:20]):
        formatted[20 + i, 0] = float(price)
        formatted[20 + i, 1] = float(size)
    
    # Shift buffer and add new data
    buffer[:-1] = buffer[1:]
    buffer[-1] = formatted
    
    # Update current price (mid price)
    if len(bids) > 0 and len(asks) > 0:
        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        current_prices[symbol] = mid_price


def prepare_model_input(symbol: str) -> Optional[torch.Tensor]:
    """Prepare order book buffer for model inference"""
    if symbol not in order_book_buffer:
        return None
    
    buffer = order_book_buffer[symbol]
    
    # Check if buffer is sufficiently filled
    if np.allclose(buffer[0], 0):  # First timestep is still zero
        return None
    
    # Convert to tensor: (1, seq_len, 40, 2)
    tensor = torch.from_numpy(buffer).unsqueeze(0).float()
    
    return tensor


async def run_inference(symbol: str, device: str) -> Optional[Dict]:
    """Run LiT inference and generate trading signals"""
    global model, strategy, inventory_manager, current_prices
    
    if model is None:
        return None
    
    # Prepare input
    model_input = prepare_model_input(symbol)
    if model_input is None:
        return None
    
    model_input = model_input.to(device)
    
    # Run inference
    inference_start = time.time()
    with torch.inference_mode():
        probs = model(model_input)  # (1, 3)
    
    inference_time = (time.time() - inference_start) * 1000  # ms
    
    # Extract probabilities
    probs_np = probs.cpu().numpy()[0]
    p_up, p_down, p_stationary = probs_np[0], probs_np[1], probs_np[2]
    
    # Compute prediction and confidence
    lit_prediction = p_up - p_down  # Range: [-1, 1]
    confidence = max(p_up, p_down, p_stationary)
    
    # Get current market state
    mid_price = current_prices.get(symbol, 0.0)
    if mid_price == 0.0:
        return None
    
    # Create strategy state
    # Note: In production, these would come from Rust metrics
    strategy_state = StrategyState(
        inventory=inventory_manager.current_inventory,
        mid_price=mid_price,
        volatility=0.02,  # TODO: Get from historical data
        time_to_horizon=1.0,  # 1 hour horizon
        lit_prediction=lit_prediction,
        confidence=confidence,
    )
    
    # Compute quotes
    quotes = strategy.compute_quotes(strategy_state)
    
    return {
        "symbol": symbol,
        "lit_prediction": float(lit_prediction),
        "confidence": float(confidence),
        "probabilities": {
            "up": float(p_up),
            "down": float(p_down),
            "stationary": float(p_stationary),
        },
        "quotes": {
            "bid_price": float(quotes.bid_price),
            "ask_price": float(quotes.ask_price),
            "bid_size": float(quotes.bid_size),
            "ask_size": float(quotes.ask_size),
        },
        "inference_time_ms": inference_time,
    }


async def market_data_consumer(publisher_socket, context):
    """Consume market data from ZeroMQ and process"""
    socket = context.socket(zmq.SUB)
    socket.connect(MARKET_DATA_SOCKET)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
    
    print(f"📡 Connected to market data socket: {MARKET_DATA_SOCKET}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processed_count = 0
    last_log_time = time.time()
    
    while True:
        try:
            # Receive message (non-blocking)
            try:
                message = await asyncio.wait_for(
                    socket.recv_multipart(zmq.NOBLOCK),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                await asyncio.sleep(0.001)
                continue
            
            if not message:
                continue
            
            # Parse book update
            book_update = parse_book_update(message[0])
            if not book_update:
                continue
            
            # Update order book buffer
            symbol = book_update.get("symbol", "BTC-USD")
            bids = book_update.get("bids", [])
            asks = book_update.get("asks", [])
            timestamp = book_update.get("timestamp", 0)
            
            update_order_book_buffer(symbol, bids, asks, timestamp)
            
            # Run inference
            result = await run_inference(symbol, device)
            if result is None:
                continue
            
            # Send signal to Rust (OrderInstruction format)
            # Rust serde will deserialize string enums
            signal_data_bid = {
                "symbol": result["symbol"],
                "side": "Buy",
                "price": float(result["quotes"]["bid_price"]),
                "size": float(result["quotes"]["bid_size"]),
                "order_type": "PostOnly",
                "reduce_only": False,
            }
            
            signal_data_ask = {
                "symbol": result["symbol"],
                "side": "Sell",
                "price": float(result["quotes"]["ask_price"]),
                "size": float(result["quotes"]["ask_size"]),
                "order_type": "PostOnly",
                "reduce_only": False,
            }
            
            # Send to Rust via ZeroMQ (single message per order)
            try:
                publisher_socket.send(
                    msgpack.packb(signal_data_bid, use_bin_type=True),
                    zmq.NOBLOCK
                )
                publisher_socket.send(
                    msgpack.packb(signal_data_ask, use_bin_type=True),
                    zmq.NOBLOCK
                )
            except zmq.Again:
                pass  # Non-blocking send failed, skip
            
            processed_count += 1
            
            # Log periodically
            if time.time() - last_log_time > 5.0:
                print(f"📊 Processed {processed_count} updates | "
                      f"Prediction: {result['lit_prediction']:.3f} | "
                      f"Confidence: {result['confidence']:.3f} | "
                      f"Inference: {result['inference_time_ms']:.2f}ms")
                last_log_time = time.time()
                processed_count = 0
                
        except Exception as e:
            print(f"Error in market data consumer: {e}")
            await asyncio.sleep(0.1)


async def main():
    """Main async entry point"""
    # Set event loop policy to uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    print("🧠 Starting Hyperliquid HFT Alpha Engine...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_model(device)
    
    # Setup ZeroMQ sockets
    context = zmq.asyncio.Context()
    
    # Publisher for sending signals to Rust
    publisher = context.socket(zmq.PUB)
    publisher.bind(SIGNALS_SOCKET)
    await asyncio.sleep(0.1)  # Give bind time to establish
    print(f"📤 Signal publisher bound to: {SIGNALS_SOCKET}")
    
    # Handle shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start market data consumer
    consumer_task = asyncio.create_task(
        market_data_consumer(publisher, context)
    )
    
    print("✅ Alpha engine running...")
    
    # Wait for shutdown
    await shutdown_event.wait()
    
    # Cleanup
    consumer_task.cancel()
    publisher.close()
    context.term()
    
    print("👋 Alpha engine stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

