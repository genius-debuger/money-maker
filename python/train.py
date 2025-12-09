"""
LiT (Limit Order Book Transformer) Training Script
Trains the model on historical Hyperliquid data using Triple-Barrier labeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import polars as pl
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
import argparse
from datetime import datetime

from model import LiTTransformer, create_lit_model


class OrderBookDataset(Dataset):
    """
    Dataset for order book sequences
    Loads Parquet files and creates sequences for training
    """
    
    def __init__(
        self,
        data_dir: Path,
        sequence_length: int = 100,
        num_levels: int = 20,
        profit_take: float = 0.001,  # 0.1% profit take
        stop_loss: float = 0.0005,    # 0.05% stop loss
        time_limit: int = 10,         # 10 timesteps
    ):
        self.sequence_length = sequence_length
        self.num_levels = num_levels
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        self.time_limit = time_limit
        
        # Load all Parquet files
        self.data_files = list(data_dir.glob("**/*.parquet"))
        print(f"Found {len(self.data_files)} data files")
        
        # Load and preprocess data
        self.sequences = []
        self.labels = []
        
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess data from Parquet files"""
        for file_path in self.data_files:
            try:
                df = pl.read_parquet(file_path)
                
                # Group by timestamp to get snapshots
                snapshots = df.group_by("timestamp").agg([
                    pl.col("price").filter(pl.col("side") == "bid").sort().head(self.num_levels).alias("bid_prices"),
                    pl.col("size").filter(pl.col("side") == "bid").sort().head(self.num_levels).alias("bid_sizes"),
                    pl.col("price").filter(pl.col("side") == "ask").sort(descending=True).head(self.num_levels).alias("ask_prices"),
                    pl.col("size").filter(pl.col("side") == "ask").sort(descending=True).head(self.num_levels).alias("ask_sizes"),
                ]).sort("timestamp")
                
                # Create sequences
                for i in range(len(snapshots) - self.sequence_length - self.time_limit):
                    sequence_data = snapshots[i:i + self.sequence_length]
                    
                    # Build input tensor: (seq_len, 40, 2)
                    # 40 = 20 bids + 20 asks, 2 = (price, size)
                    sequence = np.zeros((self.sequence_length, 40, 2), dtype=np.float32)
                    
                    for t, row in enumerate(sequence_data.iter_rows(named=True)):
                        bid_prices = row.get("bid_prices", [])
                        bid_sizes = row.get("bid_sizes", [])
                        ask_prices = row.get("ask_prices", [])
                        ask_sizes = row.get("ask_sizes", [])
                        
                        # Fill bids (first 20 levels)
                        for level in range(min(len(bid_prices), self.num_levels)):
                            if level < len(bid_prices) and level < len(bid_sizes):
                                sequence[t, level, 0] = float(bid_prices[level])
                                sequence[t, level, 1] = float(bid_sizes[level])
                        
                        # Fill asks (next 20 levels)
                        for level in range(min(len(ask_prices), self.num_levels)):
                            if level < len(ask_prices) and level < len(ask_sizes):
                                sequence[t, 20 + level, 0] = float(ask_prices[level])
                                sequence[t, 20 + level, 1] = float(ask_sizes[level])
                    
                    # Label using Triple-Barrier Method
                    label = self._triple_barrier_label(sequence_data, i + self.sequence_length)
                    
                    if label is not None:
                        self.sequences.append(sequence)
                        self.labels.append(label)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(self.sequences)} sequences with labels")
    
    def _triple_barrier_label(self, future_data, start_idx: int) -> Optional[int]:
        """
        Triple-Barrier Method: Classify as Up (0), Down (1), or Stationary (2)
        
        Returns:
            0 = Up (profit take hit first)
            1 = Down (stop loss hit first)
            2 = Stationary (time limit hit first)
            None = Invalid sequence
        """
        if start_idx + self.time_limit >= len(future_data):
            return None
        
        # Get current mid price
        current_snapshot = future_data[start_idx]
        current_mid = self._get_mid_price(current_snapshot)
        
        if current_mid is None:
            return None
        
        # Check future timesteps
        for i in range(1, self.time_limit + 1):
            if start_idx + i >= len(future_data):
                break
            
            future_snapshot = future_data[start_idx + i]
            future_mid = self._get_mid_price(future_snapshot)
            
            if future_mid is None:
                continue
            
            # Calculate return
            ret = (future_mid - current_mid) / current_mid
            
            # Check barriers
            if ret >= self.profit_take:
                return 0  # Up
            elif ret <= -self.stop_loss:
                return 1  # Down
        
        # Time limit reached without hitting barriers
        return 2  # Stationary
    
    def _get_mid_price(self, snapshot) -> Optional[float]:
        """Extract mid price from snapshot"""
        bid_prices = snapshot.get("bid_prices", [])
        ask_prices = snapshot.get("ask_prices", [])
        
        if not bid_prices or not ask_prices:
            return None
        
        best_bid = float(bid_prices[0]) if bid_prices else None
        best_ask = float(ask_prices[0]) if ask_prices else None
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.from_numpy(self.sequences[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        
        # Compute loss (cross-entropy)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / max(total, 1)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train LiT Transformer")
    parser.add_argument("--data-dir", type=Path, default=Path("data/parquet"), help="Data directory")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Model save directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting LiT training on {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = OrderBookDataset(
        args.data_dir,
        sequence_length=args.seq_len,
        num_levels=20,
    )
    
    if len(dataset) == 0:
        print("❌ No data found! Please run data_miner.py first.")
        return
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = create_lit_model(
        seq_len=args.seq_len,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        lstm_hidden=256,
        dropout=0.1,
    ).to(args.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, args.device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = args.model_dir / "lit_v1.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  💾 Saved best model to {model_path}")
        
        print()
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()

