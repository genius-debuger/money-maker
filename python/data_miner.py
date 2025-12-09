"""
Hyperliquid S3 Data Miner
Downloads historical L2 order book data from Hyperliquid's S3 archive
"""

import boto3
import lz4.frame
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional
import traceback

# S3 Configuration
S3_BUCKET = "hyperliquid-archive"
S3_PREFIX = "market_data"
S3_REQUEST_PAYER = "requester"

# Local storage
DATA_DIR = Path("data/parquet")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_and_process_file(
    s3_client: boto3.client,
    date: str,
    coin: str,
    output_path: Path,
) -> bool:
    """
    Download a single L2 order book file from S3, decompress, and save as Parquet
    
    Args:
        s3_client: Boto3 S3 client
        date: Date string in YYYY-MM-DD format
        coin: Coin symbol (e.g., "BTC")
        output_path: Path to save the Parquet file
    
    Returns:
        True if successful, False otherwise
    """
    s3_key = f"{S3_PREFIX}/{date}/l2Book/{coin}.lz4"
    
    try:
        print(f"Downloading: s3://{S3_BUCKET}/{s3_key}")
        
        # Download compressed file
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            RequestPayer=S3_REQUEST_PAYER,
        )
        
        compressed_data = response["Body"].read()
        
        # Decompress LZ4
        print(f"Decompressing {len(compressed_data)} bytes...")
        decompressed_data = lz4.frame.decompress(compressed_data)
        
        # Parse JSON lines (assuming each line is a JSON object)
        # Hyperliquid format: each line is a snapshot
        lines = decompressed_data.decode("utf-8").strip().split("\n")
        
        # Parse JSON lines into Polars DataFrame
        records = []
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue
            
            try:
                import json
                snapshot = json.loads(line)
                
                # Extract timestamp, bids, asks
                # Format may vary - adjust based on actual Hyperliquid format
                timestamp = snapshot.get("time", snapshot.get("timestamp", line_num))
                bids = snapshot.get("bids", snapshot.get("data", {}).get("bids", []))
                asks = snapshot.get("asks", snapshot.get("data", {}).get("asks", []))
                
                # Flatten bids and asks (price, size pairs)
                for level, (price, size) in enumerate(bids[:20]):  # Top 20 levels
                    records.append({
                        "timestamp": timestamp,
                        "side": "bid",
                        "level": level,
                        "price": float(price),
                        "size": float(size),
                    })
                
                for level, (price, size) in enumerate(asks[:20]):  # Top 20 levels
                    records.append({
                        "timestamp": timestamp,
                        "side": "ask",
                        "level": level,
                        "price": float(price),
                        "size": float(size),
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
        
        if not records:
            print(f"Warning: No records extracted from {s3_key}")
            return False
        
        # Create Polars DataFrame
        df = pl.DataFrame(records)
        
        # Save as Parquet (compressed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            output_path,
            compression="zstd",  # Fast compression
            use_pyarrow=False,
        )
        
        print(f"✅ Saved {len(records)} records to {output_path}")
        return True
        
    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  File not found: {s3_key}")
        return False
    except Exception as e:
        print(f"❌ Error processing {s3_key}: {e}")
        traceback.print_exc()
        return False


def download_date_range(
    start_date: str,
    end_date: str,
    coins: list[str],
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> None:
    """
    Download data for a date range and list of coins
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        coins: List of coin symbols
        aws_access_key_id: Optional AWS access key
        aws_secret_access_key: Optional AWS secret key
    """
    # Initialize S3 client
    if aws_access_key_id and aws_secret_access_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        # Use default credentials (from ~/.aws/credentials or environment)
        s3_client = boto3.client("s3")
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    total_files = 0
    successful = 0
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"\n📅 Processing date: {date_str}")
        
        for coin in coins:
            output_path = DATA_DIR / date_str / f"{coin}.parquet"
            
            # Skip if already exists
            if output_path.exists():
                print(f"⏭️  Skipping {coin} (already exists)")
                continue
            
            total_files += 1
            if download_and_process_file(s3_client, date_str, coin, output_path):
                successful += 1
        
        current += timedelta(days=1)
    
    print(f"\n✅ Download complete: {successful}/{total_files} files successful")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Hyperliquid historical data")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH"], help="Coin symbols")
    parser.add_argument("--aws-access-key-id", help="AWS access key ID")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key")
    
    args = parser.parse_args()
    
    download_date_range(
        args.start_date,
        args.end_date,
        args.coins,
        args.aws_access_key_id,
        args.aws_secret_access_key,
    )


if __name__ == "__main__":
    main()

