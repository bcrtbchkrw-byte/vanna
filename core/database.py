"""
Vanna Database Module

SQLite database for trades, positions, and P&L tracking.
Uses aiosqlite for async operations.
"""

import aiosqlite
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from core.logger import get_logger

logger = get_logger()

_database: Optional['Database'] = None


class Database:
    """
    SQLite database manager for Vanna.
    
    Tables:
        - trades: Trade history with P&L
        - positions: Current open positions
        - pnl_history: Daily P&L snapshots
        - ai_decisions: AI decision audit trail
    """
    
    def __init__(self, db_path: str = "data/vanna.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self) -> None:
        """Open database connection and create tables."""
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info(f"Database connected: {self.db_path}")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database disconnected")
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        await self._connection.executescript("""
            -- Trades table
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity INTEGER NOT NULL,
                pnl REAL,
                status TEXT DEFAULT 'OPEN',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Positions table
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                legs TEXT NOT NULL,  -- JSON array of option legs
                entry_time TIMESTAMP NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                target_profit REAL,
                stop_loss REAL,
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- P&L History table
            CREATE TABLE IF NOT EXISTS pnl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- AI Decisions table
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                phase TEXT NOT NULL,  -- gemini, claude
                decision TEXT NOT NULL,  -- approve, reject
                reasoning TEXT,
                confidence REAL,
                cost REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol ON ai_decisions(symbol);
        """)
        await self._connection.commit()
    
    async def execute(self, query: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a query and return cursor."""
        cursor = await self._connection.execute(query, params)
        await self._connection.commit()
        return cursor
    
    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Fetch a single row as dict."""
        cursor = await self._connection.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None
    
    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as list of dicts."""
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Trade Operations
    # =========================================================================
    
    async def insert_trade(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        quantity: int,
        notes: str = ""
    ) -> int:
        """Insert a new trade."""
        cursor = await self.execute(
            """
            INSERT INTO trades (symbol, strategy, entry_time, entry_price, quantity, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, strategy, datetime.now(), entry_price, quantity, notes)
        )
        logger.info(f"TRADE: Inserted trade {symbol} {strategy}")
        return cursor.lastrowid
    
    async def close_trade(self, trade_id: int, exit_price: float, pnl: float) -> None:
        """Close an existing trade."""
        await self.execute(
            """
            UPDATE trades 
            SET exit_time = ?, exit_price = ?, pnl = ?, status = 'CLOSED'
            WHERE id = ?
            """,
            (datetime.now(), exit_price, pnl, trade_id)
        )
        logger.info(f"TRADE: Closed trade {trade_id} P&L: ${pnl:.2f}")
    
    async def get_open_trades(self) -> list[dict]:
        """Get all open trades."""
        return await self.fetch_all(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time DESC"
        )
    
    # =========================================================================
    # AI Decision Logging
    # =========================================================================
    
    async def log_ai_decision(
        self,
        symbol: str,
        phase: str,
        decision: str,
        reasoning: str,
        confidence: float,
        cost: float
    ) -> int:
        """Log an AI decision for audit."""
        cursor = await self.execute(
            """
            INSERT INTO ai_decisions (symbol, phase, decision, reasoning, confidence, cost)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, phase, decision, reasoning, confidence, cost)
        )
        return cursor.lastrowid


async def get_database() -> Database:
    """
    Get the global database instance.
    
    Returns:
        Database: Connected database instance
    """
    global _database
    
    if _database is None:
        _database = Database()
        await _database.connect()
    
    return _database
