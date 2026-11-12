"""
Cryptographic SHA-256 Hash-Chained Audit Log Module
Provides immutable, tamper-proof record of all PI gain proposals, RLS estimates,
safety gating decisions, active learning updates, and automatic fallbacks/rollbacks.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple


class AuditBlock:
    """A single cryptographic block in the ShadowTune audit ledger."""

    def __init__(
        self,
        index: int,
        timestamp: str,
        event_type: str,
        data: Dict[str, Any],
        prev_hash: str,
    ):
        self.index = index
        self.timestamp = timestamp
        self.event_type = event_type
        self.data = data
        self.prev_hash = prev_hash
        self.block_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of block contents."""
        serialized = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "event_type": self.event_type,
                "data": self.data,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "prev_hash": self.prev_hash,
            "block_hash": self.block_hash,
            "data": self.data,
        }


class HashChainedAuditLedger:
    """
    Append-only SHA-256 blockchain-style audit log ensuring integrity
    and regulatory compliance (IEC 61851-23 / ISO 26262).
    """

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self.chain: List[AuditBlock] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis_data = {
            "system": "ShadowTune EV Fast Charger PI Auto-Tuner",
            "version": "2.0-Verified",
            "initialization": "Genesis Block Root",
            "status": "OPERATIONAL"
        }
        timestamp = datetime.now(timezone.utc).isoformat()
        genesis = AuditBlock(
            index=0,
            timestamp=timestamp,
            event_type="GENESIS",
            data=genesis_data,
            prev_hash=self.GENESIS_HASH,
        )
        self.chain.append(genesis)

    def append_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> AuditBlock:
        """Append a new verified record to the hash chain."""
        prev_block = self.chain[-1]
        timestamp = datetime.now(timezone.utc).isoformat()
        new_block = AuditBlock(
            index=len(self.chain),
            timestamp=timestamp,
            event_type=event_type,
            data=data,
            prev_hash=prev_block.block_hash,
        )
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify that the entire audit chain has not been tampered with.
        Checks both link continuity and hash validity for every block.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # 1. Verify previous hash pointer
            if current.prev_hash != previous.block_hash:
                return False, f"Broken link at block #{current.index}: prev_hash does not match block #{previous.index} hash."

            # 2. Verify current block hash integrity
            recomputed = current.compute_hash()
            if current.block_hash != recomputed:
                return False, f"Tampered data in block #{current.index}: stored hash {current.block_hash} != recomputed {recomputed}."

        return True, "Chain integrity verified: all cryptographic hashes valid."

    def export_json(self, filepath: str):
        """Export full audit ledger to formatted JSON file."""
        records = [b.to_dict() for b in self.chain]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def export_summary_table(self) -> List[Dict[str, Any]]:
        """Export tabular summary for reporting."""
        summary = []
        for b in self.chain:
            d = b.data
            summary.append({
                "block_idx": b.index,
                "timestamp": b.timestamp[:19],
                "event": b.event_type,
                "short_hash": b.block_hash[:10] + "...",
                "decision": d.get("decision", d.get("status", "N/A")),
                "kp": d.get("kp", d.get("applied_kp", "N/A")),
                "ki": d.get("ki", d.get("applied_ki", "N/A")),
                "cost": d.get("actual_cost", d.get("composite_cost", "N/A")),
                "rationale": d.get("rationale", d.get("reason", "OK")),
            })
        return summary
