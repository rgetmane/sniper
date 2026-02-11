"""
Yellowstone gRPC scanner — streams Pump.fun transactions in real-time.
Detects new token mints and bonding curve events < 50ms from chain.
"""

import asyncio
import grpc
import grpc.aio
import logging
import os
import sys
import time
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))
from proto import geyser_pb2, geyser_pb2_grpc

logger = logging.getLogger("scanner")

PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
RAYDIUM_AMM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMP_CREATE_DISC = bytes([0x18, 0x1e, 0xc8, 0x28, 0x05, 0x1c, 0x07, 0x77])
PUMP_BUY_DISC = bytes([0x66, 0x06, 0x3d, 0x12, 0x01, 0xda, 0xeb, 0xea])

scanner_stats: Dict[str, Any] = {
    "started": 0.0,
    "txns_seen": 0,
    "pumps_detected": 0,
    "last_pump": None,
    "last_pump_ts": 0.0,
    "connected": False,
    "errors": 0,
}


def _b58encode(data: bytes) -> str:
    alphabet = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    num = int.from_bytes(data, "big")
    result = bytearray()
    while num > 0:
        num, rem = divmod(num, 58)
        result.append(alphabet[rem])
    for byte in data:
        if byte == 0:
            result.append(alphabet[0])
        else:
            break
    return bytes(reversed(result)).decode("ascii")


def parse_pump_tx(tx_data: Any) -> Optional[Dict[str, Any]]:
    try:
        tx = tx_data.transaction
        meta = tx_data.meta
        if not tx or not meta:
            return None
        if meta.err and meta.err.err:
            return None
        msg = tx.message
        if not msg:
            return None

        account_keys = [_b58encode(k) for k in msg.account_keys]
        if meta.loaded_writable_addresses:
            account_keys.extend([_b58encode(k) for k in meta.loaded_writable_addresses])
        if meta.loaded_readonly_addresses:
            account_keys.extend([_b58encode(k) for k in meta.loaded_readonly_addresses])

        for ix in msg.instructions:
            program_idx = ix.program_id_index
            if program_idx >= len(account_keys):
                continue
            if account_keys[program_idx] != PUMPFUN_PROGRAM:
                continue
            data = ix.data
            if len(data) < 8:
                continue
            disc = data[:8]
            event_type = None
            if disc == PUMP_CREATE_DISC:
                event_type = "create"
            elif disc == PUMP_BUY_DISC:
                event_type = "buy"
            else:
                continue
            mint_idx = 0 if event_type == "create" else 2
            if mint_idx < len(ix.accounts):
                acct_idx = ix.accounts[mint_idx]
                if acct_idx < len(account_keys):
                    return {
                        "type": event_type,
                        "mint": account_keys[acct_idx],
                        "signature": _b58encode(tx_data.signature),
                        "ts": time.time(),
                    }

        for inner in meta.inner_instructions:
            for ix in inner.instructions:
                program_idx = ix.program_id_index
                if program_idx >= len(account_keys):
                    continue
                if account_keys[program_idx] != PUMPFUN_PROGRAM:
                    continue
                data = ix.data
                if len(data) >= 8 and data[:8] == PUMP_CREATE_DISC:
                    if len(ix.accounts) > 0 and ix.accounts[0] < len(account_keys):
                        return {
                            "type": "create",
                            "mint": account_keys[ix.accounts[0]],
                            "signature": _b58encode(tx_data.signature),
                            "ts": time.time(),
                        }
    except Exception as e:
        logger.debug(f"parse error: {e}")
        return None
    return None


async def _subscribe_stream(stub):
    async def request_iterator():
        req = geyser_pb2.SubscribeRequest()
        pump_filter = geyser_pb2.SubscribeRequestFilterTransactions()
        pump_filter.vote = False
        pump_filter.failed = False
        pump_filter.account_include.append(PUMPFUN_PROGRAM)
        req.transactions["pumpfun"].CopyFrom(pump_filter)
        ray_filter = geyser_pb2.SubscribeRequestFilterTransactions()
        ray_filter.vote = False
        ray_filter.failed = False
        ray_filter.account_include.append(RAYDIUM_AMM)
        req.transactions["raydium"].CopyFrom(ray_filter)
        req.commitment = geyser_pb2.PROCESSED
        yield req
        while True:
            await asyncio.sleep(30)
            ping_req = geyser_pb2.SubscribeRequest()
            ping_req.ping.id = int(time.time()) % 2147483647
            yield ping_req

    return stub.Subscribe(request_iterator())


async def run_scanner(on_pump=None, endpoint=None, api_key=None):
    if not endpoint:
        endpoint = os.getenv("YELLOWSTONE_GRPC_ENDPOINT", "laserstream-mainnet-ewr.helius-rpc.com:443")
    if not api_key:
        api_key = os.getenv("YELLOWSTONE_API_KEY", "")

    backoff = 1.0
    scanner_stats["started"] = time.time()

    while True:
        channel = None
        try:
            logger.info(f"connecting to {endpoint}")
            ssl_creds = grpc.ssl_channel_credentials()
            if api_key:
                auth_metadata = [("x-token", api_key)]
                call_creds = grpc.metadata_call_credentials(
                    lambda context, callback: callback(auth_metadata, None)
                )
                channel_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)
            else:
                channel_creds = ssl_creds
            options = [
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.initial_reconnect_backoff_ms", 500),
            ]
            channel = grpc.aio.secure_channel(endpoint, channel_creds, options=options)
            stub = geyser_pb2_grpc.GeyserStub(channel)
            response_stream = await _subscribe_stream(stub)
            scanner_stats["connected"] = True
            backoff = 1.0
            logger.info("yellowstone stream LIVE")

            async for update in response_stream:
                if update.HasField("pong") or update.HasField("ping"):
                    continue
                if update.HasField("transaction"):
                    scanner_stats["txns_seen"] += 1
                    tx_info = update.transaction.transaction
                    event = parse_pump_tx(tx_info)
                    if event:
                        scanner_stats["pumps_detected"] += 1
                        scanner_stats["last_pump"] = event["mint"]
                        scanner_stats["last_pump_ts"] = event["ts"]
                        if event["type"] == "create":
                            logger.info(f"NEW PUMP: {event['mint']} sig={event['signature'][:16]}...")
                        if on_pump:
                            try:
                                await on_pump(event)
                            except Exception as e:
                                logger.error(f"on_pump error: {e}")

        except grpc.aio.AioRpcError as e:
            scanner_stats["connected"] = False
            scanner_stats["errors"] += 1
            logger.warning(f"gRPC error: {e.code()} {e.details()}")
        except Exception as e:
            scanner_stats["connected"] = False
            scanner_stats["errors"] += 1
            logger.warning(f"scanner error: {type(e).__name__}: {e}")
        finally:
            scanner_stats["connected"] = False
            if channel:
                try:
                    await channel.close()
                except Exception:
                    pass
        logger.info(f"reconnecting in {backoff:.1f}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


def get_stats() -> Dict[str, Any]:
    return {
        "connected": scanner_stats["connected"],
        "uptime": time.time() - scanner_stats["started"] if scanner_stats["started"] else 0,
        "txns_seen": scanner_stats["txns_seen"],
        "pumps_detected": scanner_stats["pumps_detected"],
        "last_pump": scanner_stats["last_pump"],
        "last_pump_ts": scanner_stats["last_pump_ts"],
        "errors": scanner_stats["errors"],
    }
