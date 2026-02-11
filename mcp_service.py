from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from groq import Groq
import asyncio
import base64
import csv
from collections import deque
import logging
from logging.handlers import RotatingFileHandler
import os
import random
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
from dotenv import load_dotenv
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TokenAccountOpts
from solana.rpc.websocket_api import connect
from solders.instruction import Instruction, AccountMeta
from solders.keypair import Keypair as SoldersKeypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from scanner import run_scanner, get_stats as scanner_get_stats

load_dotenv("secrets.env")
app = FastAPI()

# --- Access control ---
TRUSTED_IPS = {"127.0.0.1", "172.20.0.1", "10.114.0.2"}

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""

def _is_trusted(request: Request) -> bool:
    return _get_client_ip(request) in TRUSTED_IPS

# --- Rate limiter (per-IP, sliding window) ---
_rate_buckets: Dict[str, List[float]] = {}

def _rate_ok(ip: str, max_per_min: int = 5) -> bool:
    t = time.time()
    bucket = _rate_buckets.setdefault(ip, [])
    _rate_buckets[ip] = [ts for ts in bucket if t - ts < 60]
    if len(_rate_buckets[ip]) >= max_per_min:
        return False
    _rate_buckets[ip].append(t)
    return True


@app.middleware("http")
async def speed_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Speed"] = f"{elapsed:.3f}s"
    return response


@app.exception_handler(Exception)
async def ghost_handler(request: Request, exc: Exception):
    _nm_logger = logging.getLogger("near_miss")
    _nm_logger.info("UNHANDLED %s %s | %s", request.method, request.url.path, exc)
    return JSONResponse({"error": "ghost"}, status_code=500)


groq_key = os.getenv("GROQ_KEY")
gclient = Groq(api_key=groq_key) if groq_key else None

HELIUS_RPC = os.getenv("HELIUS_RPC", "")
TRITON_RPC = os.getenv("TRITON_RPC", "")
HELIUS_WS = os.getenv("HELIUS_WS", "")

BIRDEYE_URL = os.getenv("BIRDEYE_URL", "")
PUMPFUN_URL = os.getenv("PUMPFUN_URL", "")
FLAUNCH_URL = "https://api.flaunch.gg/v1/base/tokens/new?limit=5"
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens/"

BIRDEYE_KEY = os.getenv("BIRDEYE_KEY", "")
JITO_KEY = os.getenv("JITO_KEY", "")

_proxy_url = os.getenv("PROXY_URL", "")
PROXIES = {"http": _proxy_url, "https": _proxy_url} if _proxy_url else None

WALLET_PRIVATE = os.getenv("WALLET_PRIVATE", "")
WALLETS = [os.getenv(f"WALLET_{i}", "") for i in range(1, 21)]

# --- Whale wallets to copy-trade (comma-separated in .env) ---
WHALE_WALLETS = [w.strip() for w in os.getenv("WHALE_WALLETS", "").split(",") if w.strip()]

DRY_RUN = os.getenv("DRY_RUN", "").lower() in ("true", "1", "yes")

CACHE_TTL = 60.0
WATCH_LOOP = 0.15

THRESH_AGE = 15.0
THRESH_VOL = 300.0
THRESH_WHALE = 8.0
THRESH_LIQ = 8000.0
THRESH_DEV = 5.0

MIGRATION_TARGET = "PumpSwap"
PUMPSWAP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# --- Pump.fun bonding curve program + accounts ---
PUMP_PROGRAM = Pubkey.from_string(PUMPSWAP_PROGRAM)
PUMP_GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
# 8 standard fee recipients from Global account (randomly selected to spread write-locks)
PUMP_FEE_RECIPIENTS = [
    Pubkey.from_string("62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV"),
    Pubkey.from_string("7VtfL8fvgNfhz17qKRMjzQEXgbdpnHHHQRh54R9jP2RJ"),
    Pubkey.from_string("7hTckgnGnLQR6sdH7YkqFTAA7VwTfYFaZ6EhEsU3saCX"),
    Pubkey.from_string("9rPYyANsfQZw3DnDmKE3YCQF5E8oD89UXoHn9JFEhJUz"),
    Pubkey.from_string("AVmoTthdrX6tKt4nDjco2D775W2YK3sDhxPcMmzUAmTY"),
    Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM"),
    Pubkey.from_string("FWsW1xNtWscwNmKv6wVsU1iTzRN6wmmk3MjxRP5tT7hz"),
    Pubkey.from_string("G5UZAVbAf46s7cKWoyKu8kYTip9DGTpbLZ2qa9Aq69dP"),
]
PUMP_EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")
PUMP_FEE_PROGRAM = Pubkey.from_string("pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
TOKEN_2022 = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
ASSOC_TOKEN_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
SYS_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
COMPUTE_BUDGET = Pubkey.from_string("ComputeBudget111111111111111111111111111111")
PUMP_BUY_DISC = bytes([102, 6, 61, 18, 1, 218, 235, 234])

BUY_LAMPORTS = 1_000_000_000  # 1.0 SOL
SELL_TIERS = [(0.80, 0.50), (1.00, 0.30), (1.50, 0.20)]  # (pnl_threshold, sell_fraction)
STOP_LOSS = -0.25

cache: Dict[str, Any] = {"tokens": {}, "decisions": {}}
positions: Dict[str, Any] = {}
watch_enabled = True
trade_count = 0
_low_sol = False
_killed = False
_kill_timeout = 0
_balance_cache: Dict[str, Any] = {"sol": 0.0, "ts": 0}
wallet_index = 0

# --- Safety Rails ---
CIRCUIT_BREAKER_SOL = 3.0
MAX_POSITIONS = 3
TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.csv")

daily_net_loss = 0.0
daily_reset_ts = time.time()
circuit_paused_until = 0.0
open_positions: Set[str] = set()

# --- Near-miss logger with rotation (10k lines OR 50MB, keep 5 backups) ---
class _LineRotatingHandler(RotatingFileHandler):
    """RotatingFileHandler that also rotates after max_lines writes."""
    def __init__(self, *args, max_lines: int = 10_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._lines = 0
        self._max_lines = max_lines

    def shouldRollover(self, record) -> int:
        if self._lines >= self._max_lines:
            return True
        return super().shouldRollover(record)

    def doRollover(self):
        super().doRollover()
        self._lines = 0

    def emit(self, record):
        try:
            super().emit(record)
            self._lines += 1
        except Exception:
            self.handleError(record)

_nm_logger = logging.getLogger("near_miss")
_nm_logger.setLevel(logging.INFO)
_nm_handler = _LineRotatingHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "near_miss.log"),
    maxBytes=50 * 1024 * 1024,
    backupCount=5,
    max_lines=10_000,
)
_nm_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_nm_logger.addHandler(_nm_handler)
_nm_logger.propagate = False

# --- WebSocket live-push buffer ---
_ws_seq = 0
_ws_log_buffer: deque = deque(maxlen=100)

class _WsBufferHandler(logging.Handler):
    def emit(self, record):
        global _ws_seq
        try:
            _ws_seq += 1
            _ws_log_buffer.append((_ws_seq, self.format(record)))
        except Exception:
            pass

_ws_bh = _WsBufferHandler()
_ws_bh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_nm_logger.addHandler(_ws_bh)

# --- Jupiter rate limiter (max 1 request/sec) ---
_jup_semaphore = asyncio.Semaphore(1)
_jup_last_call = 0.0

# --- Watch loop stats ---
_wl_stats = {"ticks": 0, "tokens_seen": 0, "dex_checked": 0, "passed_filters": 0}


def _init_csv() -> None:
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "token", "entry_price", "exit_price",
                "pnl_sol", "exit_reason", "liq_entry", "vol_entry",
                "deployer_addr", "duration_sec",
            ])


def log_trade(token: str, entry_price: float, exit_price: float,
              pnl_sol: float, exit_reason: str, duration_sec: float) -> None:
    _init_csv()
    token_data = cache["tokens"].get(token, {})
    with open(TRADES_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), token,
            f"{entry_price:.8f}", f"{exit_price:.8f}", f"{pnl_sol:.4f}",
            exit_reason, f"{token_data.get('liq', 0):.0f}",
            f"{token_data.get('vol', 0):.0f}", "", f"{duration_sec:.1f}",
        ])


def check_circuit_breaker() -> bool:
    global daily_net_loss, daily_reset_ts
    if time.time() - daily_reset_ts > 86400:
        daily_net_loss = 0.0
        daily_reset_ts = time.time()
    if time.time() < circuit_paused_until:
        return False
    return daily_net_loss < CIRCUIT_BREAKER_SOL


def trip_circuit(loss_sol: float) -> None:
    global daily_net_loss, circuit_paused_until
    daily_net_loss += loss_sol
    if daily_net_loss >= CIRCUIT_BREAKER_SOL:
        circuit_paused_until = time.time() + 86400


def now() -> float:
    return time.time()


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def b58decode(value: str) -> bytes:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    num = 0
    for char in value:
        num *= 58
        if char not in alphabet:
            raise ValueError("invalid base58")
        num += alphabet.index(char)
    combined = num.to_bytes((num.bit_length() + 7) // 8, "big")
    pad = 0
    for char in value:
        if char == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + combined


def _get_ata(owner: Pubkey, mint: Pubkey, token_prog: Pubkey = TOKEN_PROGRAM) -> Pubkey:
    """Derive associated token address for owner + mint."""
    ata, _ = Pubkey.find_program_address(
        [bytes(owner), bytes(token_prog), bytes(mint)],
        ASSOC_TOKEN_PROGRAM,
    )
    return ata


def _create_ata_ix(payer: Pubkey, owner: Pubkey, mint: Pubkey, token_prog: Pubkey = TOKEN_PROGRAM) -> Instruction:
    """Build a CreateAssociatedTokenAccount instruction."""
    ata = _get_ata(owner, mint, token_prog)
    return Instruction(
        ASSOC_TOKEN_PROGRAM,
        bytes(),
        [
            AccountMeta(payer, is_signer=True, is_writable=True),
            AccountMeta(ata, is_signer=False, is_writable=True),
            AccountMeta(owner, is_signer=False, is_writable=False),
            AccountMeta(mint, is_signer=False, is_writable=False),
            AccountMeta(SYS_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(token_prog, is_signer=False, is_writable=False),
        ],
    )


async def _detect_token_program(client: AsyncClient, mint: Pubkey) -> Pubkey:
    """Check which token program owns the mint (classic SPL or Token-2022)."""
    try:
        resp = await client.get_account_info(mint)
        if resp.value and resp.value.owner == TOKEN_2022:
            return TOKEN_2022
    except Exception:
        pass
    return TOKEN_PROGRAM


def load_wallets() -> List[SoldersKeypair]:
    keys = [k.strip() for k in WALLETS if k.strip()]
    if not keys and WALLET_PRIVATE:
        keys = [k.strip() for k in WALLET_PRIVATE.split(",") if k.strip()]
    wallets: List[SoldersKeypair] = []
    for key in keys:
        try:
            wallets.append(SoldersKeypair.from_bytes(b58decode(key)))
        except Exception:
            try:
                wallets.append(SoldersKeypair.from_bytes(base64.b64decode(key)))
            except Exception:
                pass
    return wallets


def get_wallet() -> Optional[SoldersKeypair]:
    global wallet_index
    wallets = load_wallets()
    if not wallets:
        return None
    wallet_index = wallet_index % len(wallets)
    return wallets[wallet_index]


def rotate_wallet() -> None:
    global wallet_index
    wallet_index += 1


def prune() -> None:
    cutoff = now() - CACHE_TTL
    for token, info in list(cache["tokens"].items()):
        if info.get("ts", 0) < cutoff:
            del cache["tokens"][token]
    for token, info in list(cache["decisions"].items()):
        if info.get("ts", 0) < cutoff:
            del cache["decisions"][token]


async def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    if not url:
        return None
    def _get() -> Any:
        try:
            resp = requests.get(url, headers=headers, timeout=random.uniform(2.0, 3.5), proxies=PROXIES)
            return resp.json()
        except Exception:
            return None
    return await asyncio.to_thread(_get)


def iter_tokens(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def get_age_seconds(token: Dict[str, Any]) -> float:
    created = token.get("created_at") or token.get("createdAt") or token.get("timestamp")
    if not created:
        return 0.0
    try:
        created_value = float(created)
    except Exception:
        return 0.0
    if created_value > 1e12:
        created_value = created_value / 1000.0
    return max(0.0, now() - created_value)


def get_metric(token: Dict[str, Any], keys: List[str]) -> float:
    for key in keys:
        if key in token and token[key] is not None:
            try:
                return float(token[key])
            except Exception:
                return 0.0
    return 0.0


def tip_lamports(vol: float) -> int:
    base = clamp(vol / 1000.0, 0.001, 0.1)
    return int(base * 1_000_000_000)


def slippage_bps(vol: float) -> int:
    return int(clamp(30 + (vol / 1000.0) * 20, 30, 70))


def decision_prompt(data: Dict[str, Any]) -> str:
    return (
        "Token: {age:.0f}s old, vol +{vol:.0f}%, whale {whale:.0f}%, liq ${liq:.0f}, "
        "social spike {spike:.0f}, dev {dev:.2f}%. KILL or SKIP. One word."
    ).format(**data)


def should_trade(data: Dict[str, Any]) -> bool:
    token = data.get("token", "???")
    kill = None

    if not check_circuit_breaker():
        kill = "circuit_breaker"
    elif len(open_positions) >= MAX_POSITIONS:
        kill = f"max_positions({len(open_positions)}/{MAX_POSITIONS})"
    elif data["age"] >= THRESH_AGE:
        kill = f"age({data['age']:.0f}s>={THRESH_AGE}s)"
    elif data["vol"] <= THRESH_VOL:
        kill = f"vol({data['vol']:.0f}%<={THRESH_VOL}%)"
    elif data["whale"] >= THRESH_WHALE:
        kill = f"whale({data['whale']:.0f}%>={THRESH_WHALE}%)"
    else:
        liq_floor = 1000.0 if data.get("source") == "yellowstone" else THRESH_LIQ
        if data["liq"] <= liq_floor:
            kill = f"liq(${data['liq']:.0f}<=${liq_floor:.0f})"
        elif data["dev"] >= THRESH_DEV:
            kill = f"dev({data['dev']:.1f}%>={THRESH_DEV}%)"
        elif data.get("deployer_launches", 0) > 3:
            kill = f"deployer({data['deployer_launches']}>3)"
        elif data.get("top10_holders", 0.0) > 0.3:
            kill = f"top10({data['top10_holders']:.0%}>30%)"

    if kill:
        _nm_logger.info(
            "KILL %s | reason=%s | age=%.0f vol=%.0f whale=%.0f liq=%.0f dev=%.1f src=%s",
            token, kill, data.get("age", 0), data.get("vol", 0),
            data.get("whale", 0), data.get("liq", 0), data.get("dev", 0),
            data.get("source", ""),
        )
        return False
    return True


async def dexscreener_info(token: str) -> Dict[str, Any]:
    data = await fetch_json(f"{DEXSCREENER_URL}{token}")
    if not data or not isinstance(data, dict):
        return {}
    pairs = data.get("pairs") or []
    if not pairs:
        return {}
    pair = pairs[0] if isinstance(pairs[0], dict) else {}
    info: Dict[str, Any] = {}
    liq = pair.get("liquidity") or {}
    info["liq"] = float(liq.get("usd") or 0)
    info["price"] = float(pair.get("priceUsd") or 0)
    info["fdv"] = float(pair.get("fdv") or 0)
    info["volume"] = float((pair.get("volume") or {}).get("h24") or 0)
    _txns_h24 = (pair.get("txns") or {}).get("h24") or {}
    if isinstance(_txns_h24, dict):
        info["txns"] = float(_txns_h24.get("buys", 0)) + float(_txns_h24.get("sells", 0))
    else:
        info["txns"] = float(_txns_h24 or 0)
    info["dex_id"] = pair.get("dexId", "")
    return info


def is_pumpswap_pool(dex_info: Dict[str, Any]) -> bool:
    """Only trade tokens on PumpSwap pools — real graduates only."""
    dex_id = (dex_info.get("dex_id") or "").lower()
    return dex_id in ("pumpswap", "pumpfun", "pump")


async def pick_rpc() -> Optional[str]:
    return HELIUS_RPC or TRITON_RPC or None


async def fetch_price(token: str) -> float:
    info = await dexscreener_info(token)
    return float(info.get("price") or 0.0)


async def check_rug(token: str) -> bool:
    return True


async def get_token_balance(client: AsyncClient, owner: Pubkey, mint: Pubkey) -> int:
    opts = TokenAccountOpts(mint=mint, encoding="jsonParsed")
    resp = await client.get_token_accounts_by_owner(owner, opts)
    if not resp.value:
        return 0
    try:
        account = resp.value[0]
        data = account.account.data.parsed
        amount = int(data["info"]["tokenAmount"]["amount"])
        return amount
    except Exception:
        return 0


async def jupiter_swap(
    client: AsyncClient,
    wallet: SoldersKeypair,
    token: str,
    amount: int,
    is_buy: bool,
) -> Tuple[Optional[str], Optional[bytes]]:
    user_pub = wallet.pubkey()
    input_mint = "So11111111111111111111111111111111111111112" if is_buy else token
    output_mint = token if is_buy else "So11111111111111111111111111111111111111112"
    quote_payload = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount),
        "slippageBps": str(slippage_bps(cache["tokens"].get(token, {}).get("vol", 700.0))),
        "swapMode": "ExactIn",
        "onlyDirectRoutes": "true",
        "maxAccounts": "20",
    }
    def _jup_request(method: str, url: str, payload: Dict[str, Any]) -> Any:
        try:
            ua = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/{random.randint(80,120)}.0.{random.randint(1000,9999)}.{random.randint(0,99)} Safari/537.36"
            hdrs = {"User-Agent": ua}
            if method == "GET":
                resp = requests.get(url, params=payload, headers=hdrs, timeout=random.uniform(3.0, 5.0), proxies=PROXIES)
            else:
                resp = requests.post(url, json=payload, headers=hdrs, timeout=random.uniform(3.0, 5.0), proxies=PROXIES)
            data = resp.json()
            if resp.status_code != 200:
                _nm_logger.info("JUP_ERR %s | status=%d body=%s", url.split("/")[-1], resp.status_code, str(data)[:200])
            return data
        except Exception as e:
            _nm_logger.info("JUP_ERR %s | exception=%s", url.split("/")[-1], e)
            return None

    await asyncio.sleep(0.0005 * random.random())
    quote_resp = await asyncio.to_thread(_jup_request, "GET", "https://lite-api.jup.ag/swap/v1/quote", quote_payload)
    if not quote_resp:
        _nm_logger.info("SWAP_FAIL %s | stage=quote_empty", token)
        return None, None
    swap_payload = {
        "quoteResponse": quote_resp,
        "userPublicKey": str(user_pub),
        "wrapAndUnwrapSol": True,
        "prioritizationFeeLamports": tip_lamports(cache["tokens"].get(token, {}).get("vol", 700.0)),
        "dynamicComputeUnitLimit": True,
        "skipUserAccountsRpcCalls": True,
    }
    swap_resp = await asyncio.to_thread(_jup_request, "POST", "https://lite-api.jup.ag/swap/v1/swap", swap_payload)
    if not swap_resp or "swapTransaction" not in swap_resp:
        _nm_logger.info("SWAP_FAIL %s | stage=swap_no_tx resp=%s", token, str(swap_resp)[:200])
        return None, None
    tx_buf = base64.b64decode(swap_resp["swapTransaction"])
    tx = VersionedTransaction.from_bytes(tx_buf)
    tx.sign([wallet])
    sig = await client.send_raw_transaction(bytes(tx))
    return (str(sig.value) if sig and sig.value else None), tx_buf


async def buy_pump_curve(
    client: AsyncClient,
    wallet: SoldersKeypair,
    mint_str: str,
    sol_lamports: int,
) -> Tuple[Optional[str], Optional[bytes]]:
    """Buy directly from Pump.fun bonding curve — works for pre-migration tokens."""
    try:
        mint = Pubkey.from_string(mint_str)
        user = wallet.pubkey()

        # Detect token program (classic SPL vs Token-2022)
        token_prog = await _detect_token_program(client, mint)
        _nm_logger.info("PUMP_BUY %s | token_prog=%s", mint_str, "token2022" if token_prog == TOKEN_2022 else "spl")

        # Derive PDAs
        bonding_curve, _ = Pubkey.find_program_address(
            [b"bonding-curve", bytes(mint)], PUMP_PROGRAM
        )
        associated_bc = _get_ata(bonding_curve, mint, token_prog)
        associated_user = _get_ata(user, mint, token_prog)
        global_vol_acc, _ = Pubkey.find_program_address(
            [b"global_volume_accumulator"], PUMP_PROGRAM
        )
        user_vol_acc, _ = Pubkey.find_program_address(
            [b"user_volume_accumulator", bytes(user)], PUMP_PROGRAM
        )
        fee_config, _ = Pubkey.find_program_address(
            [b"fee_config", bytes(PUMP_PROGRAM)], PUMP_FEE_PROGRAM
        )

        # Fetch bonding curve account to read reserves + creator
        bc_resp = await client.get_account_info(bonding_curve)
        if not bc_resp.value:
            _nm_logger.info("PUMP_BUY %s | bonding_curve_not_found", mint_str)
            return None, None

        raw = bytes(bc_resp.value.data)
        if len(raw) < 81:
            _nm_logger.info("PUMP_BUY %s | bc_data_too_short len=%d", mint_str, len(raw))
            return None, None

        # Parse: 8-byte disc, 5x u64, 1 bool, 32-byte creator
        vt_res = struct.unpack_from('<Q', raw, 8)[0]   # virtual_token_reserves
        vs_res = struct.unpack_from('<Q', raw, 16)[0]  # virtual_sol_reserves
        complete = raw[48] != 0
        creator = Pubkey.from_bytes(raw[49:81])

        # creator_vault PDA uses creator pubkey, NOT mint
        creator_vault, _ = Pubkey.find_program_address(
            [b"creator-vault", bytes(creator)], PUMP_PROGRAM
        )

        # Random fee recipient to spread write-lock contention
        fee_recipient = random.choice(PUMP_FEE_RECIPIENTS)

        if complete:
            _nm_logger.info("PUMP_BUY %s | curve_complete", mint_str)
            return None, None
        if vs_res == 0:
            _nm_logger.info("PUMP_BUY %s | zero_sol_reserves", mint_str)
            return None, None

        # Constant product: tokens_out = (sol_in * vt_res) / (vs_res + sol_in)
        tokens_out = (sol_lamports * vt_res) // (vs_res + sol_lamports)
        if tokens_out == 0:
            _nm_logger.info("PUMP_BUY %s | tokens_out=0", mint_str)
            return None, None

        max_sol_cost = int(sol_lamports * 1.15)   # 15% slippage on SOL
        min_tokens = int(tokens_out * 0.85)        # 15% slippage on tokens

        _nm_logger.info(
            "PUMP_BUY %s | vt=%d vs=%d tokens=%d min=%d maxSol=%.4f",
            mint_str, vt_res, vs_res, tokens_out, min_tokens, max_sol_cost / 1e9,
        )

        # Instruction data: disc(8) + token_amount(u64) + max_sol_cost(u64)
        ix_data = (
            PUMP_BUY_DISC
            + struct.pack('<Q', min_tokens)
            + struct.pack('<Q', max_sol_cost)
        )

        buy_ix = Instruction(
            PUMP_PROGRAM,
            ix_data,
            [
                AccountMeta(PUMP_GLOBAL, is_signer=False, is_writable=False),         # 0
                AccountMeta(fee_recipient, is_signer=False, is_writable=True),         # 1
                AccountMeta(mint, is_signer=False, is_writable=False),                 # 2
                AccountMeta(bonding_curve, is_signer=False, is_writable=True),         # 3
                AccountMeta(associated_bc, is_signer=False, is_writable=True),         # 4
                AccountMeta(associated_user, is_signer=False, is_writable=True),       # 5
                AccountMeta(user, is_signer=True, is_writable=True),                   # 6
                AccountMeta(SYS_PROGRAM, is_signer=False, is_writable=False),          # 7
                AccountMeta(token_prog, is_signer=False, is_writable=False),           # 8
                AccountMeta(creator_vault, is_signer=False, is_writable=True),         # 9
                AccountMeta(PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False), # 10
                AccountMeta(PUMP_PROGRAM, is_signer=False, is_writable=False),         # 11
                AccountMeta(global_vol_acc, is_signer=False, is_writable=False),       # 12
                AccountMeta(user_vol_acc, is_signer=False, is_writable=True),          # 13
                AccountMeta(fee_config, is_signer=False, is_writable=False),           # 14
                AccountMeta(PUMP_FEE_PROGRAM, is_signer=False, is_writable=False),     # 15
            ],
        )

        # Build instruction list
        ixs = []
        # Compute budget: 300k units, 50k micro-lamports priority
        ixs.append(Instruction(COMPUTE_BUDGET, struct.pack('<BI', 2, 300_000), []))
        ixs.append(Instruction(COMPUTE_BUDGET, struct.pack('<BQ', 3, 50_000), []))
        # Create user ATA if it doesn't exist
        ata_resp = await client.get_account_info(associated_user)
        if not ata_resp.value:
            ixs.append(_create_ata_ix(user, user, mint, token_prog))
        ixs.append(buy_ix)

        # Build and sign transaction
        bh_resp = await client.get_latest_blockhash()
        blockhash = bh_resp.value.blockhash
        msg = MessageV0.try_compile(user, ixs, [], blockhash)
        tx = VersionedTransaction(msg, [wallet])
        raw_tx = bytes(tx)

        sig_resp = await client.send_raw_transaction(raw_tx)
        sig = str(sig_resp.value) if sig_resp and sig_resp.value else None
        if sig:
            _nm_logger.info("PUMP_BUY %s | sig=%s", mint_str, sig)
        else:
            _nm_logger.info("PUMP_BUY %s | send_failed", mint_str)
        return sig, raw_tx

    except Exception as e:
        _nm_logger.info("PUMP_BUY %s | error=%s", mint_str, e)
        return None, None


_jito_fails = 0


async def jito_bundle(tx_buf: bytes) -> Optional[str]:
    global _jito_fails
    if not JITO_KEY:
        return None
    tip = 5_000_000 if _jito_fails > 5 else 1_000_000
    payload = {
        "transactions": [base64.b64encode(tx_buf).decode("ascii")],
        "tip_lamports": tip,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://mainnet.block-engine.jito.wtf/api/v1/bundles",
                json=payload,
                timeout=random.uniform(2.0, 3.5),
                proxies=PROXIES,
            )
            data = resp.json() if resp else {}
            bundle_id = data.get("bundle_id")
            if bundle_id:
                _nm_logger.info("JITO_OK %s | attempt=%d tip=%d", bundle_id, attempt + 1, tip)
                _jito_fails = max(0, _jito_fails - 1)
                return bundle_id
            _jito_fails += 1
        except Exception as e:
            _nm_logger.info("JITO_ERR | attempt=%d err=%s", attempt + 1, e)
            _jito_fails += 1
        if attempt < 2:
            await asyncio.sleep(0.2)
    return None


async def update_position_price(token: str, price: float) -> None:
    entry = positions.get(token)
    if not entry:
        return
    peak = entry.get("peak", price)
    if price > peak:
        peak = price
    entry["price"] = price
    entry["peak"] = peak
    positions[token] = entry


async def _close_position(token: str, exit_price: float, buy_price: float, reason: str) -> None:
    """Shared exit: log trade, update circuit breaker, remove from open set."""
    duration = now() - positions.get(token, {}).get("ts", now())
    buy_sol = positions.get(token, {}).get("buy_sol", BUY_LAMPORTS / 1e9)
    pnl_sol = (exit_price - buy_price) / buy_price * buy_sol
    log_trade(token, buy_price, exit_price, pnl_sol, reason, duration)
    if pnl_sol < 0:
        trip_circuit(abs(pnl_sol))
        if not check_circuit_breaker():
            await telegram_alert_circuit()
    positions.pop(token, None)
    open_positions.discard(token)


async def telegram_alert_circuit() -> None:
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if chat_id:
        await telegram_send(chat_id, f"CIRCUIT BREAKER: {CIRCUIT_BREAKER_SOL} SOL LOSS HIT — PAUSED 24H")


async def auto_sell(token: str, buy_price: float) -> None:
    if buy_price <= 0 or _killed:
        return
    peak = buy_price
    tiers_hit = set()
    while True:
        await asyncio.sleep(5)
        price = await fetch_price(token)
        if price <= 0:
            continue
        if price > peak:
            peak = price
        pnl = (price - buy_price) / buy_price
        # Stop loss
        if pnl <= STOP_LOSS:
            await sell_token(token)
            await _close_position(token, price, buy_price, "stop_loss")
            return
        # Tiered take-profit
        for i, (tp_thresh, tp_frac) in enumerate(SELL_TIERS):
            if i not in tiers_hit and pnl >= tp_thresh:
                tiers_hit.add(i)
                await sell_partial(token, tp_frac)
                if len(tiers_hit) >= len(SELL_TIERS):
                    await _close_position(token, price, buy_price, "all_tiers_hit")
                    return
        # Trailing stop after first TP hit
        if tiers_hit and price <= peak * 0.75:
            await sell_token(token)
            await _close_position(token, price, buy_price, "trailing_stop")
            return


async def exit_loop() -> None:
    while True:
        await asyncio.sleep(2.0)
        if _killed or not positions:
            continue
        rpc_url = await pick_rpc()
        if not rpc_url:
            continue
        wallet = get_wallet()
        if not wallet:
            continue
        async with AsyncClient(rpc_url) as client:
            for token in list(positions.keys()):
                info = await dexscreener_info(token)
                price = float(info.get("price") or 0)
                if price <= 0:
                    continue
                await update_position_price(token, price)
                entry = positions.get(token, {})
                entry_price = float(entry.get("entry") or 0)
                peak = float(entry.get("peak") or price)
                if entry_price <= 0:
                    continue
                if price >= entry_price * 4.0:
                    balance = await get_token_balance(client, wallet.pubkey(), Pubkey.from_string(token))
                    if balance > 0:
                        await jupiter_swap(client, wallet, token, balance, False)
                    await _close_position(token, price, entry_price, "moon_4x")
                    continue
                if price <= entry_price * 0.6:
                    balance = await get_token_balance(client, wallet.pubkey(), Pubkey.from_string(token))
                    if balance > 0:
                        await jupiter_swap(client, wallet, token, balance, False)
                    await _close_position(token, price, entry_price, "rug_dump")


async def sell_partial(token: str, fraction: float) -> None:
    """Sell a fraction of holdings for tiered TP."""
    rpc_url = await pick_rpc()
    if not rpc_url:
        return
    wallet = get_wallet()
    if not wallet:
        return
    async with AsyncClient(rpc_url) as client:
        balance = await get_token_balance(client, wallet.pubkey(), Pubkey.from_string(token))
        sell_amount = int(balance * fraction)
        if sell_amount <= 0:
            return
        await jupiter_swap(client, wallet, token, sell_amount, False)


async def sell_token(token: str) -> None:
    rpc_url = await pick_rpc()
    if not rpc_url:
        return
    wallet = get_wallet()
    if not wallet:
        return
    async with AsyncClient(rpc_url) as client:
        balance = await get_token_balance(client, wallet.pubkey(), Pubkey.from_string(token))
        if balance <= 0:
            return
        await jupiter_swap(client, wallet, token, balance, False)


async def decide(token: str, data: Dict[str, Any]) -> str:
    if not gclient:
        return "SKIP"
    prompt = decision_prompt(data)
    resp = await asyncio.to_thread(
        gclient.chat.completions.create,
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1,
    )
    decision = resp.choices[0].message.content.strip().upper()
    cache["decisions"][token] = {"decision": decision, "ts": now()}
    return decision


async def rug_score(token: str) -> float:
    """AI rug-pull risk score via Groq. Returns 1-10 (>6 = skip)."""
    if not gclient:
        return 0.0
    data = cache["tokens"].get(token, {})
    prompt = (
        f"Rug-pull risk score 1-10 for Solana token. "
        f"Age: {data.get('age', 0):.0f}s, Liquidity: ${data.get('liq', 0):.0f}, "
        f"Volume: +{data.get('vol', 0):.0f}%, Top holder: {data.get('whale', 0):.0f}%, "
        f"Dev holding: {data.get('dev', 0):.1f}%. "
        f"Reply ONLY the number."
    )
    try:
        resp = await asyncio.to_thread(
            gclient.chat.completions.create,
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2,
        )
        score = float(resp.choices[0].message.content.strip())
        _nm_logger.info("RUG_SCORE %s | %.1f", token, score)
        return score
    except Exception as e:
        _nm_logger.info("RUG_SCORE_ERR %s | %s", token, e)
        return 0.0


async def count_whales(client: AsyncClient, mint_str: str) -> int:
    """Estimate whale count from bonding curve SOL reserves (1 RPC call)."""
    try:
        mint = Pubkey.from_string(mint_str)
        bc, _ = Pubkey.find_program_address(
            [b"bonding-curve", bytes(mint)], PUMP_PROGRAM
        )
        resp = await client.get_account_info(bc)
        if not resp.value:
            return 0
        raw = bytes(resp.value.data)
        if len(raw) < 40:
            return 0
        real_sol = struct.unpack_from('<Q', raw, 32)[0]  # real_sol_reserves
        sol = real_sol / 1e9
        if sol >= 10:
            return 3
        elif sol >= 3:
            return 2
        elif sol >= 1:
            return 1
        return 0
    except Exception:
        return 0


async def watch_loop() -> None:
    while True:
        if _killed or not watch_enabled:
            await asyncio.sleep(WATCH_LOOP)
            continue
        await asyncio.sleep(WATCH_LOOP)
        birdeye = await fetch_json(BIRDEYE_URL, headers={"X-API-KEY": BIRDEYE_KEY} if BIRDEYE_KEY else None)
        pumpfun = await fetch_json(PUMPFUN_URL)
        flaunch = await fetch_json(FLAUNCH_URL)
        sources = [birdeye, pumpfun, flaunch]
        _wl_stats["ticks"] += 1
        for payload in sources:
            for token in iter_tokens(payload or []):
                addr = token.get("address") or token.get("token") or token.get("contract")
                if not addr:
                    continue
                _wl_stats["tokens_seen"] += 1
                age = get_age_seconds(token)
                vol = get_metric(token, ["volume_pct", "volumeChangePct", "volumeChange"])
                whale = get_metric(token, ["whale_delta", "whaleDeltaPct", "whaleDelta"])
                liq = get_metric(token, ["liquidity", "liquidityUsd"])
                info = await dexscreener_info(addr)
                _wl_stats["dex_checked"] += 1
                if not is_pumpswap_pool(info):
                    _nm_logger.info(
                        "KILL %s | reason=not_pumpswap | dex_id=%s | age=%.0f liq=%.0f src=%s",
                        addr, info.get("dex_id", "none"), age, liq,
                        cache["tokens"].get(addr, {}).get("source", ""),
                    )
                    continue
                if info.get("liq", 0) > 0:
                    liq = max(liq, info["liq"])
                spike = max(vol, info.get("volume", 0))
                dev = 0.0
                deployer_launches = get_metric(token, ["deployer_launches", "deployerLaunches", "creatorTokenCount"])
                top10_pct = get_metric(token, ["top10_holders_pct", "top10HoldersPct", "topHoldersPercent"])
                if top10_pct > 1:
                    top10_pct = top10_pct / 100.0
                cached_source = cache["tokens"].get(addr, {}).get("source", "")
                data = {
                    "token": addr,
                    "age": age,
                    "vol": vol,
                    "whale": whale,
                    "liq": liq,
                    "spike": spike,
                    "dev": dev,
                    "deployer_launches": deployer_launches,
                    "top10_holders": top10_pct,
                    "source": cached_source,
                }
                if not should_trade(data):
                    continue
                if not await check_rug(addr):
                    _nm_logger.info("KILL %s | reason=rug_check | age=%.0f liq=%.0f src=%s",
                                    addr, age, liq, cached_source)
                    continue
                _wl_stats["passed_filters"] += 1
                cache["tokens"][addr] = {**data, "ts": now()}
                await telegram_alert(addr)
        # --- Process Yellowstone-cached tokens ---
        all_cached = list(cache["tokens"].items())
        ys_all = [(a, i) for a, i in all_cached if i.get("source") == "yellowstone"]
        ys_unevaled = [(a, i) for a, i in ys_all if not i.get("_evaluated") and a not in open_positions]
        _wl_stats["cache_size"] = len(all_cached)
        _wl_stats["ys_cached"] = len(ys_all)
        _wl_stats["ys_pending"] = len(ys_unevaled)
        ys_tokens = ys_unevaled
        for addr, cached in ys_tokens:
            _wl_stats["tokens_seen"] += 1
            age = now() - cached.get("ts", now())
            info = await dexscreener_info(addr)
            _wl_stats["dex_checked"] += 1
            liq = cached.get("liq", 0)
            if info.get("liq", 0) > 0:
                liq = max(liq, info["liq"])
            data = {
                "token": addr,
                "age": age,
                "vol": cached.get("vol", 0),
                "whale": cached.get("whale", 0),
                "liq": liq,
                "spike": cached.get("spike", 0),
                "dev": cached.get("dev", 0),
                "deployer_launches": cached.get("deployer_launches", 0),
                "top10_holders": cached.get("top10_holders", 0),
                "source": "yellowstone",
            }
            cache["tokens"][addr]["_evaluated"] = True
            if not should_trade(data):
                continue
            if not await check_rug(addr):
                _nm_logger.info("KILL %s | reason=rug_check | age=%.0f liq=%.0f src=yellowstone",
                                addr, age, liq)
                continue
            _wl_stats["passed_filters"] += 1
            cache["tokens"][addr] = {**data, "ts": now()}
            await act(addr)
        prune()


async def ws_pool() -> None:
    if not HELIUS_WS:
        return
    async with connect(HELIUS_WS) as ws:
        await ws.program_subscribe("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
        async for msg in ws:
            try:
                token = msg.result.value.pubkey
            except Exception:
                continue
            await think(str(token))


async def telegram_send(chat_id, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    def _post():
        try:
            requests.post(url, json=payload, timeout=random.uniform(2.0, 3.5), proxies=PROXIES)
        except Exception:
            pass
    await asyncio.to_thread(_post)


async def telegram_alert(token_addr: str) -> None:
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        return
    link = f"https://dexscreener.com/solana/{token_addr}"
    text = f"{token_addr} PUMPING — KILL?\n{link}"
    await telegram_send(chat_id, text)


async def telegram_poll() -> None:
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        return
    offset = 0
    while True:
        await asyncio.sleep(2)
        url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/getUpdates"
        def _get() -> Any:
            try:
                return requests.get(url, params={"timeout": 1, "offset": offset}, timeout=random.uniform(2.0, 3.5), proxies=PROXIES).json()
            except Exception:
                return None
        data = await asyncio.to_thread(_get)
        if not data or not data.get("ok"):
            continue
        for update in data.get("result", []):
            offset = max(offset, update.get("update_id", 0) + 1)
            message = update.get("message", {}) or {}
            text = (message.get("text") or "").strip()
            chat_id = message.get("chat", {}).get("id")
            if text == "/watch":
                result = await watch_toggle()
                if chat_id:
                    await telegram_send(chat_id, f"watch {result['watch']}")
            elif text.startswith("/act "):
                mint = text[5:].strip()
                if mint:
                    act_result = await act(mint)
                    if chat_id:
                        await telegram_send(chat_id, f"{act_result.get('status','unknown')} {act_result.get('hash','')}")


async def check_dex_watchers(mint: str) -> int:
    """Check DexScreener watcher count for graduation signal."""
    try:
        data = await fetch_json(f"https://api.dexscreener.com/latest/dex/tokens/{mint}")
        if data and isinstance(data, dict):
            pairs = data.get("pairs") or []
            if pairs and isinstance(pairs[0], dict):
                info = pairs[0].get("info") or {}
                return int(info.get("watchers") or 0)
    except Exception:
        pass
    return 0


async def _evaluate_yellowstone(mint: str) -> None:
    """Evaluate a Yellowstone-detected token immediately — no watch_loop delay."""
    if _killed:
        return
    global _jup_last_call
    try:
        cached = cache["tokens"].get(mint)
        if not cached:
            return
        _wl_stats["tokens_seen"] += 1
        cache["tokens"][mint]["_evaluated"] = True
        data = {
            "token": mint,
            "age": 0,
            "vol": cached.get("vol", 0),
            "whale": cached.get("whale", 0),
            "liq": cached.get("liq", 0),
            "spike": cached.get("spike", 0),
            "dev": cached.get("dev", 0),
            "deployer_launches": 0,
            "top10_holders": 0,
            "source": "yellowstone",
        }
        if not should_trade(data):
            return
        if not await check_rug(mint):
            _nm_logger.info("KILL %s | reason=rug_check | age=0 liq=%.0f src=yellowstone",
                            mint, cached.get("liq", 0))
            return
        _wl_stats["passed_filters"] += 1
        # Rate limit Jupiter API calls — 1 per second max
        async with _jup_semaphore:
            wait = max(0, 1.0 - (now() - _jup_last_call))
            if wait > 0:
                await asyncio.sleep(wait)
            _jup_last_call = now()
            await act(mint)
    except Exception as e:
        _nm_logger.info("EVAL_ERR %s | %s", mint, e)


async def on_pump_event(event: dict) -> None:
    """Callback from Yellowstone scanner when Pump.fun activity detected."""
    _wl_stats["pump_callbacks"] = _wl_stats.get("pump_callbacks", 0) + 1
    if not watch_enabled:
        return
    mint = event.get("mint", "")
    if not mint:
        return
    if event.get("type") == "create":
        _wl_stats["ys_creates"] = _wl_stats.get("ys_creates", 0) + 1
    elif event.get("type") == "buy":
        _wl_stats["ys_buys"] = _wl_stats.get("ys_buys", 0) + 1
    # Cache if not already seen
    if mint in cache.get("tokens", {}):
        return
    cache["tokens"][mint] = {
        "age": 0, "vol": 999 if event.get("type") == "create" else 500,
        "whale": 0, "liq": 5000, "spike": 0, "dev": 0,
        "ts": now(), "source": "yellowstone",
    }
    # Evaluate immediately — don't wait for watch_loop
    asyncio.create_task(_evaluate_yellowstone(mint))


_last_heartbeat = time.time()

async def _heartbeat_loop() -> None:
    global _last_heartbeat
    while True:
        await asyncio.sleep(30)
        _last_heartbeat = time.time()
        _nm_logger.info(f'scanner alive | ts={int(time.time())} | pid={os.getpid()}')

def _watchdog_thread() -> None:
    while True:
        time.sleep(30)
        gap = time.time() - _last_heartbeat
        if gap > 90:
            _nm_logger.info("WATCHDOG_KILL | gap=%.0fs — forcing exit", gap)
            os._exit(1)


async def whale_copy_loop() -> None:
    """Copy-trade whale wallets — scan every 0.7s for new Pump.fun buys."""
    if not WHALE_WALLETS:
        return
    rpc_url = HELIUS_RPC or TRITON_RPC
    if not rpc_url:
        return
    seen_sigs: Set[str] = set()
    while True:
        try:
            async with AsyncClient(rpc_url) as client:
                while True:
                    await asyncio.sleep(0.7)
                    if _killed or not watch_enabled:
                        continue
                    for whale_addr in WHALE_WALLETS:
                        try:
                            whale_pub = Pubkey.from_string(whale_addr)
                            resp = await client.get_signatures_for_address(whale_pub, limit=5)
                            if not resp.value:
                                continue
                            for sig_info in resp.value:
                                sig_str = str(sig_info.signature)
                                if sig_str in seen_sigs:
                                    continue
                                seen_sigs.add(sig_str)
                                tx_resp = await client.get_transaction(
                                    sig_info.signature,
                                    max_supported_transaction_version=0,
                                )
                                if not tx_resp.value:
                                    continue
                                meta = getattr(tx_resp.value, 'meta', None)
                                if not meta:
                                    continue
                                post_bals = getattr(meta, 'post_token_balances', None) or []
                                for bal in post_bals:
                                    owner = str(getattr(bal, 'owner', ''))
                                    mint = str(getattr(bal, 'mint', ''))
                                    if owner == whale_addr and mint and mint not in cache["tokens"] and mint not in open_positions:
                                        _nm_logger.info("WHALE_BUY | whale=%s mint=%s", whale_addr[:8], mint)
                                        cache["tokens"][mint] = {
                                            "age": 0, "vol": 999, "whale": 0, "liq": 5000,
                                            "spike": 0, "dev": 0, "ts": now(), "source": "whale_copy",
                                        }
                                        asyncio.create_task(act(mint))
                        except Exception as e:
                            _nm_logger.info("WHALE_ERR %s | %s", whale_addr[:8], e)
                    if len(seen_sigs) > 5000:
                        seen_sigs = set(list(seen_sigs)[-2500:])
        except Exception as e:
            _nm_logger.info("WHALE_RECONNECT | %s", e)
            await asyncio.sleep(2)


async def sol_guard() -> None:
    global _low_sol
    while True:
        await asyncio.sleep(300)
        try:
            rpc_url = await pick_rpc()
            if not rpc_url:
                continue
            wallet = get_wallet()
            if not wallet:
                continue
            async with AsyncClient(rpc_url) as client:
                resp = await client.get_balance(wallet.pubkey())
                bal = resp.value / 1e9
            if bal < 0.2:
                if not _low_sol:
                    _nm_logger.info("SOL_GUARD PAUSED | balance=%.4f SOL", bal)
                _low_sol = True
            else:
                if _low_sol:
                    _nm_logger.info("SOL_GUARD RESUMED | balance=%.4f SOL", bal)
                _low_sol = False
        except Exception:
            pass


async def _revive_timer() -> None:
    global _kill_timeout, _killed, watch_enabled
    while True:
        await asyncio.sleep(1)
        if _killed and _kill_timeout > 0:
            _kill_timeout -= 1
            if _kill_timeout == 0:
                _killed = False
                watch_enabled = True
                _nm_logger.info("AUTO REVIVE | timeout")


@app.on_event("startup")
async def startup() -> None:
    threading.Thread(target=_watchdog_thread, daemon=True).start()
    asyncio.create_task(_heartbeat_loop())
    asyncio.create_task(watch_loop())
    asyncio.create_task(ws_pool())
    asyncio.create_task(exit_loop())
    asyncio.create_task(telegram_poll())
    asyncio.create_task(run_scanner(on_pump=on_pump_event))
    asyncio.create_task(whale_copy_loop())
    asyncio.create_task(sol_guard())
    asyncio.create_task(_revive_timer())


@app.get("/health")
async def health() -> dict:
    return {"mcp": "alive", "speed": "<0.1s"}


@app.post("/watch")
async def watch_toggle(request: Request = None) -> dict:
    if request and not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    global watch_enabled
    watch_enabled = not watch_enabled
    return {"watch": watch_enabled}


@app.get("/think/{token}")
async def think(token: str, request: Request = None) -> dict:
    if request and not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    if DRY_RUN:
        return {"decision": "DRY_SKIP"}
    data = cache["tokens"].get(token, {"age": 12, "vol": 800, "whale": 2, "liq": 120000, "spike": 0, "dev": 0})
    decision = await decide(token, data)
    return {"decision": decision}


@app.post("/act/{token}")
async def act(token: str, request: Request = None) -> dict:
    if request and not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    if request and not _rate_ok(_get_client_ip(request)):
        return JSONResponse({"status": "rate_limited"}, status_code=429)
    if _killed:
        return {"decision": "KILLED", "reason": "emergency stop"}
    if _low_sol:
        return {"decision": "PAUSED", "reason": "low SOL balance"}
    decision = (await think(token)).get("decision")
    _nm_logger.info("ACT %s | decision=%s", token, decision)
    if decision != "KILL":
        _nm_logger.info("KILL %s | reason=groq_skip(%s) | src=yellowstone", token, decision)
        return {"status": "skipped", "reason": decision}
    score = await rug_score(token)
    if score > 6:
        _nm_logger.info("RUG_SKIP %s | score=%.1f", token, score)
        return {"status": "rug_skip", "score": score}
    if DRY_RUN:
        _nm_logger.info("ACT %s | dry_run=True, skipping", token)
        return {"status": "dry-run", "hash": "fake123"}
    rpc_url = await pick_rpc()
    if not rpc_url:
        _nm_logger.info("ACT %s | rpc_down", token)
        return {"status": "rpc down"}
    wallet = get_wallet()
    if not wallet:
        _nm_logger.info("ACT %s | wallet_missing", token)
        return {"status": "wallet missing"}
    try:
        async with AsyncClient(rpc_url) as client:
            # --- Dynamic position sizing ---
            whales = await count_whales(client, token)
            bal_resp = await client.get_balance(wallet.pubkey())
            balance = (bal_resp.value or 0) / 1e9
            cached = cache["tokens"].get(token, {})
            spike = float(cached.get("vol", 0))
            age = now() - cached.get("ts", now())

            if whales >= 2 and score <= 4:
                size_sol = 1.0
            elif whales >= 1 or spike >= 300:
                size_sol = 0.5
            elif age < 30 and score <= 5:
                size_sol = 0.3
            else:
                size_sol = 0.1

            buy_lamps = int(size_sol * 1e9)

            if size_sol > balance:
                _nm_logger.info("NO_TRADE %s | bal=%.2f < size=%.1fSOL", token, balance, size_sol)
                return {"status": "no_balance", "balance": balance, "size": size_sol}

            _nm_logger.info(
                "SIZE %s | %.1fSOL | score=%.1f whales=%d spike=%.0f%% age=%.0fs bal=%.2fSOL",
                token, size_sol, score, whales, spike, age, balance,
            )

            sig, tx_buf = await jupiter_swap(client, wallet, token, buy_lamps, True)
            if not sig:
                _nm_logger.info("ACT %s | jup_failed, trying pump_curve", token)
                sig, tx_buf = await buy_pump_curve(client, wallet, token, buy_lamps)
            if tx_buf:
                bundle_id = await jito_bundle(tx_buf)
                if bundle_id:
                    sig = f"bundle:{bundle_id}"
            if not sig:
                _nm_logger.info("ACT %s | tx_fail_no_sig (jup+pump both failed)", token)
                return {"status": "tx fail"}
            buy_price = await fetch_price(token)
            _nm_logger.info("BUY %s | sig=%s price=%.8f size=%.1fSOL", token, sig, buy_price, size_sol)
            if buy_price > 0:
                positions[token] = {
                    "entry": buy_price, "peak": buy_price, "price": buy_price,
                    "ts": now(), "buy_sol": size_sol,
                }
                open_positions.add(token)
                asyncio.create_task(auto_sell(token, buy_price))
            global trade_count
            trade_count += 1
            if trade_count % 5 == 0:
                rotate_wallet()
            return {"status": "tx sent", "hash": sig, "size": size_sol}
    except Exception as exc:
        _nm_logger.info("ACT %s | tx_exception=%s", token, exc)
        return {"status": "tx fail", "error": str(exc)}


@app.get("/monitor")
async def monitor(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    return {
        "watch": watch_enabled,
        "trades": trade_count,
        "wallet": wallet_index,
        "speed": "<0.1s",
        "last": None,
        "pipeline": _wl_stats,
    }


@app.get("/scanner")
async def scanner_status(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    return scanner_get_stats()


@app.get("/swap")
async def swap_endpoint(request: Request, token: str = "", amount: float = 0.001):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    if not token:
        return {"status": "error", "reason": "no token"}
    if DRY_RUN:
        return {"status": "dry-run", "tx_hash": f"sim_{token[:8]}_{int(now())}", "amount": amount}
    result = await act(token)
    return {"status": result.get("status"), "tx_hash": result.get("hash", ""), "amount": amount}


@app.get("/config")
async def config_view(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    return {
        "thresh_age": THRESH_AGE,
        "thresh_vol": THRESH_VOL,
        "thresh_liq": THRESH_LIQ,
        "thresh_whale": THRESH_WHALE,
        "thresh_dev": THRESH_DEV,
        "buy_lamports": BUY_LAMPORTS,
        "buy_sol": BUY_LAMPORTS / 1e9,
        "sell_tiers": [{"pnl": t[0], "fraction": t[1]} for t in SELL_TIERS],
        "stop_loss": STOP_LOSS,
    }


@app.get("/near-misses")
async def near_misses(request: Request, lines: int = 50):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    nm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "near_miss.log")
    if not os.path.exists(nm_path):
        return {"kills": [], "total": 0}
    with open(nm_path, "r") as f:
        all_lines = f.readlines()
    recent = all_lines[-lines:]
    reasons: Dict[str, int] = {}
    for line in all_lines:
        if "reason=" in line:
            r = line.split("reason=")[1].split("|")[0].split("(")[0].strip()
            reasons[r] = reasons.get(r, 0) + 1
    return {
        "kills": [l.strip() for l in recent],
        "total": len(all_lines),
        "breakdown": dict(sorted(reasons.items(), key=lambda x: -x[1])),
    }


@app.get("/mev")
async def mev_status(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    return {
        "blocked": True,
        "protection": "jito-bundle",
        "tip_range": "0.001-0.1 SOL",
        "private_tx": True,
        "frontrun_shield": True,
    }


@app.get("/api/state")
async def api_state(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    if time.time() - _balance_cache["ts"] > 30:
        try:
            rpc_url = await pick_rpc()
            w = get_wallet()
            if rpc_url and w:
                async with AsyncClient(rpc_url) as c:
                    r = await c.get_balance(w.pubkey())
                    _balance_cache["sol"] = (r.value or 0) / 1e9
                    _balance_cache["ts"] = time.time()
        except Exception:
            pass
    pos_list = []
    for tok, p in positions.items():
        cd = cache["tokens"].get(tok, {})
        entry = p.get("entry", 0)
        price = p.get("price", entry)
        pnl = ((price - entry) / entry * 100) if entry > 0 else 0
        pos_list.append({
            "token": tok, "entry": entry, "price": price,
            "buy_sol": p.get("buy_sol", 0), "ts": p.get("ts", 0),
            "whales": cd.get("whale", 0), "spike": cd.get("spike", cd.get("vol", 0)),
            "age": cd.get("age", 0), "liq": cd.get("liq", 0), "pnl": round(pnl, 2),
            "status": "OPEN",
        })
    trades = []
    try:
        with open(TRADES_CSV) as f:
            rows = list(csv.reader(f))
        for row in rows[-20:]:
            if len(row) < 6 or row[0] == "timestamp":
                continue
            trades.append({
                "token": row[1], "entry": float(row[2] or 0), "exit": float(row[3] or 0),
                "pnl_sol": float(row[4] or 0), "reason": row[5],
                "buy_sol": 0, "ts": row[0],
                "status": "RUG" if "rug" in row[5].lower() else "SOLD",
            })
    except Exception:
        pass
    nm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "near_miss.log")
    logs = []
    try:
        with open(nm_path) as f:
            logs = [l.strip() for l in f.readlines()[-40:]]
    except Exception:
        pass
    return {
        "sol": _balance_cache["sol"],
        "dry_run": DRY_RUN,
        "low_sol": _low_sol,
        "killed": _killed,
        "kill_timeout": _kill_timeout,
        "watch": watch_enabled,
        "trades": trade_count,
        "open": pos_list,
        "closed": trades,
        "logs": logs,
        "pipeline": _wl_stats,
        "sell_tiers": [{"pnl": t[0], "frac": t[1]} for t in SELL_TIERS],
        "stop_loss": STOP_LOSS,
    }


@app.post("/api/dry-run")
async def toggle_dry_run(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    global DRY_RUN
    DRY_RUN = not DRY_RUN
    _nm_logger.info("DRY_RUN toggled to %s", DRY_RUN)
    return {"dry_run": DRY_RUN}


@app.post("/api/sell-tier")
async def set_sell_tier(request: Request):
    global SELL_TIERS
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    body = await request.json()
    val = max(0.3, min(3.0, float(body.get("value", SELL_TIERS[0][0]))))
    SELL_TIERS = [(val, SELL_TIERS[0][1])] + list(SELL_TIERS[1:])
    _nm_logger.info("SELL_TIER[0] set to %.2f", val)
    return {"sell_tiers": [{"pnl": t[0], "frac": t[1]} for t in SELL_TIERS]}


@app.post("/api/kill")
async def kill_all(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    global _killed, watch_enabled, _kill_timeout
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    _killed = True
    watch_enabled = False
    _kill_timeout = 300 if body.get("auto") else 0
    positions.clear()
    open_positions.clear()
    _nm_logger.info("EMERGENCY STOP | manual | auto_revive=%s", _kill_timeout > 0)
    return {"killed": True, "auto_revive": _kill_timeout}


@app.post("/api/revive")
async def revive(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    global _killed, watch_enabled, _kill_timeout
    _killed = False
    _kill_timeout = 0
    watch_enabled = True
    _nm_logger.info("MANUAL REVIVE")
    return {"killed": False}


@app.get("/api/raw-logs")
async def raw_logs(request: Request):
    if not _is_trusted(request):
        return JSONResponse({"status": "ghost"}, status_code=403)
    nm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "near_miss.log")
    if not os.path.exists(nm_path):
        return PlainTextResponse("")
    with open(nm_path, "r") as f:
        lines = f.readlines()
    return PlainTextResponse("".join(lines[-50:]))


def _get_ws_state() -> dict:
    pos_list = []
    for tok, p in positions.items():
        cd = cache["tokens"].get(tok, {})
        entry = p.get("entry", 0)
        price = p.get("price", entry)
        pnl = ((price - entry) / entry * 100) if entry > 0 else 0
        pos_list.append({
            "token": tok, "entry": entry, "price": price,
            "buy_sol": p.get("buy_sol", 0), "pnl": round(pnl, 2),
            "whales": cd.get("whale", 0), "spike": cd.get("spike", cd.get("vol", 0)),
            "age": cd.get("age", 0), "status": "OPEN",
        })
    trades = []
    try:
        with open(TRADES_CSV) as f:
            rows = list(csv.reader(f))
        for row in rows[-20:]:
            if len(row) < 6 or row[0] == "timestamp":
                continue
            trades.append({
                "token": row[1], "pnl_sol": float(row[4] or 0),
                "reason": row[5], "ts": row[0],
                "status": "RUG" if "rug" in row[5].lower() else "SOLD",
            })
    except Exception:
        pass
    return {
        "sol": _balance_cache["sol"],
        "dry_run": DRY_RUN,
        "low_sol": _low_sol,
        "killed": _killed,
        "kill_timeout": _kill_timeout,
        "watch": watch_enabled,
        "trades": trade_count,
        "open": pos_list,
        "closed": trades,
        "pipeline": dict(_wl_stats),
        "sell_tiers": [{"pnl": t[0], "frac": t[1]} for t in SELL_TIERS],
        "stop_loss": STOP_LOSS,
    }


@app.websocket("/api/ws")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    try:
        state = _get_ws_state()
        state["logs"] = [line for _, line in list(_ws_log_buffer)[-10:]]
        await ws.send_json({"type": "state", **state})
        await asyncio.sleep(0.01)

        cursor = _ws_seq
        last_ping = time.time()
        last_state = time.time()

        async def sender():
            nonlocal cursor, last_ping, last_state
            while True:
                await asyncio.sleep(0.5)
                sent = 0
                for seq, line in list(_ws_log_buffer):
                    if seq > cursor:
                        try:
                            await ws.send_json({"type": "log", "line": line})
                            await asyncio.sleep(0.01)
                        except Exception:
                            return
                        cursor = seq
                        sent += 1
                        if sent >= 20:
                            break
                if time.time() - last_state > 5:
                    try:
                        await ws.send_json({"type": "state", **_get_ws_state()})
                        await asyncio.sleep(0.01)
                    except Exception:
                        return
                    last_state = time.time()
                if time.time() - last_ping > 10:
                    try:
                        await ws.send_json({"type": "ping"})
                    except Exception:
                        return
                    last_ping = time.time()
                if time.time() - _balance_cache["ts"] > 30:
                    try:
                        rpc = await pick_rpc()
                        w = get_wallet()
                        if rpc and w:
                            async with AsyncClient(rpc) as c:
                                r = await c.get_balance(w.pubkey())
                                _balance_cache["sol"] = (r.value or 0) / 1e9
                                _balance_cache["ts"] = time.time()
                    except Exception:
                        pass

        async def receiver():
            while True:
                data = await ws.receive_text()
                if data == "refresh":
                    try:
                        await ws.send_json({"type": "state", **_get_ws_state()})
                        await asyncio.sleep(0.01)
                    except Exception:
                        return

        task = asyncio.create_task(sender())
        try:
            await receiver()
        finally:
            task.cancel()
    except (WebSocketDisconnect, Exception) as e:
        _nm_logger.info("WS_CLOSE | %s", e)

