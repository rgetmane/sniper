package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	bin "github.com/gagliardetto/binary"
	"github.com/gagliardetto/solana-go"
	"github.com/gagliardetto/solana-go/programs/system"
	"github.com/gagliardetto/solana-go/rpc"
	"github.com/gagliardetto/solana-go/rpc/ws"
	"github.com/joho/godotenv"
	_ "github.com/mattn/go-sqlite3"
)

// ============================================================================
// TYPES
// ============================================================================

type Token struct {
	ID        int
	Mint      string
	Score     float64
	Bought    bool
	Sold      bool
	TxBuy     string
	TxSell    string
	BuyPrice  float64
	SellPrice float64
	PnL       float64
	BuyTime   time.Time
	SellTime  time.Time
	Reason    string
}

type RugCheckResponse struct {
	Score            float64          `json:"score"`
	Risks            []Risk           `json:"risks"`
	TokenMeta        TokenMeta        `json:"tokenMeta"`
	TopHolders       []Holder         `json:"topHolders"`
	Markets          []Market         `json:"markets"`
	Freezable        bool             `json:"freezeAuthority"`
	Mintable         bool             `json:"mintAuthority"`
	RiskLevel        string           `json:"riskLevel"`
	TrustScore       TrustScore       `json:"trustScore"`
	Scams            []Scam           `json:"scams"`
	LiquidityDetails LiquidityDetails `json:"liquidityDetails"`
}

type Risk struct {
	Name        string `json:"name"`
	Level       string `json:"level"`
	Description string `json:"description"`
	Severity    string `json:"severity"`
}

type TokenMeta struct {
	Name   string `json:"name"`
	Symbol string `json:"symbol"`
}

type Holder struct {
	Address string  `json:"address"`
	Pct     float64 `json:"pct"`
}

type Market struct {
	MarketType   string          `json:"marketType"`
	LiquidityRaw json.RawMessage `json:"lp"`
}

// Liquidity safely parses the lp field which may be float64 or nested object
func (m Market) Liquidity() float64 {
	if len(m.LiquidityRaw) == 0 {
		return 0
	}
	var f float64
	if json.Unmarshal(m.LiquidityRaw, &f) == nil {
		return f
	}
	var obj map[string]interface{}
	if json.Unmarshal(m.LiquidityRaw, &obj) == nil {
		if v, ok := obj["usd"].(float64); ok {
			return v
		}
		for _, v := range obj {
			if fv, ok := v.(float64); ok {
				return fv
			}
		}
	}
	return 0
}

type TrustScore struct {
	Value float64 `json:"value"`
}

type Scam struct {
	Type     string `json:"type"`
	Severity string `json:"severity"`
}

type LiquidityDetails struct {
	LiquidityLocked float64 `json:"liquidityLocked"`
	TotalLiquidity  float64 `json:"totalLiquidity"`
}

// Enhanced safety check result
type SafetyCheckResult struct {
	Safe        bool
	Reason      string
	Details     string
	SlippageBps int     // volume-tier slippage override (0 = use default)
	BuySizeSOL  float64 // volume-tier buy size override (0 = use config)
}

// RPC mint account data
type MintAccountData struct {
	MintAuthority   *string `json:"mintAuthority"`
	FreezeAuthority *string `json:"freezeAuthority"`
}

type Stats struct {
	TotalBuys      int64
	TotalSells     int64
	SuccessfulBuys int64
	FailedBuys     int64
	ProfitTrades   int64
	LossTrades     int64
	TotalPnL       float64
	StartTime      time.Time
}

// Memory system types
type SnipeRecord struct {
	Token      string    `json:"token"`
	BuyPrice   float64   `json:"buy_price"`
	SellPrice  float64   `json:"sell_price"`
	ProfitPct  float64   `json:"profit_pct"`
	Timestamp  time.Time `json:"timestamp"`
}

type MemoryStats struct {
	TotalWins   int     `json:"total_wins"`
	TotalLosses int     `json:"total_losses"`
	BestDayPct  float64 `json:"best_day_pct"`
}

type BotMemory struct {
	Snipes []SnipeRecord  `json:"snipes"`
	Rugs   map[string]bool `json:"rugs"`
	Stats  MemoryStats    `json:"stats"`
	mu     sync.RWMutex
}

type Config struct {
	MaxBuySOL       float64
	ProfitTargetPct float64
	StopLossPct     float64
	TrailDropPct    float64
	MinRugScore     float64
	MaxTopHolderPct float64
	MinLiquidity    float64
	BuyCooldownSec  int
	PositionTimeout int
	SimulateFirst   bool
	JitoTipLamports uint64
	JitoEndpoint    string
	MaxFailedBuys   int
	ProgramWhitelist map[string]bool
}

// ============================================================================
// GLOBALS
// ============================================================================

var (
	db            *sql.DB
	wallets       []solana.PrivateKey
	walletIdx     int
	tradeCount    int
	rpcClient     *rpc.Client
	wsClient      *ws.Client
	jupiterAPIKey string

	pumpID      = solana.MustPublicKeyFromBase58("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
	jitoTipAcct = solana.MustPublicKeyFromBase58("96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5")
	solMint     = "So11111111111111111111111111111111111111112"

	config      Config
	stats       Stats
	mu          sync.Mutex
	statsMu     sync.RWMutex
	lastBuyTime time.Time
	httpClient  = &http.Client{Timeout: 20 * time.Second}
	logFile     *os.File

	blacklist    = make(map[string]bool)
	blacklistMu  sync.RWMutex
	activePos    = make(map[string]bool)
	activePosMu  sync.RWMutex
	wsReconnects int64
	memory       *BotMemory
)

// ============================================================================
// MAIN
// ============================================================================

func main() {
	rand.Seed(time.Now().UnixNano())
	_ = godotenv.Load("secrets.env")
	_ = godotenv.Load()

	initLogging()
	loadConfig()
	loadWallets()
	initRPC()
	initDatabase()
	loadBlacklist()
	loadMemory()

	stats.StartTime = time.Now()

	logConfig()
	
	// Send startup notification to Telegram
	sendTelegram("🟢 Sniper alive\n\n✅ Bot started successfully\n⏰ " + time.Now().Format("2006-01-02 15:04:05") + " UTC")

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	defer cleanup()

	go monitorPump(ctx)
	go statsReporter(ctx)
	go healthCheck(ctx)

	<-ctx.Done()
	log.Println("shutting down gracefully...")
	printFinalStats()
}

// ============================================================================
// INITIALIZATION
// ============================================================================

func initLogging() {
	var err error
	logFile, err = os.OpenFile("sniper.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Fatal("cannot open log file:", err)
	}
	log.SetOutput(io.MultiWriter(os.Stdout, logFile))
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
}

func loadConfig() {
	config = Config{
		MaxBuySOL:       envFloat("MAX_BUY_SOL", 0.05),
		ProfitTargetPct: envFloat("PROFIT_TARGET_PCT", 100.0),
		StopLossPct:     envFloat("STOP_LOSS_PCT", 25.0),
		TrailDropPct:    envFloat("TRAIL_DROP_PCT", 15.0),
		MinRugScore:     envFloat("MIN_RUG_SCORE", 30.0),
		MaxTopHolderPct: envFloat("MAX_TOP_HOLDER_PCT", 25.0),
		MinLiquidity:    envFloat("MIN_LIQUIDITY_SOL", 5.0),
		BuyCooldownSec:  envInt("BUY_COOLDOWN_SEC", 10),
		PositionTimeout: envInt("POSITION_TIMEOUT_MIN", 5),
		SimulateFirst:   envBool("SIMULATE_BEFORE_SEND", true),
		JitoTipLamports: uint64(envInt("JITO_TIP_LAMPORTS", 2000)),
		JitoEndpoint:    envString("JITO_ENDPOINT", "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/bundles"),
		MaxFailedBuys:   envInt("MAX_FAILED_BUYS", 3),
		ProgramWhitelist: map[string]bool{
			"JUP6LkbZbjwQRus81QE3E6Bg8JSqwhbzd69nWj2kxg8": true,
			"9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFu": true,
			"6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P": true,
		},
	}

	required := []string{"HELIUS_API_KEY", "JUPITER_API_KEY"}
	for _, k := range required {
		if os.Getenv(k) == "" {
			log.Fatalf("FATAL: missing required env var: %s", k)
		}
	}

	jupiterAPIKey = os.Getenv("JUPITER_API_KEY")
}

func loadWallets() {
	if keys := os.Getenv("PRIVATE_KEYS"); keys != "" {
		for _, k := range strings.Split(keys, ",") {
			k = strings.TrimSpace(k)
			if k == "" {
				continue
			}
			pk, err := solana.PrivateKeyFromBase58(k)
			if err != nil {
				log.Fatalf("FATAL: invalid key in PRIVATE_KEYS: %v", err)
			}
			wallets = append(wallets, pk)
		}
	} else if pk := os.Getenv("PRIVATE_KEY"); pk != "" {
		w, err := solana.PrivateKeyFromBase58(pk)
		if err != nil {
			log.Fatal("FATAL: invalid PRIVATE_KEY:", err)
		}
		wallets = append(wallets, w)
	}

	if len(wallets) == 0 {
		log.Fatal("FATAL: no wallets configured (set PRIVATE_KEY or PRIVATE_KEYS)")
	}
}

func initRPC() {
	heliusKey := os.Getenv("HELIUS_API_KEY")
	rpcURL := fmt.Sprintf("https://mainnet.helius-rpc.com/?api-key=%s", heliusKey)
	wsURL := fmt.Sprintf("wss://mainnet.helius-rpc.com/?api-key=%s", heliusKey)

	rpcClient = rpc.New(rpcURL)

	var err error
	wsClient, err = ws.Connect(context.Background(), wsURL)
	if err != nil {
		log.Fatal("FATAL: ws connect failed:", err)
	}
}

func initDatabase() {
	var err error
	db, err = sql.Open("sqlite3", "sniper.db?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		log.Fatal("FATAL: db open failed:", err)
	}

	schema := `
	CREATE TABLE IF NOT EXISTS tokens (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		mint TEXT UNIQUE,
		symbol TEXT DEFAULT '',
		score REAL DEFAULT 0,
		bought INTEGER DEFAULT 0,
		sold INTEGER DEFAULT 0,
		txbuy TEXT DEFAULT '',
		txsell TEXT DEFAULT '',
		buyprice REAL DEFAULT 0,
		sellprice REAL DEFAULT 0,
		pnl REAL DEFAULT 0,
		buytime DATETIME DEFAULT CURRENT_TIMESTAMP,
		selltime DATETIME,
		reason TEXT DEFAULT '',
		wallet TEXT DEFAULT ''
	);

	CREATE TABLE IF NOT EXISTS blacklist (
		mint TEXT PRIMARY KEY,
		reason TEXT DEFAULT '',
		added_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS stats_history (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
		total_buys INTEGER,
		total_sells INTEGER,
		profit_trades INTEGER,
		loss_trades INTEGER,
		total_pnl REAL
	);

	CREATE INDEX IF NOT EXISTS idx_tokens_mint ON tokens(mint);
	CREATE INDEX IF NOT EXISTS idx_tokens_buytime ON tokens(buytime);
	`

	_, err = db.Exec(schema)
	if err != nil {
		log.Printf("WARN: schema exec: %v", err)
	}

	// Self-heal: add missing columns to existing tables
	migrations := []string{
		"ALTER TABLE tokens ADD COLUMN buytime DATETIME DEFAULT CURRENT_TIMESTAMP",
		"ALTER TABLE tokens ADD COLUMN wallet TEXT DEFAULT ''",
		"ALTER TABLE tokens ADD COLUMN symbol TEXT DEFAULT ''",
		"ALTER TABLE tokens ADD COLUMN selltime DATETIME",
		"ALTER TABLE tokens ADD COLUMN reason TEXT DEFAULT ''",
	}
	for _, m := range migrations {
		_, _ = db.Exec(m) // ignore "duplicate column" errors
	}

	// Verify blacklist schema, recreate if corrupt
	if _, testErr := db.Exec("SELECT mint, reason FROM blacklist LIMIT 0"); testErr != nil {
		log.Printf("DB: blacklist schema mismatch, recreating: %v", testErr)
		db.Exec("DROP TABLE IF EXISTS blacklist")
		db.Exec(`CREATE TABLE blacklist (
			mint TEXT PRIMARY KEY,
			reason TEXT DEFAULT '',
			added_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`)
		log.Println("DB: blacklist table recreated with correct schema")
	}
}

func loadBlacklist() {
	rows, err := db.Query("SELECT mint FROM blacklist")
	if err != nil {
		log.Printf("WARN: load blacklist: %v", err)
		return
	}
	defer rows.Close()

	blacklistMu.Lock()
	defer blacklistMu.Unlock()

	for rows.Next() {
		var mint string
		if rows.Scan(&mint) == nil {
			blacklist[mint] = true
		}
	}
	log.Printf("Loaded %d blacklisted tokens", len(blacklist))
}

func logConfig() {
	log.Println("========================================")
	log.Println("       MOLD SNIPER BOT v2.0            ")
	log.Println("========================================")
	log.Printf("Max Buy:        %.4f SOL", config.MaxBuySOL)
	log.Printf("Profit Target:  %.1f%%", config.ProfitTargetPct)
	log.Printf("Stop Loss:      %.1f%%", config.StopLossPct)
	log.Printf("Trailing Stop:  %.1f%%", config.TrailDropPct)
	log.Printf("Min Rug Score:  %.0f", config.MinRugScore)
	log.Printf("Max Top Holder: %.1f%%", config.MaxTopHolderPct)
	log.Printf("Min Liquidity:  %.1f SOL", config.MinLiquidity)
	log.Printf("Buy Cooldown:   %ds", config.BuyCooldownSec)
	log.Printf("Position Timeout: %d min", config.PositionTimeout)
	log.Printf("Simulation:     %v", config.SimulateFirst)
	log.Printf("Jito Tip:       %d lamports", config.JitoTipLamports)
	log.Printf("Jito Endpoint:  %s", config.JitoEndpoint)
	log.Printf("Wallets:        %d (rotating every 5 trades)", len(wallets))
	log.Printf("Active Wallet:  %s", currentWallet().PublicKey())
	log.Println("========================================")
}

func cleanup() {
	if db != nil {
		saveStatsSnapshot()
		db.Close()
	}
	if wsClient != nil {
		wsClient.Close()
	}
	if logFile != nil {
		logFile.Close()
	}
}

// ============================================================================
// WALLET MANAGEMENT
// ============================================================================

func currentWallet() solana.PrivateKey {
	mu.Lock()
	defer mu.Unlock()
	return wallets[walletIdx%len(wallets)]
}

func rotateWallet() {
	mu.Lock()
	defer mu.Unlock()
	tradeCount++
	if tradeCount%5 == 0 && len(wallets) > 1 {
		walletIdx = (walletIdx + 1) % len(wallets)
		log.Printf("WALLET: Rotated to wallet %d: %s", walletIdx, wallets[walletIdx].PublicKey())
	}
}

func getWalletBalance(ctx context.Context, wallet solana.PrivateKey) (float64, error) {
	bal, err := rpcClient.GetBalance(ctx, wallet.PublicKey(), rpc.CommitmentConfirmed)
	if err != nil || bal == nil {
		return 0, err
	}
	return float64(bal.Value) / 1e9, nil
}

// ============================================================================
// STATS & MONITORING
// ============================================================================

func statsReporter(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			printStats()
			saveStatsSnapshot()
		}
	}
}

func printStats() {
	statsMu.RLock()
	defer statsMu.RUnlock()

	uptime := time.Since(stats.StartTime).Round(time.Second)
	log.Println("========== STATS ==========")
	log.Printf("Uptime:          %s", uptime)
	log.Printf("Total Buys:      %d (success: %d, failed: %d)", stats.TotalBuys, stats.SuccessfulBuys, stats.FailedBuys)
	log.Printf("Total Sells:     %d", stats.TotalSells)
	log.Printf("Profit Trades:   %d", stats.ProfitTrades)
	log.Printf("Loss Trades:     %d", stats.LossTrades)
	log.Printf("Total PnL:       %.6f SOL", stats.TotalPnL)
	log.Printf("WS Reconnects:   %d", atomic.LoadInt64(&wsReconnects))
	log.Println("===========================")
}

func printFinalStats() {
	log.Println("\n========== FINAL STATS ==========")
	printStats()
	log.Println("==================================")
}

func saveStatsSnapshot() {
	statsMu.RLock()
	defer statsMu.RUnlock()

	_, _ = db.Exec(`INSERT INTO stats_history (total_buys, total_sells, profit_trades, loss_trades, total_pnl)
		VALUES (?, ?, ?, ?, ?)`,
		stats.TotalBuys, stats.TotalSells, stats.ProfitTrades, stats.LossTrades, stats.TotalPnL)
}

func incrementStat(field string) {
	statsMu.Lock()
	defer statsMu.Unlock()

	switch field {
	case "total_buys":
		stats.TotalBuys++
	case "successful_buys":
		stats.SuccessfulBuys++
	case "failed_buys":
		stats.FailedBuys++
	case "total_sells":
		stats.TotalSells++
	case "profit_trades":
		stats.ProfitTrades++
	case "loss_trades":
		stats.LossTrades++
	}
}

func addPnL(pnl float64) {
	statsMu.Lock()
	defer statsMu.Unlock()
	stats.TotalPnL += pnl
}

func healthCheck(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	lastAlert := time.Time{}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			wallet := currentWallet()
			bal, err := getWalletBalance(ctx, wallet)
			if err != nil {
				log.Printf("HEALTH: wallet balance check failed: %v", err)
			} else if bal < config.MaxBuySOL+0.005 {
				log.Printf("HEALTH: LOW BALANCE WARNING: %.4f SOL", bal)
				
				// 2026 PRO: Critical balance alert (< 0.005 SOL) with Telegram
				if bal < 0.005 && time.Since(lastAlert) > 5*time.Minute {
					walletAddr := wallet.PublicKey().String()
					shortAddr := walletAddr[:4] + "..." + walletAddr[len(walletAddr)-3:]
					sendTelegram(fmt.Sprintf("⚠️ Wallet empty — send 0.1 SOL to %s or snipes stop\nCurrent balance: %.6f SOL", shortAddr, bal))
					lastAlert = time.Now()
				}
			}
		}
	}
}

// ============================================================================
// PUMP.FUN MONITORING
// ============================================================================

func monitorPump(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := subscribeAndListen(ctx); err != nil {
			if ctx.Err() != nil {
				return
			}
			log.Printf("PUMP: subscription error: %v, reconnecting...", err)
			atomic.AddInt64(&wsReconnects, 1)

			// Recreate WebSocket connection on failure
			if wsClient != nil {
				wsClient.Close()
			}
			heliusKey := os.Getenv("HELIUS_API_KEY")
			wsURL := fmt.Sprintf("wss://mainnet.helius-rpc.com/?api-key=%s", heliusKey)
			newWs, connErr := ws.Connect(ctx, wsURL)
			if connErr != nil {
				log.Printf("PUMP: ws reconnect failed: %v, retrying in 5s...", connErr)
				time.Sleep(5 * time.Second)
				continue
			}
			wsClient = newWs
			log.Println("PUMP: ws reconnected successfully")
			time.Sleep(1 * time.Second)
		}
	}
}

func subscribeAndListen(ctx context.Context) error {
	sub, err := wsClient.LogsSubscribeMentions(pumpID, rpc.CommitmentConfirmed)
	if err != nil {
		return fmt.Errorf("subscribe: %w", err)
	}
	defer sub.Unsubscribe()

	log.Println("PUMP: subscribed to Pump.fun program logs")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		msg, err := sub.Recv()
		if err != nil {
			return fmt.Errorf("recv: %w", err)
		}

		if msg.Value.Err != nil {
			continue
		}

		logs := strings.Join(msg.Value.Logs, " ")
		if strings.Contains(logs, "Instruction: Create") {
			go handleToken(ctx, msg.Value.Signature)
		}
	}
}

// ============================================================================
// TOKEN HANDLING
// ============================================================================

func handleToken(ctx context.Context, sig solana.Signature) {
	time.Sleep(500 * time.Millisecond)

	tx, err := rpcClient.GetTransaction(ctx, sig, &rpc.GetTransactionOpts{
		MaxSupportedTransactionVersion: ptr(uint64(0)),
		Commitment:                     rpc.CommitmentConfirmed,
	})
	if err != nil || tx == nil {
		return
	}

	mint := parseMint(tx)
	if mint == "" {
		return
	}

	if isBlacklisted(mint) {
		log.Printf("SKIP: %s is blacklisted", mint)
		return
	}

	if tokenExists(mint) {
		return
	}

	if !canBuy() {
		return
	}

	// ENHANCED SAFETY CHECK: Comprehensive honeypot/rug-pull detection
	safetyResult := isSafeToken(ctx, mint)
	if !safetyResult.Safe {
		blockMsg := fmt.Sprintf("🔴 <b>SAFETY BLOCKED</b> — Guard Engaged\nToken: %s\nReason: %s\nDetails: %s", 
			mint[:16]+"...", safetyResult.Reason, safetyResult.Details)
		sendTelegram(blockMsg)
		log.Printf("SAFETY: %s REJECTED - %s: %s", mint, safetyResult.Reason, safetyResult.Details)
		addToBlacklist(mint, safetyResult.Reason)
		return
	}

	log.Printf("SAFETY: %s PASSED all checks ✓", mint)

	// Legacy check for backward compatibility (some fields still used)
	rug, err := checkRugCheck(ctx, mint)
	if err != nil {
		log.Printf("RUGCHECK: %s failed: %v", mint, err)
		rug = &RugCheckResponse{Score: 50}
	}

	wallet := currentWallet()
	bal, err := getWalletBalance(ctx, wallet)
	if err != nil {
		log.Printf("BALANCE: check failed: %v", err)
		return
	}

	// Use volume-tier buy size for balance check
	buySize := safetyResult.BuySizeSOL
	if buySize <= 0 {
		buySize = config.MaxBuySOL
	}
	need := buySize + 0.005
	if bal < need {
		log.Printf("BALANCE: insufficient %.4f < %.4f SOL", bal, need)
		return
	}

	symbol := rug.TokenMeta.Symbol
	if symbol == "" {
		symbol = "???"
	}

	log.Printf("BUY: %s (%s) score=%.0f slip=%dbps size=%.3fSOL", mint, symbol, rug.Score, safetyResult.SlippageBps, buySize)
	buy(ctx, mint, symbol, rug.Score, safetyResult.SlippageBps, buySize)
}

func parseMint(tx *rpc.GetTransactionResult) string {
	if tx.Meta == nil {
		return ""
	}

	re := regexp.MustCompile(`Create\(([1-9A-HJ-NP-Za-km-z]{32,44})\)`)
	for _, l := range tx.Meta.LogMessages {
		if m := re.FindStringSubmatch(l); len(m) > 1 {
			return m[1]
		}
	}

	for _, b := range tx.Meta.PostTokenBalances {
		if s := b.Mint.String(); s != "" && s != solMint {
			return s
		}
	}

	return ""
}

func isBlacklisted(mint string) bool {
	blacklistMu.RLock()
	defer blacklistMu.RUnlock()
	return blacklist[mint]
}

func addToBlacklist(mint, reason string) {
	blacklistMu.Lock()
	blacklist[mint] = true
	blacklistMu.Unlock()

	_, _ = db.Exec("INSERT OR IGNORE INTO blacklist (mint, reason) VALUES (?, ?)", mint, reason)
	recordRug(mint)
}

func tokenExists(mint string) bool {
	var exists string
	return db.QueryRow("SELECT mint FROM tokens WHERE mint=?", mint).Scan(&exists) == nil
}

func canBuy() bool {
	mu.Lock()
	defer mu.Unlock()

	cooldown := time.Duration(config.BuyCooldownSec) * time.Second
	if time.Since(lastBuyTime) < cooldown {
		return false
	}
	return true
}

func passesRugCheck(mint string, rug *RugCheckResponse) bool {
	if rug.Score < config.MinRugScore {
		log.Printf("SKIP: %s score=%.0f < %.0f", mint, rug.Score, config.MinRugScore)
		addToBlacklist(mint, "low_score")
		return false
	}

	for _, r := range rug.Risks {
		if r.Level == "critical" || r.Level == "high" {
			log.Printf("SKIP: %s risk=%s (%s)", mint, r.Name, r.Level)
			addToBlacklist(mint, "risk_"+r.Name)
			return false
		}
	}

	if rug.Freezable {
		log.Printf("SKIP: %s has freeze authority", mint)
		addToBlacklist(mint, "freezable")
		return false
	}

	if rug.Mintable {
		log.Printf("SKIP: %s has mint authority", mint)
		addToBlacklist(mint, "mintable")
		return false
	}

	if len(rug.TopHolders) > 0 && rug.TopHolders[0].Pct > config.MaxTopHolderPct {
		log.Printf("SKIP: %s top holder %.1f%% > %.1f%%", mint, rug.TopHolders[0].Pct, config.MaxTopHolderPct)
		addToBlacklist(mint, "concentrated")
		return false
	}

	var totalLiq float64
	for _, m := range rug.Markets {
		totalLiq += m.Liquidity()
	}
	if totalLiq > 0 && totalLiq < config.MinLiquidity*1e9 {
		log.Printf("SKIP: %s liquidity %.2f < %.2f SOL", mint, totalLiq/1e9, config.MinLiquidity)
		return false
	}

	return true
}

// ============================================================================
// ENHANCED SAFETY CHECKS (2026 Production-Grade)
// ============================================================================

// isSafeToken performs comprehensive honeypot/rug-pull detection
func isSafeToken(ctx context.Context, mint string) SafetyCheckResult {
	// Step 1: RugCheck API scan with retry logic
	rugResult := checkRugCheckEnhanced(ctx, mint)
	if !rugResult.Safe {
		sendTelegram(fmt.Sprintf("⚠️ Rug detected on %s\n%s\nDetails: %s", 
			mint, rugResult.Reason, rugResult.Details))
		return rugResult
	}

	// Step 2: RPC-based mint/freeze authority check
	rpcResult := checkMintAuthorities(ctx, mint)
	if !rpcResult.Safe {
		sendTelegram(fmt.Sprintf("⚠️ Authority risk on %s\n%s\nDetails: %s",
			mint, rpcResult.Reason, rpcResult.Details))
		return rpcResult
	}

	// Step 3: Volume momentum filter via Dexscreener
	volResult := checkVolumeMomentum(ctx, mint)
	if !volResult.Safe {
		return volResult
	}

	// All checks passed — carry volume tier through
	return SafetyCheckResult{
		Safe:        true,
		Reason:      "all_checks_passed",
		Details:     fmt.Sprintf("Token passed all safety checks | %s", volResult.Details),
		SlippageBps: volResult.SlippageBps,
		BuySizeSOL:  volResult.BuySizeSOL,
	}
}

// checkVolumeMomentum queries Dexscreener for 5-min volume to filter flat tokens
// and dynamically size position + slippage based on momentum.
func checkVolumeMomentum(ctx context.Context, mint string) SafetyCheckResult {
	dexURL := fmt.Sprintf("https://api.dexscreener.com/latest/dex/tokens/%s", mint)

	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt) * time.Second)
		}

		req, err := http.NewRequestWithContext(ctx, "GET", dexURL, nil)
		if err != nil {
			lastErr = err
			continue
		}

		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == 429 {
			lastErr = fmt.Errorf("rate limited (429)")
			log.Printf("VOLUME: %s Dexscreener 429 — backing off %ds", mint, (attempt+1)*5)
			time.Sleep(time.Duration(attempt+1) * 5 * time.Second)
			continue
		}

		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("status %d", resp.StatusCode)
			continue
		}

		var data struct {
			Pairs []struct {
				Volume struct {
					M5 float64 `json:"m5"`
					H1 float64 `json:"h1"`
				} `json:"volume"`
			} `json:"pairs"`
		}

		if err := json.Unmarshal(body, &data); err != nil {
			lastErr = err
			continue
		}

		// No pairs yet — token just launched, allow with defaults
		if len(data.Pairs) == 0 {
			log.Printf("VOLUME: %s no pairs on Dexscreener yet, using defaults", mint)
			return SafetyCheckResult{
				Safe: true, Reason: "volume_no_data",
				Details:     "No Dexscreener data, using defaults",
				SlippageBps: 500, BuySizeSOL: config.MaxBuySOL,
			}
		}

		vol5m := data.Pairs[0].Volume.M5

		// REJECT: flat token
		if vol5m < 20000 {
			log.Printf("LOW VOLUME: %s $%.0f (5min) — skipping flat token", mint, vol5m)
			return SafetyCheckResult{
				Safe:   false,
				Reason: "low_volume",
				Details: fmt.Sprintf("5min vol $%.0f < $20,000", vol5m),
			}
		}

		// TIER 2: high momentum
		if vol5m > 100000 {
			log.Printf("VOLUME: %s $%.0f (5min) — HIGH MOMENTUM, sizing up", mint, vol5m)
			return SafetyCheckResult{
				Safe: true, Reason: "high_volume",
				Details:     fmt.Sprintf("5min vol $%.0f", vol5m),
				SlippageBps: 1000, BuySizeSOL: 0.03,
			}
		}

		// TIER 1: normal momentum
		log.Printf("VOLUME: %s $%.0f (5min) — momentum confirmed", mint, vol5m)
		return SafetyCheckResult{
			Safe: true, Reason: "volume_ok",
			Details:     fmt.Sprintf("5min vol $%.0f", vol5m),
			SlippageBps: 800, BuySizeSOL: 0.02,
		}
	}

	// All 3 retries failed — fallback: allow buy with defaults (RPC-only mode)
	log.Printf("VOLUME: %s Dexscreener failed after 3 retries: %v — fallback to defaults", mint, lastErr)
	return SafetyCheckResult{
		Safe: true, Reason: "volume_api_fail",
		Details:     "Dexscreener unavailable, using defaults",
		SlippageBps: 500, BuySizeSOL: config.MaxBuySOL,
	}
}

// checkRugCheckEnhanced performs enhanced RugCheck API validation
func checkRugCheckEnhanced(ctx context.Context, mint string) SafetyCheckResult {
	apiKey := os.Getenv("RUGCHECK_API_KEY")
	if apiKey == "" {
		log.Println("WARN: RUGCHECK_API_KEY not set, using basic checks")
	}

	var rug *RugCheckResponse
	var err error

	// Retry logic with backoff for 429/5xx
	for attempt := 0; attempt < 3; attempt++ {
		rug, err = checkRugCheckAPI(ctx, mint, apiKey, attempt)
		if err == nil {
			break
		}

		if attempt < 2 {
			backoff := time.Duration(attempt+1) * 2 * time.Second
			log.Printf("RUGCHECK: attempt %d failed, retrying in %v: %v", attempt+1, backoff, err)
			time.Sleep(backoff)
		}
	}

	if err != nil {
		log.Printf("RUGCHECK: all attempts failed for %s: %v", mint, err)
		// Fallback to basic RPC checks if API is down
		return SafetyCheckResult{Safe: true, Reason: "api_unavailable", Details: "RugCheck API unavailable, proceeding with RPC checks"}
	}

	// Check 1: Risk Level
	if rug.RiskLevel == "HIGH" || rug.RiskLevel == "CRITICAL" {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "high_risk_level",
			Details: fmt.Sprintf("RiskLevel: %s", rug.RiskLevel),
		}
	}

	// Check 2: Trust Score (must be >= 80)
	if rug.TrustScore.Value > 0 && rug.TrustScore.Value < 80 {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "low_trust_score",
			Details: fmt.Sprintf("TrustScore: %.1f < 80", rug.TrustScore.Value),
		}
	}

	// Check 3: Scams array - check for honeypot/rug_pull
	for _, scam := range rug.Scams {
		scamType := strings.ToUpper(scam.Type)
		if scamType == "HONEYPOT" || scamType == "RUG_PULL" || scamType == "SCAM" {
			return SafetyCheckResult{
				Safe:    false,
				Reason:  "scam_detected",
				Details: fmt.Sprintf("Scam type: %s (severity: %s)", scam.Type, scam.Severity),
			}
		}
		if scam.Severity == "high" || scam.Severity == "critical" {
			return SafetyCheckResult{
				Safe:    false,
				Reason:  "high_severity_scam",
				Details: fmt.Sprintf("Scam: %s (severity: %s)", scam.Type, scam.Severity),
			}
		}
	}

	// Check 4: Critical/High risks
	for _, r := range rug.Risks {
		if r.Level == "critical" || r.Level == "high" {
			return SafetyCheckResult{
				Safe:    false,
				Reason:  "critical_risk",
				Details: fmt.Sprintf("Risk: %s (%s)", r.Name, r.Level),
			}
		}
	}

	// Check 5: Freeze/Mint authorities
	if rug.Freezable {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "freeze_authority_enabled",
			Details: "Token has freeze authority - dev can freeze accounts",
		}
	}

	if rug.Mintable {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "mint_authority_enabled",
			Details: "Token has mint authority - dev can mint unlimited supply",
		}
	}

	// Check 6: Liquidity requirements
	if rug.LiquidityDetails.LiquidityLocked > 0 && rug.LiquidityDetails.LiquidityLocked < 80 {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "low_liquidity_locked",
			Details: fmt.Sprintf("Only %.1f%% liquidity locked", rug.LiquidityDetails.LiquidityLocked),
		}
	}

	// 2026 PRO: $5000 USD liquidity floor
	if rug.LiquidityDetails.TotalLiquidity > 0 && rug.LiquidityDetails.TotalLiquidity < 5000 {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "low_total_liquidity",
			Details: fmt.Sprintf("Total liquidity: $%.0f < $5,000", rug.LiquidityDetails.TotalLiquidity),
		}
	}

	// Check 7: Top holder concentration
	if len(rug.TopHolders) > 0 && rug.TopHolders[0].Pct > config.MaxTopHolderPct {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "concentrated_holdings",
			Details: fmt.Sprintf("Top holder: %.1f%% > %.1f%%", rug.TopHolders[0].Pct, config.MaxTopHolderPct),
		}
	}

	// Check 8: Minimum rug score
	if rug.Score < config.MinRugScore {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "low_rug_score",
			Details: fmt.Sprintf("Score: %.0f < %.0f", rug.Score, config.MinRugScore),
		}
	}

	return SafetyCheckResult{Safe: true, Reason: "rugcheck_passed", Details: "Passed all RugCheck validations"}
}

// checkRugCheckAPI calls the RugCheck API with optional API key
func checkRugCheckAPI(ctx context.Context, mint, apiKey string, attempt int) (*RugCheckResponse, error) {
	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(reqCtx, "GET", "https://api.rugcheck.xyz/v1/tokens/"+mint+"/report", nil)
	req.Header.Set("User-Agent", "mold-v2")
	req.Header.Set("Accept", "application/json")

	if apiKey != "" {
		req.Header.Set("X-API-KEY", apiKey)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	// Handle rate limiting and server errors with retry
	if resp.StatusCode == 429 || resp.StatusCode >= 500 {
		return nil, fmt.Errorf("retryable error %d: %s", resp.StatusCode, body)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}

	var r RugCheckResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	return &r, nil
}

// checkMintAuthorities validates mint/freeze authorities via RPC
func checkMintAuthorities(ctx context.Context, mint string) SafetyCheckResult {
	mintPubkey, err := solana.PublicKeyFromBase58(mint)
	if err != nil {
		return SafetyCheckResult{Safe: false, Reason: "invalid_mint_address", Details: err.Error()}
	}

	reqCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	accountInfo, err := rpcClient.GetAccountInfo(reqCtx, mintPubkey)
	if err != nil {
		log.Printf("RPC: mint authority check failed for %s: %v", mint, err)
		// Don't fail on RPC errors, just log
		return SafetyCheckResult{Safe: true, Reason: "rpc_check_skipped", Details: "RPC check failed, proceeding"}
	}

	if accountInfo == nil || accountInfo.Value == nil {
		return SafetyCheckResult{Safe: false, Reason: "mint_account_not_found", Details: "Mint account doesn't exist"}
	}

	// Parse mint account data (SPL Token Mint layout)
	data := accountInfo.Value.Data.GetBinary()
	if len(data) < 82 {
		return SafetyCheckResult{Safe: false, Reason: "invalid_mint_data", Details: "Mint account data too short"}
	}

	// Mint authority: byte 0-4 (option flag + pubkey if set)
	// Byte 0: 0 = no authority, 1 = has authority
	hasMintAuthority := data[0] == 1

	// Freeze authority: byte 46-50 (option flag + pubkey if set)
	hasFreezeAuthority := len(data) > 46 && data[46] == 1

	if hasMintAuthority {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "mint_authority_detected",
			Details: "RPC confirmed: token has mint authority (can create unlimited supply)",
		}
	}

	if hasFreezeAuthority {
		return SafetyCheckResult{
			Safe:    false,
			Reason:  "freeze_authority_detected",
			Details: "RPC confirmed: token has freeze authority (can freeze user accounts)",
		}
	}

	return SafetyCheckResult{Safe: true, Reason: "authorities_revoked", Details: "Mint and freeze authorities properly revoked"}
}

// Legacy function for backward compatibility
func checkRugCheck(ctx context.Context, mint string) (*RugCheckResponse, error) {
	apiKey := os.Getenv("RUGCHECK_API_KEY")
	return checkRugCheckAPI(ctx, mint, apiKey, 0)
}

// ============================================================================
// BUY FLOW
// ============================================================================

func buy(ctx context.Context, mint, symbol string, score float64, volSlippage int, volBuySOL float64) {
	// FAIL-SAFE: Check if we've exceeded max failed buys
	statsMu.RLock()
	failedBuys := stats.FailedBuys
	statsMu.RUnlock()
	
	if failedBuys >= int64(config.MaxFailedBuys) {
		log.Printf("🔥 FORENSIC LOCK: Failed buys (%d) >= MAX_FAILED_BUYS (%d). HALTING.", failedBuys, config.MaxFailedBuys)
		sendTelegram(fmt.Sprintf("🔥 FORENSIC LOCK ENGAGED\nFailed buys: %d / %d\nBot halted for safety.", failedBuys, config.MaxFailedBuys))
		return
	}

	mu.Lock()
	lastBuyTime = time.Now()
	mu.Unlock()

	setActivePosition(mint, true)
	defer setActivePosition(mint, false)

	incrementStat("total_buys")

	activePosMu.RLock()
	if len(activePos) >= 3 {
		activePosMu.RUnlock()
		log.Printf("SKIP: concurrent buy limit reached (max 3)")
		incrementStat("failed_buys")
		return
	}
	activePosMu.RUnlock()

	// Volume-tier buy size (fallback to config)
	buySOL := volBuySOL
	if buySOL <= 0 {
		buySOL = config.MaxBuySOL
	}

	// Volume-tier base slippage (fallback to 500 bps)
	baseSlip := volSlippage
	if baseSlip <= 0 {
		baseSlip = 500
	}

	wallet := currentWallet()
	amt := uint64(buySOL * 1e9)

	quote, err := jupiterQuote(ctx, solMint, mint, amt, baseSlip)
	if err != nil {
		log.Printf("BUY: quote failed: %v", err)
		incrementStat("failed_buys")
		return
	}

	outAmtStr, _ := quote["outAmount"].(string)
	outAmt, _ := strconv.ParseFloat(outAmtStr, 64)

	priceImpact := getPriceImpact(quote)
	if priceImpact > 10.0 {
		log.Printf("BUY: price impact too high: %.2f%%", priceImpact)
		incrementStat("failed_buys")
		return
	}

	slip := calculateSlippage(outAmt, priceImpact)
	// Volume-tier minimum — never go below what momentum demands
	if volSlippage > slip {
		slip = volSlippage
	}

	if slip != baseSlip {
		quote, err = jupiterQuote(ctx, solMint, mint, amt, slip)
		if err != nil {
			log.Printf("BUY: re-quote failed: %v", err)
			incrementStat("failed_buys")
			return
		}
		outAmtStr, _ = quote["outAmount"].(string)
		outAmt, _ = strconv.ParseFloat(outAmtStr, 64)
	}

	swap, err := jupiterSwap(ctx, quote, wallet)
	if err != nil {
		log.Printf("BUY: swap failed: %v", err)
		incrementStat("failed_buys")
		return
	}

	txB64, _ := swap["swapTransaction"].(string)
	if txB64 == "" {
		log.Println("BUY: no swapTransaction in response")
		incrementStat("failed_buys")
		return
	}

	txBytes, _ := base64.StdEncoding.DecodeString(txB64)
	transaction, err := solana.TransactionFromDecoder(bin.NewBinDecoder(txBytes))
	if err != nil {
		log.Printf("BUY: parse tx: %v", err)
		incrementStat("failed_buys")
		return
	}

	_ = system.NewTransferInstruction(config.JitoTipLamports, wallet.PublicKey(), jitoTipAcct).Build()

	getOrAddAccount := func(pk solana.PublicKey) uint16 {
		for i, acc := range transaction.Message.AccountKeys {
			if acc.Equals(pk) {
				return uint16(i)
			}
		}
		transaction.Message.AccountKeys = append(transaction.Message.AccountKeys, pk)
		return uint16(len(transaction.Message.AccountKeys) - 1)
	}

	fromIdx := getOrAddAccount(wallet.PublicKey())
	toIdx := getOrAddAccount(jitoTipAcct)
	sysProgIdx := getOrAddAccount(solana.SystemProgramID)

	var tipData [12]byte
	tipData[0] = 2
	tipData[1] = 0
	tipData[2] = 0
	tipData[3] = 0
	for i := 0; i < 8; i++ {
		tipData[4+i] = byte(config.JitoTipLamports >> (i * 8))
	}

	compiledTip := solana.CompiledInstruction{
		ProgramIDIndex: sysProgIdx,
		Accounts:       []uint16{fromIdx, toIdx},
		Data:           tipData[:],
	}

	transaction.Message.Instructions = append(transaction.Message.Instructions, compiledTip)

	_, err = transaction.Sign(func(p solana.PublicKey) *solana.PrivateKey {
		if p.Equals(wallet.PublicKey()) {
			return &wallet
		}
		return nil
	})
	if err != nil {
		log.Printf("BUY: sign: %v", err)
		incrementStat("failed_buys")
		return
	}

	if config.SimulateFirst {
		sim, err := rpcClient.SimulateTransaction(ctx, transaction)
		if err != nil {
			log.Printf("BUY: simulation err: %v", err)
			incrementStat("failed_buys")
			return
		}
		if sim.Value != nil && sim.Value.Err != nil {
			log.Printf("BUY: simulation failed: %v", sim.Value.Err)
			incrementStat("failed_buys")
			return
		}
	}

	sig, err := bundleTx(ctx, transaction)
	if err != nil {
		log.Printf("BUY: send failed: %v", err)
		incrementStat("failed_buys")
		return
	}

	incrementStat("successful_buys")
	log.Printf("BUY TX: https://solscan.io/tx/%s", sig)

	_, _ = db.Exec(`INSERT INTO tokens (mint, symbol, score, bought, txbuy, buyprice, buytime, wallet)
		VALUES (?, ?, ?, 1, ?, ?, ?, ?)`,
		mint, symbol, score, sig.String(), outAmt, time.Now(), wallet.PublicKey().String())

	sendTelegram(fmt.Sprintf("🟢 BUY %s (%s)\n%.4f SOL | Score: %.0f\nhttps://solscan.io/tx/%s",
		mint, symbol, config.MaxBuySOL, score, sig))

	// FAIL-SAFE: Add 2-second delay between buys to prevent rapid drain
	time.Sleep(2 * time.Second)

	rotateWallet()

	go monitorPosition(ctx, mint, symbol, outAmt, wallet)
}

func getPriceImpact(quote map[string]interface{}) float64 {
	if pi, ok := quote["priceImpactPct"].(string); ok {
		v, _ := strconv.ParseFloat(pi, 64)
		return math.Abs(v)
	}
	return 0
}

func calculateSlippage(outAmt, priceImpact float64) int {
	if priceImpact > 5.0 {
		return 500
	}
	if outAmt > 0 && config.MaxBuySOL > 0.05 {
		return 200 + rand.Intn(200)
	}
	return 500
}

func setActivePosition(mint string, active bool) {
	activePosMu.Lock()
	defer activePosMu.Unlock()
	if active {
		activePos[mint] = true
	} else {
		delete(activePos, mint)
	}
}

// ============================================================================
// JUPITER API
// ============================================================================

func jupiterQuote(ctx context.Context, inMint, outMint string, amount uint64, slipBps int) (map[string]interface{}, error) {
	u := fmt.Sprintf("https://api.jup.ag/swap/v1/quote?inputMint=%s&outputMint=%s&amount=%d&slippageBps=%d",
		inMint, outMint, amount, slipBps)

	var lastErr error
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
		req.Header.Set("x-api-key", jupiterAPIKey)
		req.Header.Set("User-Agent", "mold-v2")

		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("quote %d: %s", resp.StatusCode, body)
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil {
			lastErr = fmt.Errorf("decode: %w", err)
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		if e, ok := m["error"].(string); ok {
			return nil, fmt.Errorf(e)
		}

		return m, nil
	}
	return nil, lastErr
}

func jupiterSwap(ctx context.Context, quote map[string]interface{}, wallet solana.PrivateKey) (map[string]interface{}, error) {
	reqBody, _ := json.Marshal(map[string]interface{}{
		"quoteResponse":             quote,
		"userPublicKey":             wallet.PublicKey().String(),
		"wrapAndUnwrapSol":          true,
		"dynamicComputeUnitLimit":   true,
		"prioritizationFeeLamports": "auto",
	})

	var lastErr error
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.jup.ag/swap/v1/swap", bytes.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", jupiterAPIKey)
		req.Header.Set("User-Agent", "mold-v2")

		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("swap %d: %s", resp.StatusCode, body)
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil {
			lastErr = fmt.Errorf("decode: %w", err)
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		if e, ok := m["error"].(string); ok {
			return nil, fmt.Errorf(e)
		}

		return m, nil
	}
	return nil, lastErr
}

// ============================================================================
// POSITION MONITORING
// ============================================================================

func monitorPosition(ctx context.Context, mint, symbol string, buyAmt float64, wallet solana.PrivateKey) {
	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()

	timeout := time.After(time.Duration(config.PositionTimeout) * time.Minute)
	maxVal := buyAmt
	partialSold := false

	log.Printf("MONITOR: started for %s (%s)", mint, symbol)

	for {
		select {
		case <-ctx.Done():
			return
		case <-timeout:
			setActivePosition(mint, false)
			sell(ctx, mint, symbol, buyAmt, "timeout", wallet, 100)
			return
		case <-tick.C:
			bal, err := getTokenBalance(ctx, mint, wallet)
			if err != nil {
				log.Printf("MONITOR: %s balance error: %v", mint, err)
				continue
			}
			if bal == 0 {
				log.Printf("MONITOR: %s position closed externally", mint)
				return
			}

			val, err := estimateValue(ctx, mint, bal)
			if err != nil {
				continue
			}
			if buyAmt == 0 {
				continue
			}

			if val > maxVal {
				maxVal = val
			}

			pnl := ((val - buyAmt) / buyAmt) * 100

			// 2026 PRO: Partial sell at 50%+ gain
			if !partialSold && pnl >= 50.0 {
				setActivePosition(mint, false)
				log.Printf("Auto-sell %s at +%.0f%% — selling 70%%, holding 30%%", mint, pnl)
				sell(ctx, mint, symbol, buyAmt, fmt.Sprintf("partial_70pct_at_+%.0f%%", pnl), wallet, 70)
				partialSold = true
				setActivePosition(mint, true)
				continue
			}

			if pnl >= config.ProfitTargetPct {
				setActivePosition(mint, false)
				sell(ctx, mint, symbol, buyAmt, fmt.Sprintf("profit_%.0f%%", pnl), wallet, 100)
				return
			}

			if pnl <= -config.StopLossPct {
				setActivePosition(mint, false)
				sell(ctx, mint, symbol, buyAmt, fmt.Sprintf("stoploss_%.0f%%", pnl), wallet, 100)
				return
			}

			if pnl > 0 && maxVal > 0 {
				dropFromPeak := ((maxVal - val) / maxVal) * 100
				if dropFromPeak >= config.TrailDropPct {
					setActivePosition(mint, false)
					sell(ctx, mint, symbol, buyAmt, fmt.Sprintf("trail_%.0f%%", dropFromPeak), wallet, 100)
					return
				}
			}
		}
	}
}

func getTokenBalance(ctx context.Context, mint string, wallet solana.PrivateKey) (uint64, error) {
	pk := solana.MustPublicKeyFromBase58(mint)
	accs, err := rpcClient.GetTokenAccountsByOwner(ctx, wallet.PublicKey(),
		&rpc.GetTokenAccountsConfig{Mint: &pk},
		&rpc.GetTokenAccountsOpts{Commitment: rpc.CommitmentConfirmed})
	if err != nil {
		return 0, err
	}

	if len(accs.Value) == 0 {
		return 0, nil
	}

	data := accs.Value[0].Account.Data.GetBinary()
	if len(data) < 72 {
		return 0, nil
	}

	amount := uint64(data[64]) | uint64(data[65])<<8 | uint64(data[66])<<16 | uint64(data[67])<<24 |
		uint64(data[68])<<32 | uint64(data[69])<<40 | uint64(data[70])<<48 | uint64(data[71])<<56

	return amount, nil
}

func estimateValue(ctx context.Context, mint string, amt uint64) (float64, error) {
	q, err := jupiterQuote(ctx, mint, solMint, amt, 500)
	if err != nil {
		return 0, err
	}
	if s, ok := q["outAmount"].(string); ok {
		v, _ := strconv.ParseFloat(s, 64)
		return v, nil
	}
	return 0, nil
}

// ============================================================================
// SELL FLOW
// ============================================================================

func sell(ctx context.Context, mint, symbol string, buyAmt float64, reason string, wallet solana.PrivateKey, sellPct int) {
	incrementStat("total_sells")

	bal, _ := getTokenBalance(ctx, mint, wallet)
	if bal == 0 {
		log.Printf("SELL: %s no balance", mint)
		return
	}

	// Calculate amount to sell based on percentage
	sellBal := bal
	if sellPct < 100 {
		sellBal = uint64(float64(bal) * float64(sellPct) / 100.0)
		log.Printf("SELL: %s selling %d%% (%d tokens, keeping %d)", mint, sellPct, sellBal, bal-sellBal)
	}

	var sig solana.Signature
	var sellAmt float64

	for i := 0; i < 3; i++ {
		quote, err := jupiterQuote(ctx, mint, solMint, sellBal, 500)
		if err != nil {
			log.Printf("SELL: %s quote %d: %v", mint, i+1, err)
			time.Sleep(time.Duration(i+1) * 2 * time.Second)
			continue
		}

		if s, ok := quote["outAmount"].(string); ok {
			sellAmt, _ = strconv.ParseFloat(s, 64)
		}

		swap, err := jupiterSwap(ctx, quote, wallet)
		if err != nil {
			log.Printf("SELL: %s swap %d: %v", mint, i+1, err)
			time.Sleep(time.Duration(i+1) * 2 * time.Second)
			continue
		}

		txB64, ok := swap["swapTransaction"].(string)
		if !ok || txB64 == "" {
			log.Printf("SELL: %s no swapTransaction %d", mint, i+1)
			continue
		}

		txBytes, _ := base64.StdEncoding.DecodeString(txB64)
		tx, err := solana.TransactionFromDecoder(bin.NewBinDecoder(txBytes))
		if err != nil {
			log.Printf("SELL: %s parse %d: %v", mint, i+1, err)
			continue
		}

		_ = system.NewTransferInstruction(config.JitoTipLamports, wallet.PublicKey(), jitoTipAcct).Build()

		getOrAddAccount := func(pk solana.PublicKey) uint16 {
			for i, acc := range tx.Message.AccountKeys {
				if acc.Equals(pk) {
					return uint16(i)
				}
			}
			tx.Message.AccountKeys = append(tx.Message.AccountKeys, pk)
			return uint16(len(tx.Message.AccountKeys) - 1)
		}

		fromIdx := getOrAddAccount(wallet.PublicKey())
		toIdx := getOrAddAccount(jitoTipAcct)
		sysProgIdx := getOrAddAccount(solana.SystemProgramID)

		var tipData [12]byte
		tipData[0] = 2
		tipData[1] = 0
		tipData[2] = 0
		tipData[3] = 0
		for i := 0; i < 8; i++ {
			tipData[4+i] = byte(config.JitoTipLamports >> (i * 8))
		}

		compiledTip := solana.CompiledInstruction{
			ProgramIDIndex: sysProgIdx,
			Accounts:       []uint16{fromIdx, toIdx},
			Data:           tipData[:],
		}

		tx.Message.Instructions = append(tx.Message.Instructions, compiledTip)

		_, err = tx.Sign(func(p solana.PublicKey) *solana.PrivateKey {
			if p.Equals(wallet.PublicKey()) {
				return &wallet
			}
			return nil
		})
		if err != nil {
			log.Printf("SELL: %s sign %d: %v", mint, i+1, err)
			continue
		}

		if config.SimulateFirst {
			sim, err := rpcClient.SimulateTransaction(ctx, tx)
			if err != nil {
				log.Printf("SELL: %s sim err %d: %v", mint, i+1, err)
				time.Sleep(2 * time.Second)
				continue
			}
			if sim.Value != nil && sim.Value.Err != nil {
				log.Printf("SELL: %s sim fail %d: %v", mint, i+1, sim.Value.Err)
				time.Sleep(2 * time.Second)
				continue
			}
		}

		sig, err = bundleTx(ctx, tx)
		if err != nil {
			log.Printf("SELL: %s send %d: %v", mint, i+1, err)
			time.Sleep(time.Duration(i+1) * 2 * time.Second)
			continue
		}
		break
	}

	if sig.IsZero() {
		log.Printf("SELL FAILED: %s (%s) - no tx sent", mint, reason)
		return
	}

	pnl := (sellAmt - buyAmt) / 1e9
	addPnL(pnl)

	pnlPct := 0.0
	if buyAmt > 0 {
		pnlPct = ((sellAmt - buyAmt) / buyAmt) * 100
	}
	recordSnipe(mint, buyAmt, sellAmt, pnlPct)

	if pnl > 0 {
		incrementStat("profit_trades")
	} else {
		incrementStat("loss_trades")
	}

	log.Printf("SELL: %s (%s) reason=%s pnl=%.6f SOL https://solscan.io/tx/%s", mint, symbol, reason, pnl, sig)

	_, _ = db.Exec(`UPDATE tokens SET sold=1, txsell=?, sellprice=?, pnl=?, selltime=?, reason=? WHERE mint=?`,
		sig.String(), sellAmt, pnl, time.Now(), reason, mint)

	emoji := "🔴"
	if pnl > 0 {
		emoji = "🟢"
	}
	sendTelegram(fmt.Sprintf("%s SELL %s (%s)\n%s | PnL: %.6f SOL\nhttps://solscan.io/tx/%s",
		emoji, mint, symbol, reason, pnl, sig))
}

// ============================================================================
// TELEGRAM
// ============================================================================

func sendTelegram(msg string) {
	tok, chat := os.Getenv("TELEGRAM_TOKEN"), os.Getenv("CHAT_ID")
	if tok == "" || chat == "" {
		return
	}

	urlStr := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage?chat_id=%s&text=%s&parse_mode=HTML",
		tok, chat, url.QueryEscape(msg))

	// Enhanced retry with exponential backoff: 1s → 2s → 4s
	backoffs := []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second}
	for i := 0; i < 3; i++ {
		resp, err := http.Get(urlStr)
		if err != nil {
			log.Printf("TELEGRAM: send fail (attempt %d/3): %v", i+1, err)
			if i < 2 {
				time.Sleep(backoffs[i])
				continue
			}
			break
		}
		resp.Body.Close()
		if resp.StatusCode >= 400 {
			log.Printf("TELEGRAM: error %d (attempt %d/3)", resp.StatusCode, i+1)
			if i < 2 {
				time.Sleep(backoffs[i])
				continue
			}
			break
		}
		// Success
		log.Printf("TELEGRAM: SENT ✓")
		return
	}
}

// ============================================================================
// JITO BUNDLE
// ============================================================================

func bundleTx(ctx context.Context, tx *solana.Transaction) (solana.Signature, error) {
	txData, err := tx.MarshalBinary()
	if err != nil {
		return solana.Signature{}, err
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "sendBundle",
		"params":  [][]string{{base64.StdEncoding.EncodeToString(txData)}},
	})

	req, _ := http.NewRequestWithContext(ctx, "POST", config.JitoEndpoint, bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "mold-v2")

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("JITO: bundle failed: %v", err)
		return solana.Signature{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("JITO: error %d: %s", resp.StatusCode, body)
		return solana.Signature{}, fmt.Errorf("bundle rejected %d", resp.StatusCode)
	}

	sig := tx.Signatures[0]

	for i := 0; i < 5; i++ {
		time.Sleep(400 * time.Millisecond)

		status, err := rpcClient.GetSignatureStatuses(ctx, true, sig)
		if err != nil {
			log.Printf("JITO: poll %d error: %v", i+1, err)
			continue
		}

		if len(status.Value) == 0 || status.Value[0] == nil {
			log.Printf("JITO: poll %d not found", i+1)
			continue
		}

		if status.Value[0].Err != nil {
			log.Printf("JITO: tx failed on-chain: %v", status.Value[0].Err)
			return solana.Signature{}, fmt.Errorf("tx failed: %v", status.Value[0].Err)
		}

		cs := status.Value[0].ConfirmationStatus
		if cs == rpc.ConfirmationStatusConfirmed || cs == rpc.ConfirmationStatusFinalized {
			log.Printf("JITO: tx confirmed (%s): %s", cs, sig)
			return sig, nil
		}

		log.Printf("JITO: poll %d status: %s", i+1, cs)
	}

	log.Printf("JITO: tx not confirmed after 2s: %s", sig)
	return solana.Signature{}, fmt.Errorf("tx not confirmed after 2s")
}

// ============================================================================
// HELPERS
// ============================================================================

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	v := strings.ToLower(os.Getenv(key))
	if v == "true" || v == "1" {
		return true
	}
	if v == "false" || v == "0" {
		return false
	}
	return def
}

func envString(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func ptr[T any](v T) *T { return &v }

// ============================================================================
// MEMORY SYSTEM
// ============================================================================

const memoryFile = "sniper_memory.json"

func loadMemory() {
	memory = &BotMemory{
		Rugs: make(map[string]bool),
	}

	data, err := os.ReadFile(memoryFile)
	if err != nil {
		log.Println("MEMORY: no existing memory file, starting fresh")
		return
	}

	memory.mu.Lock()
	defer memory.mu.Unlock()

	if err := json.Unmarshal(data, memory); err != nil {
		log.Printf("MEMORY: parse error, starting fresh: %v", err)
		memory.Rugs = make(map[string]bool)
		return
	}

	if memory.Rugs == nil {
		memory.Rugs = make(map[string]bool)
	}

	log.Printf("MEMORY: loaded %d snipes, %d rugs, %d wins/%d losses",
		len(memory.Snipes), len(memory.Rugs), memory.Stats.TotalWins, memory.Stats.TotalLosses)
}

func saveMemory() {
	if memory == nil {
		return
	}

	memory.mu.RLock()
	defer memory.mu.RUnlock()

	data, err := json.MarshalIndent(memory, "", "  ")
	if err != nil {
		log.Printf("MEMORY: marshal error: %v", err)
		return
	}

	if err := os.WriteFile(memoryFile, data, 0644); err != nil {
		log.Printf("MEMORY: write error: %v", err)
	}
}

func recordSnipe(token string, buyPrice, sellPrice, profitPct float64) {
	if memory == nil {
		return
	}

	memory.mu.Lock()
	defer memory.mu.Unlock()

	memory.Snipes = append(memory.Snipes, SnipeRecord{
		Token:     token,
		BuyPrice:  buyPrice,
		SellPrice: sellPrice,
		ProfitPct: profitPct,
		Timestamp: time.Now(),
	})

	// Keep only last 20
	if len(memory.Snipes) > 20 {
		memory.Snipes = memory.Snipes[len(memory.Snipes)-20:]
	}

	if profitPct > 0 {
		memory.Stats.TotalWins++
	} else {
		memory.Stats.TotalLosses++
	}

	if profitPct > memory.Stats.BestDayPct {
		memory.Stats.BestDayPct = profitPct
	}

	go saveMemory()
}

func recordRug(mint string) {
	if memory == nil {
		return
	}

	memory.mu.Lock()
	memory.Rugs[mint] = true
	memory.mu.Unlock()

	go saveMemory()
}
