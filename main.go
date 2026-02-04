package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gagliardetto/solana-go"
	"github.com/gagliardetto/solana-go/rpc"
	"github.com/gagliardetto/solana-go/rpc/ws"
	"github.com/joho/godotenv"
	_ "github.com/mattn/go-sqlite3"
)

func parseFloat(v interface{}) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case string:
		f, _ := strconv.ParseFloat(val, 64)
		return f
	}
	return 0
}

var heliusTipAccounts = []string{
	"4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
	"D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
	"9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
}

func bundleTx(ctx context.Context, tx *solana.Transaction) (solana.Signature, error) {
	tipAccount := solana.MustPublicKeyFromBase58(heliusTipAccounts[rand.Intn(len(heliusTipAccounts))])

	// System program transfer: instruction index 2, amount 200000 lamports
	tipData := []byte{2, 0, 0, 0, 64, 13, 3, 0, 0, 0, 0, 0}

	msg := tx.Message
	newInstructions := append(msg.Instructions, solana.CompiledInstruction{})

	accountMap := make(map[solana.PublicKey]uint16)
	for i, acc := range msg.AccountKeys {
		accountMap[acc] = uint16(i)
	}

	newAccounts := msg.AccountKeys
	fromIdx, fromExists := accountMap[wallet.PublicKey()]
	if !fromExists {
		fromIdx = uint16(len(newAccounts))
		newAccounts = append(newAccounts, wallet.PublicKey())
		accountMap[wallet.PublicKey()] = fromIdx
	}

	toIdx, toExists := accountMap[tipAccount]
	if !toExists {
		toIdx = uint16(len(newAccounts))
		newAccounts = append(newAccounts, tipAccount)
		accountMap[tipAccount] = toIdx
	}

	sysIdx, sysExists := accountMap[solana.SystemProgramID]
	if !sysExists {
		sysIdx = uint16(len(newAccounts))
		newAccounts = append(newAccounts, solana.SystemProgramID)
		accountMap[solana.SystemProgramID] = sysIdx
	}

	newInstructions[len(newInstructions)-1] = solana.CompiledInstruction{
		ProgramIDIndex: sysIdx,
		Accounts:       []uint16{fromIdx, toIdx},
		Data:           tipData,
	}

	newMsg := solana.Message{
		Header:          msg.Header,
		AccountKeys:     newAccounts,
		RecentBlockhash: msg.RecentBlockhash,
		Instructions:    newInstructions,
	}

	if !fromExists {
		newMsg.Header.NumRequiredSignatures++
	}

	newTx := &solana.Transaction{
		Message: newMsg,
	}

	_, err := newTx.Sign(func(pub solana.PublicKey) *solana.PrivateKey {
		if pub.Equals(wallet.PublicKey()) {
			return &wallet
		}
		return nil
	})
	if err != nil {
		return solana.Signature{}, fmt.Errorf("sign failed: %w", err)
	}

	txBytes, err := newTx.MarshalBinary()
	if err != nil {
		return solana.Signature{}, fmt.Errorf("marshal failed: %w", err)
	}
	txBase64 := base64.StdEncoding.EncodeToString(txBytes)

	reqBody := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "sendTransaction",
		"params": []interface{}{
			txBase64,
			map[string]interface{}{
				"encoding": "base64",
			},
		},
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://sender.helius-rpc.com/fast", bytes.NewReader(bodyBytes))
	if err != nil {
		return solana.Signature{}, fmt.Errorf("request create failed: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return solana.Signature{}, fmt.Errorf("send failed: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Result string `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return solana.Signature{}, fmt.Errorf("decode failed: %w", err)
	}

	if result.Error != nil {
		return solana.Signature{}, fmt.Errorf("rpc error: %s", result.Error.Message)
	}

	sig := solana.MustSignatureFromBase58(result.Result)

	for i := 0; i < 5; i++ {
		time.Sleep(400 * time.Millisecond)

		statuses, err := client.GetSignatureStatuses(ctx, true, sig)
		if err != nil {
			continue
		}

		if len(statuses.Value) > 0 && statuses.Value[0] != nil {
			status := statuses.Value[0]
			if status.ConfirmationStatus == rpc.ConfirmationStatusConfirmed ||
				status.ConfirmationStatus == rpc.ConfirmationStatusFinalized {
				return sig, nil
			}
			if status.Err != nil {
				return solana.Signature{}, fmt.Errorf("tx failed: %v", status.Err)
			}
		}
	}

	return solana.Signature{}, fmt.Errorf("tx not confirmed after 5 polls: %s", sig)
}


type Token struct {
	ID        int
	Mint      string
	Score     float64
	Honeypot  bool
	TopHolder float64
	LiqUSD    float64
	Bought    bool
	Sold      bool
	TxBuy     sql.NullString
	TxSell    sql.NullString
	BuyPrice  float64
	BuyTime   time.Time
}

type RugCheckResponse struct {
	TokenMeta struct {
		Name   string `json:"name"`
		Symbol string `json:"symbol"`
	} `json:"tokenMeta"`
	Score      float64 `json:"score"`
	Risks      []Risk  `json:"risks"`
	TopHolders []struct {
		Address string  `json:"address"`
		Pct     float64 `json:"pct"`
	} `json:"topHolders"`
	Markets []struct {
		LiquidityA interface{} `json:"liquidityA"`
		LiquidityB interface{} `json:"liquidityB"`
	} `json:"markets"`
}

type Risk struct {
	Name        string `json:"name"`
	Level       string `json:"level"`
	Description string `json:"description"`
}

type PriceData struct {
	Price     float64
	Timestamp time.Time
}

var (
	db              *sql.DB
	wallet          solana.PrivateKey
	client          *rpc.Client
	senderClient    *rpc.Client
	wsClient        *ws.Client
	jitoTipAccounts = []string{
		"96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
		"HFqU5x63VTqvQss8hp11i4bVNa1xJZmCkrhGnVw6nNYS",
		"Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
		"ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
		"DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
		"ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
		"DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
		"3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
	}
	pumpID  = solana.MustPublicKeyFromBase58("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
	solMint = "So11111111111111111111111111111111111111112"

	jupiterQuoteURL = "https://api.jup.ag/swap/v1/quote"
	jupiterSwapURL  = "https://api.jup.ag/swap/v1/swap"

	maxBuySOL         = 0.05
	profitTargetPct   = 100.0
	stopLossPct       = 25.0
	maxTopHolder      = 15.0
	maxRiskScore      = 50.0
	jitoTipLamports   = uint64(200000)
	minSlippage       = 200
	maxSlippage       = 500
	minPriceThreshold = 0.0001
	buyCooldown       = 2 * time.Second

	priceCache      = make(map[string][]PriceData)
	priceCacheMu    sync.RWMutex
	blacklist       = make(map[string]bool)
	blacklistMu     sync.RWMutex
	activePosition  sync.Mutex
	lastFailedBuy   time.Time
	lastFailedBuyMu sync.Mutex
	httpClient      *http.Client
)

func init() {
	rand.Seed(time.Now().UnixNano())

	if err := godotenv.Load(); err != nil {
		log.Println("no .env - using defaults")
	}

	httpClient = &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	var err error
	db, err = sql.Open("sqlite3", "sniper.db")
	if err != nil {
		log.Fatal("db open failed:", err)
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS tokens (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			mint TEXT UNIQUE,
			name TEXT,
			symbol TEXT,
			score REAL,
			honeypot BOOLEAN DEFAULT FALSE,
			topholder REAL,
			liqusd REAL,
			bought BOOLEAN DEFAULT FALSE,
			sold BOOLEAN DEFAULT FALSE,
			txbuy TEXT,
			txsell TEXT,
			buy_price REAL,
			sell_price REAL,
			buy_time DATETIME,
			sell_time DATETIME,
			pnl_pct REAL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS blacklist (
			address TEXT PRIMARY KEY,
			reason TEXT,
			added_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);
		CREATE INDEX IF NOT EXISTS idx_tokens_mint ON tokens(mint);
		CREATE INDEX IF NOT EXISTS idx_tokens_bought ON tokens(bought);
	`)
	if err != nil {
		log.Fatal("table failed:", err)
	}

	loadBlacklist()

	keyStr := os.Getenv("PRIVATE_KEY")
	if keyStr == "" {
		log.Fatal("PRIVATE_KEY missing in .env")
	}

	wallet, err = solana.PrivateKeyFromBase58(keyStr)
	if err != nil {
		log.Fatal("bad key:", err)
	}

	apiKey := os.Getenv("HELIUS_API_KEY")
	if apiKey == "" {
		log.Fatal("HELIUS_API_KEY missing in .env")
	}

	rpcURL := fmt.Sprintf("https://mainnet.helius-rpc.com/?api-key=%s", apiKey)
	wsURL := fmt.Sprintf("wss://mainnet.helius-rpc.com/?api-key=%s", apiKey)

	senderURL := os.Getenv("HELIUS_SENDER_URL")
	if senderURL == "" {
		senderURL = "https://mainnet.helius-rpc.com/?api-key=" + apiKey
	}

	client = rpc.New(rpcURL)
	senderClient = rpc.New(senderURL)
	wsClient, err = ws.Connect(context.Background(), wsURL)
	if err != nil {
		log.Fatal("ws failed:", err)
	}

	if v := os.Getenv("MAX_BUY_SOL"); v != "" {
		fmt.Sscanf(v, "%f", &maxBuySOL)
	}
	if v := os.Getenv("PROFIT_TARGET_PCT"); v != "" {
		fmt.Sscanf(v, "%f", &profitTargetPct)
	}
	if v := os.Getenv("STOP_LOSS_PCT"); v != "" {
		fmt.Sscanf(v, "%f", &stopLossPct)
	}
	if v := os.Getenv("JITO_TIP_LAMPORTS"); v != "" {
		fmt.Sscanf(v, "%d", &jitoTipLamports)
	}

	log.Printf("wallet: %s", wallet.PublicKey().String())
	log.Printf("config: buy=%.3f SOL, profit=%.0f%%, stop=%.0f%%, tip=%d lamports",
		maxBuySOL, profitTargetPct, stopLossPct, jitoTipLamports)
	log.Printf("jupiter: %s", jupiterQuoteURL)
	log.Printf("sender: %s", senderURL)
}

func loadBlacklist() {
	rows, err := db.Query("SELECT address FROM blacklist")
	if err != nil {
		return
	}
	defer rows.Close()

	blacklistMu.Lock()
	defer blacklistMu.Unlock()

	for rows.Next() {
		var addr string
		if err := rows.Scan(&addr); err == nil {
			blacklist[addr] = true
		}
	}
	log.Printf("loaded %d blacklisted addresses", len(blacklist))
}

func addToBlacklist(address, reason string) {
	blacklistMu.Lock()
	blacklist[address] = true
	blacklistMu.Unlock()
	db.Exec("INSERT OR IGNORE INTO blacklist (address, reason) VALUES (?, ?)", address, reason)
}

func isBlacklisted(address string) bool {
	blacklistMu.RLock()
	defer blacklistMu.RUnlock()
	return blacklist[address]
}

func getRandomSlippage() int {
	return minSlippage + rand.Intn(maxSlippage-minSlippage+1)
}

func checkSOLBalance(ctx context.Context) (uint64, error) {
	balance, err := client.GetBalance(ctx, wallet.PublicKey(), rpc.CommitmentConfirmed)
	if err != nil {
		return 0, err
	}
	return balance.Value, nil
}

func hasEnoughSOL(ctx context.Context) bool {
	balance, err := checkSOLBalance(ctx)
	if err != nil {
		log.Printf("balance check failed: %v", err)
		return false
	}
	required := uint64(maxBuySOL*1e9) + jitoTipLamports*2 + 10000
	if balance < required {
		log.Printf("low SOL: have %d, need %d lamports", balance, required)
		return false
	}
	return true
}

func isInCooldown() bool {
	lastFailedBuyMu.Lock()
	defer lastFailedBuyMu.Unlock()
	return time.Since(lastFailedBuy) < buyCooldown
}

func setFailedBuy() {
	lastFailedBuyMu.Lock()
	lastFailedBuy = time.Now()
	lastFailedBuyMu.Unlock()
}

func getDynamicTip(ctx context.Context) uint64 {
	fees, err := client.GetRecentPrioritizationFees(ctx, nil)
	if err != nil || len(fees) == 0 {
		return jitoTipLamports
	}

	var total uint64
	var count int
	for _, f := range fees {
		if f.PrioritizationFee > 0 {
			total += f.PrioritizationFee
			count++
		}
	}

	if count == 0 {
		return jitoTipLamports
	}

	avgFee := total / uint64(count)
	dynamicTip := uint64(float64(avgFee) * 1.5)

	if dynamicTip < jitoTipLamports {
		return jitoTipLamports
	}
	if dynamicTip > jitoTipLamports*5 {
		return jitoTipLamports * 5
	}

	return dynamicTip
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	defer db.Close()
	defer wsClient.Close()

	balance, err := checkSOLBalance(ctx)
	if err != nil {
		log.Printf("initial balance check failed: %v", err)
	} else {
		log.Printf("wallet balance: %.4f SOL", float64(balance)/1e9)
	}

	var wg sync.WaitGroup

	wg.Add(1)
	go monitorPump(ctx, &wg)

	wg.Add(1)
	go priceMonitor(ctx, &wg)

	log.Println("sniper live - waiting for tokens...")
	sendTelegram(fmt.Sprintf("🚀 Sniper started\nWallet: %s\nBalance: %.4f SOL\nJupiter: v1 API",
		wallet.PublicKey().String()[:16]+"...", float64(balance)/1e9))

	<-ctx.Done()
	log.Println("shutting down...")
	sendTelegram("🛑 Sniper stopped")
	wg.Wait()
	log.Println("shutdown complete")
}

func monitorPump(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := subscribeAndListen(ctx); err != nil {
			log.Printf("subscription error: %v, reconnecting...", err)
			time.Sleep(2 * time.Second)
		}
	}
}

func subscribeAndListen(ctx context.Context) error {
	sub, err := wsClient.LogsSubscribeMentions(pumpID, rpc.CommitmentProcessed)
	if err != nil {
		return fmt.Errorf("sub fail: %w", err)
	}
	defer sub.Unsubscribe()

	log.Println("subscribed to pump.fun logs (processed)")

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		msg, err := sub.Recv(ctx)
		if err != nil {
			return fmt.Errorf("recv error: %w", err)
		}

		if msg.Value.Err != nil {
			continue
		}

		logsJoined := strings.Join(msg.Value.Logs, " ")
		if !strings.Contains(logsJoined, "Create") {
			continue
		}

		tx := msg.Value.Signature
		detectTime := time.Now()

		go func(sig solana.Signature, detected time.Time) {
			processCreate(ctx, sig, detected)
		}(tx, detectTime)
	}
}

func processCreate(ctx context.Context, tx solana.Signature, detectTime time.Time) {
	var res *rpc.GetTransactionResult
	var err error

	for i := 0; i < 3; i++ {
		res, err = client.GetTransaction(ctx, tx, &rpc.GetTransactionOpts{
			MaxSupportedTransactionVersion: func() *uint64 { v := uint64(0); return &v }(),
			Commitment:                     rpc.CommitmentConfirmed,
		})
		if err == nil && res != nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if err != nil || res == nil {
		return
	}

	mint := parseMint(res)
	if mint == "" {
		return
	}

	latency := time.Since(detectTime)
	log.Printf("new token: %s (latency: %dms)", mint, latency.Milliseconds())

	checkToken(ctx, mint, tx, detectTime)
}

func parseMint(tx *rpc.GetTransactionResult) string {
	if tx.Meta == nil || tx.Meta.LogMessages == nil {
		return ""
	}

	for _, logMsg := range tx.Meta.LogMessages {
		if strings.Contains(logMsg, "Create") {
			re := regexp.MustCompile(`mint[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})`)
			match := re.FindStringSubmatch(logMsg)
			if len(match) > 1 {
				return strings.TrimSpace(match[1])
			}

			re2 := regexp.MustCompile(`Create\(([^)]+)\)`)
			match2 := re2.FindStringSubmatch(logMsg)
			if len(match2) > 1 {
				return strings.TrimSpace(match2[1])
			}
		}
	}

	for _, bal := range tx.Meta.PostTokenBalances {
		mintStr := bal.Mint.String()
		if mintStr != "" && mintStr != solMint {
			return mintStr
		}
	}

	return ""
}

func checkToken(ctx context.Context, mint string, tx solana.Signature, detectTime time.Time) {
	if !activePosition.TryLock() {
		log.Printf("skip %s - already in position", mint)
		return
	}
	defer activePosition.Unlock()

	if isInCooldown() {
		log.Printf("skip %s - in cooldown after failed buy", mint)
		return
	}

	if !hasEnoughSOL(ctx) {
		log.Printf("skip %s - insufficient SOL", mint)
		return
	}

	var m string
	err := db.QueryRow("SELECT mint FROM tokens WHERE mint = ? AND bought = 1", mint).Scan(&m)
	if err == nil {
		log.Printf("already bought %s, skipping", mint)
		return
	}

	rugCheck, err := getRugCheckReport(ctx, mint)
	if err != nil {
		log.Printf("rugcheck failed for %s: %v", mint, err)
		return
	}

	if rugCheck.Score > maxRiskScore {
		log.Printf("skip %s - risk score %.0f > %.0f", mint, rugCheck.Score, maxRiskScore)
		return
	}

	var topHolder float64
	for _, h := range rugCheck.TopHolders {
		if h.Pct > topHolder {
			topHolder = h.Pct
		}
		if isBlacklisted(h.Address) {
			log.Printf("skip %s - blacklisted holder %s", mint, h.Address[:8])
			return
		}
	}

	if topHolder > maxTopHolder {
		log.Printf("skip %s - top holder %.1f%% > %.1f%%", mint, topHolder, maxTopHolder)
		return
	}

	for _, risk := range rugCheck.Risks {
		if risk.Level == "critical" || risk.Level == "high" {
			riskLower := strings.ToLower(risk.Name)
			if strings.Contains(riskLower, "honeypot") ||
				strings.Contains(riskLower, "mint") ||
				strings.Contains(riskLower, "freeze") ||
				strings.Contains(riskLower, "rug") {
				log.Printf("skip %s - %s risk: %s", mint, risk.Level, risk.Name)
				return
			}
		}
	}

	var liquidity float64
	for _, m := range rugCheck.Markets {
		liquidity += parseFloat(m.LiquidityA) + parseFloat(m.LiquidityB)
	}

	totalLatency := time.Since(detectTime)
	log.Printf("token %s passed: score=%.0f, top=%.1f%%, liq=$%.0f, latency=%dms",
		mint, rugCheck.Score, topHolder, liquidity, totalLatency.Milliseconds())

	buy(ctx, mint, rugCheck, topHolder, liquidity)
}

func getRugCheckReport(ctx context.Context, mint string) (*RugCheckResponse, error) {
	reqURL := fmt.Sprintf("https://api.rugcheck.xyz/v1/tokens/%s/report", mint)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, err
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rugcheck status %d", resp.StatusCode)
	}

	var report RugCheckResponse
	if err := json.NewDecoder(resp.Body).Decode(&report); err != nil {
		return nil, err
	}

	return &report, nil
}

func getJupiterQuote(ctx context.Context, inputMint, outputMint string, amount uint64, slippage int) (map[string]interface{}, error) {
	quoteURL := fmt.Sprintf("%s?inputMint=%s&outputMint=%s&amount=%d&slippageBps=%d&swapMode=ExactIn",
		jupiterQuoteURL, inputMint, outputMint, amount, slippage)

	var quote map[string]interface{}
	for i := 0; i < 3; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, quoteURL, nil)
		if err != nil {
			continue
		}

		resp, err := httpClient.Do(req)
		if err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}

		if resp.StatusCode == http.StatusOK {
			json.NewDecoder(resp.Body).Decode(&quote)
			resp.Body.Close()
			break
		}
		resp.Body.Close()
		time.Sleep(50 * time.Millisecond)
	}

	if quote == nil {
		return nil, fmt.Errorf("quote failed after retries")
	}

	if errMsg, ok := quote["error"].(string); ok {
		return nil, fmt.Errorf("quote error: %s", errMsg)
	}

	return quote, nil
}

func getJupiterSwap(ctx context.Context, quote map[string]interface{}, priorityFee uint64) (string, error) {
	swapBody := map[string]interface{}{
		"quoteResponse":             quote,
		"userPublicKey":             wallet.PublicKey().String(),
		"wrapAndUnwrapSol":          true,
		"dynamicComputeUnitLimit":   true,
		"prioritizationFeeLamports": priorityFee,
	}

	body, _ := json.Marshal(swapBody)

	var swap map[string]interface{}
	for i := 0; i < 3; i++ {
		swapReq, err := http.NewRequestWithContext(ctx, http.MethodPost, jupiterSwapURL, bytes.NewReader(body))
		if err != nil {
			continue
		}
		swapReq.Header.Set("Content-Type", "application/json")

		swapResp, err := httpClient.Do(swapReq)
		if err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}

		if swapResp.StatusCode == http.StatusOK {
			json.NewDecoder(swapResp.Body).Decode(&swap)
			swapResp.Body.Close()
			break
		}
		swapResp.Body.Close()
		time.Sleep(50 * time.Millisecond)
	}

	if swap == nil {
		return "", fmt.Errorf("swap failed after retries")
	}

	if errMsg, ok := swap["error"].(string); ok {
		return "", fmt.Errorf("swap error: %s", errMsg)
	}

	txBase64, ok := swap["swapTransaction"].(string)
	if !ok {
		return "", fmt.Errorf("no swapTransaction in response")
	}

	return txBase64, nil
}

func simulateTransaction(ctx context.Context, tx *solana.Transaction) error {
	sim, err := client.SimulateTransaction(ctx, tx)
	if err != nil {
		return fmt.Errorf("simulate failed: %w", err)
	}

	if sim.Value.Err != nil {
		return fmt.Errorf("simulation error: %v", sim.Value.Err)
	}

	return nil
}

func buy(ctx context.Context, mint string, rugCheck *RugCheckResponse, topHolder, liquidity float64) {
	startTime := time.Now()
	log.Printf("BUY %s (%s)", mint, rugCheck.TokenMeta.Symbol)

	if !hasEnoughSOL(ctx) {
		log.Println("insufficient SOL for buy")
		setFailedBuy()
		return
	}

	slippage := getRandomSlippage()
	amountLamports := uint64(maxBuySOL * 1e9)

	quote, err := getJupiterQuote(ctx, solMint, mint, amountLamports, slippage)
	if err != nil {
		log.Printf("quote failed: %v", err)
		setFailedBuy()
		return
	}

	if inAmount, ok := quote["inAmount"].(string); ok {
		var amount uint64
		fmt.Sscanf(inAmount, "%d", &amount)
		if amount == 0 {
			log.Println("zero inAmount - no liquidity")
			setFailedBuy()
			return
		}
	}

	dynamicTip := getDynamicTip(ctx)
	txBase64, err := getJupiterSwap(ctx, quote, dynamicTip)
	if err != nil {
		log.Printf("swap failed: %v", err)
		setFailedBuy()
		return
	}

	txBytes, err := base64.StdEncoding.DecodeString(txBase64)
	if err != nil {
		log.Println("base64 decode fail:", err)
		setFailedBuy()
		return
	}

	transaction, err := solana.TransactionFromBytes(txBytes)
	if err != nil {
		log.Println("tx parse fail:", err)
		setFailedBuy()
		return
	}

	signatures, err := transaction.Sign(func(pub solana.PublicKey) *solana.PrivateKey {

		if pub.Equals(wallet.PublicKey()) {
			return &wallet
		}
		return nil
	})
	if err != nil {
		log.Println("tx sign fail:", err)
		setFailedBuy()
		return
	}

	if err := simulateTransaction(ctx, transaction); err != nil {
		log.Printf("simulation failed: %v", err)
		setFailedBuy()
		return
	}

        log.Printf("buy signed with %d signatures, slippage=%dbps", len(signatures), slippage)

	sig, err := sendWithRetry(ctx, transaction, 3)
	if err != nil {
		log.Println("send fail:", err)
		setFailedBuy()
		return
	}

	buyTime := time.Now()
	totalLatency := buyTime.Sub(startTime)
	log.Printf("BUY TX: %s (latency: %dms)", sig.String(), totalLatency.Milliseconds())

	_, err = db.ExecContext(ctx,
		`INSERT INTO tokens (mint, name, symbol, score, honeypot, topholder, liqusd, bought, txbuy, buy_price, buy_time) 
		 VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)`,
		mint, rugCheck.TokenMeta.Name, rugCheck.TokenMeta.Symbol,
		rugCheck.Score, false, topHolder, liquidity,
		sig.String(), maxBuySOL, buyTime,
	)
	if err != nil {
		log.Println("db insert fail:", err)
	}

	priceCacheMu.Lock()
	priceCache[mint] = []PriceData{{Price: maxBuySOL, Timestamp: buyTime}}
	priceCacheMu.Unlock()

	sendTelegram(fmt.Sprintf("💰 BUY %s (%s)\n%.4f SOL | Slip: %dbps | Tip: %d\nScore: %.0f | Top: %.1f%%\nhttps://solscan.io/tx/%s",
		rugCheck.TokenMeta.Symbol, mint[:8], maxBuySOL, slippage, dynamicTip, rugCheck.Score, topHolder, sig))

	go monitorPosition(ctx, mint, rugCheck.TokenMeta.Symbol, maxBuySOL, buyTime)
}

func sendWithRetry(ctx context.Context, tx *solana.Transaction, maxRetries int) (solana.Signature, error) {
	var lastErr error

	for i := 0; i < maxRetries; i++ {
		sig, err := senderClient.SendTransactionWithOpts(ctx, tx, rpc.TransactionOpts{
			SkipPreflight:       true,
			PreflightCommitment: rpc.CommitmentProcessed,
		})

		if err == nil {
			return sig, nil
		}

		lastErr = err
		log.Printf("send attempt %d failed: %v", i+1, err)

		if i < maxRetries-1 {
			time.Sleep(time.Duration(50*(i+1)) * time.Millisecond)
		}
	}

	return solana.Signature{}, lastErr
}

func monitorPosition(ctx context.Context, mint, symbol string, buyPrice float64, buyTime time.Time) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	timeout := time.After(5 * time.Minute)

	for {
		select {
		case <-ctx.Done():
			log.Printf("position monitor cancelled for %s", mint)
			executeSellWithReason(ctx, mint, symbol, buyPrice, "shutdown")
			return

		case <-timeout:
			log.Printf("position timeout for %s", mint)
			executeSellWithReason(ctx, mint, symbol, buyPrice, "timeout")
			return

		case <-ticker.C:
			currentPrice, err := getCurrentPrice(ctx, mint)
			if err != nil {
				continue
			}

			pnlPct := ((currentPrice - buyPrice) / buyPrice) * 100

			priceCacheMu.Lock()
			priceCache[mint] = append(priceCache[mint], PriceData{
				Price:     currentPrice,
				Timestamp: time.Now(),
			})
			if len(priceCache[mint]) > 100 {
				priceCache[mint] = priceCache[mint][50:]
			}
			priceCacheMu.Unlock()

			if pnlPct >= profitTargetPct {
				log.Printf("profit target hit for %s: %.1f%%", mint, pnlPct)
				executeSellWithReason(ctx, mint, symbol, buyPrice, "profit_target")
				return
			}

			if pnlPct <= -stopLossPct {
				log.Printf("stop loss hit for %s: %.1f%%", mint, pnlPct)
				executeSellWithReason(ctx, mint, symbol, buyPrice, "stop_loss")
				return
			}

			if detectDump(mint, buyPrice) {
				log.Printf("dump detected for %s", mint)
				executeSellWithReason(ctx, mint, symbol, buyPrice, "dump_detected")
				return
			}
		}
	}
}

func getCurrentPrice(ctx context.Context, mint string) (float64, error) {
	balance, err := getTokenBalance(ctx, mint)
	if err != nil || balance == 0 {
		return 0, fmt.Errorf("no balance")
	}

	slippage := getRandomSlippage()
	quote, err := getJupiterQuote(ctx, mint, solMint, balance, slippage)
	if err != nil {
		return 0, err
	}

	if outAmount, ok := quote["outAmount"].(string); ok {
		var lamports uint64
		fmt.Sscanf(outAmount, "%d", &lamports)
		return float64(lamports) / 1e9, nil
	}

	return 0, fmt.Errorf("no outAmount")
}

func detectDump(mint string, buyPrice float64) bool {
	priceCacheMu.RLock()
	prices := priceCache[mint]
	priceCacheMu.RUnlock()

	if len(prices) < 10 {
		return false
	}

	recent := prices[len(prices)-5:]
	older := prices[len(prices)-10 : len(prices)-5]

	var recentAvg, olderAvg float64
	for _, p := range recent {
		recentAvg += p.Price
	}
	for _, p := range older {
		olderAvg += p.Price
	}
	recentAvg /= float64(len(recent))
	olderAvg /= float64(len(older))

	if olderAvg < minPriceThreshold || recentAvg < minPriceThreshold {
		log.Printf("dump check: price below threshold for %s (%.6f)", mint, recentAvg)
		return false
	}

	dropPct := ((olderAvg - recentAvg) / olderAvg) * 100
	return dropPct > 30
}

func executeSellWithReason(ctx context.Context, mint, symbol string, buyPrice float64, reason string) {
	sig, soldPrice, err := executeSell(ctx, mint)
	if err != nil {
		log.Printf("sell failed for %s: %v", mint, err)
		return
	}

	pnlPct := ((soldPrice - buyPrice) / buyPrice) * 100
	emoji := "🟢"
	if pnlPct < 0 {
		emoji = "🔴"
	}

	db.Exec(`UPDATE tokens SET sold = 1, txsell = ?, sell_price = ?, sell_time = ?, pnl_pct = ? WHERE mint = ?`,
		sig, soldPrice, time.Now(), pnlPct, mint)

	priceCacheMu.Lock()
	delete(priceCache, mint)
	priceCacheMu.Unlock()

	sendTelegram(fmt.Sprintf("%s SELL %s\nReason: %s\nP/L: %.1f%%\nBuy: %.4f → Sell: %.4f SOL\nhttps://solscan.io/tx/%s",
		emoji, symbol, reason, pnlPct, buyPrice, soldPrice, sig))
}

func executeSell(ctx context.Context, mint string) (string, float64, error) {
	balance, err := getTokenBalance(ctx, mint)
	if err != nil {
		return "", 0, err
	}

	if balance == 0 {
		return "", 0, fmt.Errorf("zero balance")
	}

	log.Printf("selling %d tokens of %s", balance, mint)

	slippage := getRandomSlippage()
	quote, err := getJupiterQuote(ctx, mint, solMint, balance, slippage)
	if err != nil {
		return "", 0, err
	}

	if inAmount, ok := quote["inAmount"].(string); ok {
		var amount uint64
		fmt.Sscanf(inAmount, "%d", &amount)
		if amount == 0 {
			return "", 0, fmt.Errorf("zero inAmount - no liquidity")
		}
	}

	if outAmount, ok := quote["outAmount"].(string); ok {
		var amount uint64
		fmt.Sscanf(outAmount, "%d", &amount)
		if amount == 0 {
			return "", 0, fmt.Errorf("zero outAmount - no liquidity")
		}
	}

	var expectedSOL float64
	if outAmount, ok := quote["outAmount"].(string); ok {
		var lamports uint64
		fmt.Sscanf(outAmount, "%d", &lamports)
		expectedSOL = float64(lamports) / 1e9
	}

	dynamicTip := getDynamicTip(ctx)
	txBase64, err := getJupiterSwap(ctx, quote, dynamicTip)
	if err != nil {
		return "", 0, err
	}

	txBytes, _ := base64.StdEncoding.DecodeString(txBase64)
	transaction, err := solana.TransactionFromBytes(txBytes)
	if err != nil {
		return "", 0, fmt.Errorf("tx parse: %w", err)
	}

	signatures, err := transaction.Sign(func(pub solana.PublicKey) *solana.PrivateKey {
		if pub.Equals(wallet.PublicKey()) {
			return &wallet
		}
		return nil
	})
	if err != nil {
		return "", 0, fmt.Errorf("sign: %w", err)
	}

	if err := simulateTransaction(ctx, transaction); err != nil {
		return "", 0, fmt.Errorf("simulation: %w", err)
	}

        log.Printf("buy signed with %d signatures, slippage=%dbps", len(signatures), slippage)

	sig, err := sendWithRetry(ctx, transaction, 3)
	if err != nil {
		return "", 0, fmt.Errorf("send: %w", err)
	}

	log.Printf("SELL TX: %s", sig.String())
	return sig.String(), expectedSOL, nil
}

func getTokenBalance(ctx context.Context, mint string) (uint64, error) {
	mintPubkey := solana.MustPublicKeyFromBase58(mint)

	tokenAccounts, err := client.GetTokenAccountsByOwner(
		ctx,
		wallet.PublicKey(),
		&rpc.GetTokenAccountsConfig{
			Mint: &mintPubkey,
		},
		&rpc.GetTokenAccountsOpts{
			Commitment: rpc.CommitmentConfirmed,
		},
	)
	if err != nil {
		return 0, fmt.Errorf("get accounts: %w", err)
	}

	if len(tokenAccounts.Value) == 0 {
		return 0, fmt.Errorf("no token account for %s", mint)
	}

	for _, acc := range tokenAccounts.Value {
		data := acc.Account.Data.GetBinary()
		if data == nil || len(data) < 72 {
			continue
		}
		balance := binary.LittleEndian.Uint64(data[64:72])
		if balance > 0 {
			return balance, nil
		}
	}

	return 0, fmt.Errorf("zero balance for %s", mint)
}

func priceMonitor(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			logStats(ctx)
		}
	}
}

func logStats(ctx context.Context) {
	var total, wins, losses int
	var totalPnl float64

	rows, err := db.Query("SELECT pnl_pct FROM tokens WHERE sold = 1 AND pnl_pct IS NOT NULL")
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var pnl float64
		if err := rows.Scan(&pnl); err == nil {
			total++
			totalPnl += pnl
			if pnl > 0 {
				wins++
			} else {
				losses++
			}
		}
	}

	if total > 0 {
		winRate := float64(wins) / float64(total) * 100
		log.Printf("stats: %d trades | %.1f%% win rate | %.1f%% total P/L", total, winRate, totalPnl)
	}

	balance, err := checkSOLBalance(ctx)
	if err == nil {
		log.Printf("wallet: %.4f SOL", float64(balance)/1e9)
	}
}

func sendTelegram(msg string) {
	token := os.Getenv("TELEGRAM_TOKEN")
	chat := os.Getenv("CHAT_ID")
	if token == "" || chat == "" {
		return
	}

	apiURL := fmt.Sprintf(
		"https://api.telegram.org/bot%s/sendMessage?chat_id=%s&text=%s&parse_mode=HTML",
		token, chat, url.QueryEscape(msg),
	)

	resp, err := http.Get(apiURL)
	if err != nil {
		log.Println("telegram fail:", err)
		return
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
}
