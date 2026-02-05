package main

import (
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	_ = godotenv.Load("secrets.env")
	_ = godotenv.Load()

	tok := os.Getenv("TELEGRAM_TOKEN")
	chat := os.Getenv("CHAT_ID")

	if tok == "" || chat == "" {
		log.Fatal("TELEGRAM_TOKEN or CHAT_ID not set in secrets.env")
	}

	log.Printf("Testing Telegram connection...")
	log.Printf("Token: %s...", tok[:20])
	log.Printf("Chat ID: %s", chat)

	msg := "🟢 Sniper alive\n\n✅ Bot started successfully\n⏰ " + time.Now().Format("2006-01-02 15:04:05") + " UTC"
	
	urlStr := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage?chat_id=%s&text=%s&parse_mode=HTML",
		tok, chat, url.QueryEscape(msg))

	resp, err := http.Get(urlStr)
	if err != nil {
		log.Fatalf("ERROR: Failed to send message: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		log.Fatalf("ERROR: Telegram API returned status %d", resp.StatusCode)
	}

	log.Println("✅ SUCCESS: Test message sent to Telegram!")
	log.Println("Check your Telegram to verify the message was received.")
}
