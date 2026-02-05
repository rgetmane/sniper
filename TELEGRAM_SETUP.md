# Telegram Integration Setup

## Overview
The sniper bot is now fully integrated with Telegram for real-time notifications.

## Configuration

### Environment Variables (in `secrets.env`)
```env
TELEGRAM_TOKEN=8419179729:AAGubkiJq06wz-_xSN2zE-bY9wADnPUvpy0
CHAT_ID=732197719
```

## Features

### 1. Startup Notification
When the bot starts, it automatically sends a "Sniper alive" message to your Telegram:
```
🟢 Sniper alive

✅ Bot started successfully
⏰ 2026-02-04 18:51:47 UTC
```

### 2. Buy Notifications
When the bot buys a token:
```
🟢 BUY {mint} ({symbol})
{amount} SOL | Score: {score}
https://solscan.io/tx/{signature}
```

### 3. Sell Notifications
When the bot sells a token:
```
🟢 SELL {mint} ({symbol})  # Green if profit
🔴 SELL {mint} ({symbol})  # Red if loss
{reason} | PnL: {pnl} SOL
https://solscan.io/tx/{signature}
```

## Testing

### Test Telegram Connection
Run the test script to verify your Telegram integration:
```bash
cd /home/roman/sniper
go run test_telegram.go
```

Expected output:
```
2026/02/04 18:51:46 Testing Telegram connection...
2026/02/04 18:51:46 Token: 8419179729:AAGubkiJq...
2026/02/04 18:51:46 Chat ID: 732197719
2026/02/04 18:51:47 ✅ SUCCESS: Test message sent to Telegram!
2026/02/04 18:51:47 Check your Telegram to verify the message was received.
```

### Run the Bot
```bash
cd /home/roman/sniper
./sniper
```

The bot will:
1. Load configuration from `.env` and `secrets.env`
2. Initialize database, wallets, and RPC connections
3. **Send "Sniper alive" message to Telegram**
4. Start monitoring Pump.fun for new tokens
5. Send buy/sell notifications in real-time

## Troubleshooting

### No Messages Received
1. **Check bot token**: Verify `TELEGRAM_TOKEN` in `secrets.env`
2. **Check chat ID**: Verify `CHAT_ID` matches your Telegram user ID
3. **Bot not started**: Make sure you've started a chat with the bot first
4. **Check logs**: Look in `sniper.log` for error messages

### Get Your Chat ID
If you need to find your chat ID:
1. Message your bot on Telegram
2. Visit: `https://api.telegram.org/bot{YOUR_BOT_TOKEN}/getUpdates`
3. Look for `"chat":{"id":123456789}` in the response

### Create a New Bot
To create a new Telegram bot:
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot`
3. Follow the prompts to get your bot token
4. Update `TELEGRAM_TOKEN` in `secrets.env`

## Implementation Details

### Code Changes in `main.go`
Added startup notification after configuration logging:
```go
// Send startup notification to Telegram
sendTelegram("🟢 Sniper alive\n\n✅ Bot started successfully\n⏰ " + time.Now().Format("2006-01-02 15:04:05") + " UTC")
```

### Telegram Function
The `sendTelegram()` function:
- Reads `TELEGRAM_TOKEN` and `CHAT_ID` from environment
- Sends messages via Telegram Bot API
- Retries once on failure
- Logs errors but doesn't block execution

## Status

✅ **VERIFIED**: Telegram integration is working correctly
- Test message sent successfully
- Bot token and chat ID configured
- Startup notification added to main.go
- Ready for production use

## Next Steps

1. Start the sniper bot: `./sniper`
2. Monitor your Telegram for the "Sniper alive" message
3. Watch for buy/sell notifications as the bot trades
4. Check `sniper.log` for detailed operation logs
