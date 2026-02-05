#!/bin/bash
# Daily backup script for sniper bot
# Run via cron: 0 0 * * * /home/roman/sniper/backup.sh

cd /home/roman/sniper

DATE=$(date +%Y-%m-%d)
BACKUP_NAME="sniper-backup-${DATE}.zip"
BACKUP_DIR="/home/roman/backup"

echo "[$(date)] Starting backup..."

# Create backup directory if needed
mkdir -p "$BACKUP_DIR"

# Create backup zip
zip -q "$BACKUP_NAME" \
    sniper.log \
    sniper.db \
    sniper.db-shm \
    sniper.db-wal \
    .env \
    secrets.env \
    main.go

if [ $? -eq 0 ]; then
    echo "[$(date)] ✓ Backup created: $BACKUP_NAME"
else
    echo "[$(date)] ❌ Backup failed"
    exit 1
fi

# Upload to DigitalOcean if configured
if [ -f ~/.ssh/id_ed25519 ]; then
    scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no "$BACKUP_NAME" roman@206.81.4.22:/home/roman/backup/ 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] ✓ Uploaded to DO: $BACKUP_NAME"
    else
        echo "[$(date)] ⚠️  DO upload failed, keeping local copy"
    fi
fi

# Move to backup directory
mv "$BACKUP_NAME" "$BACKUP_DIR/"
echo "[$(date)] ✓ Backup saved to $BACKUP_DIR/$BACKUP_NAME"

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "sniper-backup-*.zip" -mtime +7 -delete
echo "[$(date)] ✓ Cleaned old backups (>7 days)"

echo "[$(date)] Backup complete"
