#!/bin/bash
# Build locally and auto-deploy to DigitalOcean

cd /home/roman/sniper

echo "========================================="
echo "  BUILD & DEPLOY"
echo "========================================="
echo

# Step 1: Build locally
echo "Step 1: Building locally..."
go build -o sniper main.go

if [ $? -ne 0 ]; then
    echo "❌ Local build failed"
    exit 1
fi

echo "✓ Local build successful"
echo

# Step 2: Deploy to DO
echo "Step 2: Deploying to DigitalOcean..."
./deploy.sh

if [ $? -ne 0 ]; then
    echo "❌ Deployment failed"
    exit 1
fi

echo
echo "========================================="
echo "✅ Build & Deploy Complete!"
echo "========================================="
