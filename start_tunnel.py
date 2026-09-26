"""
magicpin AI Challenge — Public Endpoint Exposure Helper
Helper utility to expose local bot.py (port 8080) to a public HTTPS URL.
"""

import sys
import subprocess
import shutil

def check_tools():
    print("=" * 60)
    print(" magicpin AI Challenge — Public Endpoint Setup")
    print("=" * 60)
    print("\nYour bot is listening on: http://localhost:8080")
    print("\nTo submit your bot to magicpin's judging portal, you need a public HTTPS URL.")
    print("Choose one of the following methods:\n")

    print("METHOD 1: Cloudflare Tunnel (No account needed, instant)")
    print("  Command: cloudflared tunnel --url http://localhost:8080")
    print("  Or run:  npx localtunnel --port 8080\n")

    print("METHOD 2: ngrok (Standard)")
    print("  Command: ngrok http 8080\n")

    print("=" * 60)
    print("Once started, copy the generated HTTPS URL (e.g. https://abc.ngrok-free.app)")
    print("and verify it with:")
    print("  curl https://<YOUR_URL>/v1/healthz")
    print("=" * 60)

if __name__ == "__main__":
    check_tools()
