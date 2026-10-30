# Polymarket + Grok Quantitative (+EV), Arbitrage & 100,000-Unit Paper Trading Bot

A complete 7-module quantitative prediction market scanner and automated virtual paper-trading fund.

## Key Features
1. **100,000-Unit Virtual Bankroll (`paper_trading_engine.py`)**:
   - Starts with **100,000.00 units** of virtual capital stored in SQLite (`paper_account` and `paper_positions`).
   - Automatically executes virtual bets on actionable `+EV`, `high_conviction` (ROI >= 100%), and `arb` opportunities using Quarter-Kelly sizing (`allocate_portfolio_kelly`) capped by real orderbook VWAP depth.
   - Marks open positions to market (MTM) on every scan using live Polymarket CLOB order books and automatically settles winning (`best_bid >= 0.98`) and losing (`best_ask <= 0.02`) contracts back into your unit balance.
2. **Automatic 4-Hour Unit & Bet Report**:
   - Every 4 hours (`PAPER_REPORT_HOURS=4.0`), the bot dispatches a comprehensive **4-Hour Paper Bankroll & Bet Report** to Telegram showing Starting Capital (`100,000.00 u`), Current Total Equity, Available Cash, Capital in Open Bets, Realized & Unrealized PnL, Win Rate, itemized Open Positions, Settled Bets since the last report, and a **Grok AI Desk Note** summarizing portfolio performance.
3. **GitHub Actions Ready (`.github/workflows/polymarket_paper_bot.yml`)**:
   - Runs automatically on GitHub's cloud runners, checkpoints the SQLite WAL file into `data/pm_engine.sqlite3`, and commits the updated 100k-unit ledger back to your repository so state is never lost between runs.

## Quick Start on GitHub
1. Create a new **Private** GitHub repository and upload all files from this ZIP archive (including the hidden `.github/workflows/polymarket_paper_bot.yml` folder).
2. In your GitHub repository, go to **Settings -> Secrets and variables -> Actions** and add:
   - `XAI_API_KEY` (your xAI Grok API key)
   - `TELEGRAM_BOT_TOKEN` (your Telegram bot token from `@BotFather`)
   - `TELEGRAM_CHAT_ID` (your Telegram chat ID)
   - `ODDS_API_KEY` *(optional, for Pinnacle/Betfair sharp sportsbook de-vigging)*
3. Go to the **Actions** tab, select **Polymarket + Grok 100k-Unit Paper Trading Bot**, and click **Run workflow** (check `force_paper_report = true` on your first run to immediately receive your initial 100,000-unit report on Telegram).
