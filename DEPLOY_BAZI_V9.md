# Deploy BAZI BTC V9 Learning

1. Install Python 3.10 or newer.
2. Put `bazi_btc_v9_learning.py` and `requirements.txt` in the same folder.
3. Open a terminal in that folder and run:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   py -m pip install -r requirements.txt
   streamlit run bazi_btc_v9_learning.py
   ```

The browser opens automatically. Enter the contract's **Price to Beat** and its live
**NOW/reference price**. The app automatically uses the next UTC quarter-hour expiry.
Press **Log this prediction** once per contract. After expiry, enter the official
contract settlement value under **Learning & outcomes**, then press **Resolve and learn**.

The learning history is stored beside the app in `bazi_learning.sqlite3`. Back up that
file before moving computers or replacing the folder. Do not delete it during upgrades.

## Important behavior

- Coinbase is the primary movement/feature feed, not the contract anchor.
- The entered contract NOW/reference value anchors the forecast to the contract.
- Kraken is secondary confirmation only.
- The model says WAIT when evidence or data quality is inadequate.
- Learning begins influencing forecasts after 30 manually verified outcomes and is
  deliberately limited so a small or biased history cannot manufacture confidence.
- Automatically resolved Coinbase proxy outcomes appear in history but never train the model.
