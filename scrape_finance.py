from langchain_community.document_loaders import WebBaseLoader
import os

os.makedirs("data", exist_ok=True)

urls = [
    # Crypto
    "https://www.investopedia.com/terms/b/bitcoin.asp",
    "https://www.investopedia.com/terms/e/ethereum.asp",
    "https://www.investopedia.com/terms/c/cryptocurrency.asp",
    "https://www.investopedia.com/terms/b/blockchain.asp",
    "https://www.investopedia.com/terms/s/smart-contracts.asp",
    "https://www.investopedia.com/defi-decentralized-finance-5113835",
    # Investissement
    "https://www.investopedia.com/terms/i/investment.asp",
    "https://www.investopedia.com/terms/s/stock.asp",
    "https://www.investopedia.com/terms/p/portfolio.asp",
    "https://www.investopedia.com/terms/d/diversification.asp",
    "https://www.investopedia.com/terms/r/riskmanagement.asp",
    "https://www.investopedia.com/terms/d/dollarcostaveraging.asp",
    # Analyse
    "https://www.investopedia.com/terms/t/technical-analysis.asp",
    "https://www.investopedia.com/terms/f/fundamentalanalysis.asp",
    "https://www.investopedia.com/terms/r/rsi.asp",
]

print("Scraping en cours...")
loader = WebBaseLoader(urls)
docs = loader.load()

for i, doc in enumerate(docs):
    filename = f"data/finance_{i+1}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(doc.page_content)
    print(f" {filename} — {len(doc.page_content)} caractères")

print(f"\n {len(docs)} documents prêts dans data/")