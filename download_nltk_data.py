import nltk

# Define the data packages your project needs
packages = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']

for package in packages:
    try:
        print(f"Downloading '{package}'...")
        nltk.download(package)
        print(f"✅ '{package}' downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download '{package}': {e}")