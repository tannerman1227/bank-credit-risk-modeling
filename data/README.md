# Data

The model is trained on the UCI Machine Learning Repository file
**Default of Credit Card Clients** (Yeh and Lien, 2009): 30,000 Taiwanese
credit-card accounts, with six months of statement history and an indicator
for default in the following month. The repository distributes it as a legacy
Excel workbook.

`python scripts/download_data.py` downloads that workbook into `data/raw/`
and converts it to CSV. The full extract is gitignored.

`sample/credit_card_sample.csv` is a 250-row stratified excerpt committed so
the column layout is visible without a download. It is not a modeling sample.
The training script will not use it.

License of the source file: Creative Commons Attribution 4.0 (UCI ML Repository).
Cite Yeh, I-Cheng, and Che-hui Lien. 2009. "The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit
card clients." *Expert Systems with Applications* 36 (2): 2473–2480.
