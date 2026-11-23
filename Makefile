.PHONY: data analysis test notebook

data:
	python scripts/download_data.py

analysis:
	python scripts/run_analysis.py

test:
	python -m pytest

notebook:
	python -m jupyter nbconvert --to notebook --execute --inplace notebooks/credit_default_analysis.ipynb
