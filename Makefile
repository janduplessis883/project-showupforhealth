install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@echo "🧽 Cleaned up successfully!"

all: install clean

# Build Disese_register only for use when building prediction dataset
disease_register:
	python showupforhealth/ml_functions/disease_register.py

# Build full pre-processed traing dataset
data:
	python showupforhealth/ml_functions/data.py


test:
	@pytest -v tests

# Specify package name
lint:
	@black showupforhealth/
