install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@rm -Rf ../data-showup/uploads/*/.DS_Store
	@echo "🧽 Cleaned up successfully!"

all: install clean

# Build Disese_register only for use when building prediction dataset
disease_register:
	python showupforhealth/ml_functions/disease_register.py

# Build full pre-processed traing dataset
train_data:
	python showupforhealth/ml_functions/data.py

app:
	@streamlit run showupforhealth/interface/app.py

app2:
	@streamlit run showupforhealth/interface/app2.py

predict:
	python showupforhealth/interface/model_predict.py $(filter-out $@,$(MAKECMDGOALS)) 

model_train:
	python showupforhealth/interface/model_train.py

real_predict:
	python showupforhealth/interface/main.py

predict_folder:
	python showupforhealth/interface/model_predict_folder.py $(filter-out $@,$(MAKECMDGOALS)) 

predict_testall:
	python showupforhealth/interface/model_predict_folder_test.py 

test:
	@pytest

# Specify package name
lint:
	@black showupforhealth/
