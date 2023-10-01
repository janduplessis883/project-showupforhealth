<img src="https://github.com/janduplessis883/project-showupforhealth/blob/master/images/Show_Up_for_Health.png?raw=true" width="900">

### Who We Are

We're a team of dedicated data scientists, healthcare professionals, and software engineers based in West London. Working with Brompton Health PCN,we're united by one mission: to modernise the way primary care is delivered and make healthcare more efficient for everyone involved.

### Our Inspiration

The NHS faces an annual cost of approximately £1 billion due to missed appointments, commonly known as DNAs (Did Not Attends). These DNAs not only drain resources but also extend wait times for other patients who could have used those slots. With a staggering 4% DNA rate among 140,000 patients in our primary care network alone, we decided it was time for a change.

### The Problem We're Solving

While the issue of DNAs has gained some attention, most existing predictive models are focused on secondary care. This leaves primary care—the frontline of healthcare—underrepresented in data-driven solutions. Moreover, telephone appointments are often not counted, concealing the true scale of the issue.

## **Our Solution: Show Up for Health**

### Cutting-Edge Technology

Our app harnesses the power of deep learning to predict the likelihood of DNAs in primary care appointments. We're not just looking at past attendance records; we're also integrating variables like health indicators, local weather conditions, index of multiple deprivation and more, to give healthcare providers a more holistic understanding of patient behaviour.

### Features

- **Deep Learning Algorithms**: Trained on a rich dataset that captures the multi-dimensionality of healthcare delivery.
- **User-Friendly Dashboard**: We provide real-time analytics in a simple, intuitive dashboard accessible via any web browser.
- **Dynamic Updates**: As more data is collected, our algorithms adapt and improve, ensuring the most accurate predictions possible.

### Who Can Benefit

- **Healthcare Providers**: Optimise your appointment scheduling and reduce overhead costs.
- **Patients**: Benefit from more timely healthcare delivery, as reducing DNAs means quicker access to healthcare providers.
- **NHS**: A reduction in DNAs could save millions, if not billions, redirecting funds to where they're needed most.

## **Join Us**

We invite you to be part of this revolutionary journey. Whether you're a healthcare provider in the Brompton Health PCN, or a curious individual interested in the future of healthcare, we want to hear from you.

For more information, please feel free to contact us at jan.duplessis@nhs.net

## The Team 👥

- [Jan du Plessis](https://www.linkedin.com/in/jan-du-plessis-b72806244?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BKL4sIxaYRwyklMNNQ2cG8w%3D%3D)
- [Micheal Melis](https://www.linkedin.com/in/michael-melis-cfa-28154b127?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BZu0IKuZFR%2BS4MRRasMqR5A%3D%3D)
- [Fabio Sparano](https://www.linkedin.com/in/fabiosparano?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BXixlqOj%2FSfGasys%2FZa%2B1Eg%3D%3D)
- [Alessio Robotti](https://www.linkedin.com/in/alex-robotti-794160255?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bfbvq4OUwTvKx3eWpEsddrw%3D%3D)

## First Impressions
### Introduction

Missed medical appointments, also known as did not attends (DNAs), are a major challenge faced by healthcare systems worldwide. DNAs lead to negative impacts on patient health, provider productivity, and system efficiency. In the UK's National Health Service (NHS), DNAs are estimated to cost over £1 billion annually, draining valuable resources from an already overburdened system. Reducing DNA rates can lead to significant cost savings and improved access to care. However, this requires a thorough understanding of the factors driving missed appointments and effective interventions tailored to patients' needs.

This paper summarizes a data science project conducted in collaboration with Brompton Health Primary Care Network (PCN) to investigate DNA prediction in the primary care setting. Using real-world clinical data from multiple GP practices, we engineered features spanning demographics, appointment details, clinical factors, and geospatial data to model DNA risk. A deep neural network framework enabled learning from high-dimensional data with imbalanced classes. The goals were to improve on predictive performance over prior secondary care studies, identify new drivers of DNA behavior, and suggest targeted interventions to improve adherence.

### Background

DNAs have been widely studied, but predominantly in secondary care settings involving specialist clinics and hospitals. Fewer studies have leveraged primary care data. Secondary care studies have identified drivers of DNA risk including younger age, minority ethnicity status, public insurance, prior DNA history, and long wait times between scheduling and appointments. However, these findings may not translate directly to primary care.

Several studies have applied machine learning techniques like neural networks and random forests to predict DNA risk at time of scheduling. A University of Reading study used deep learning on 3 million NHS records and achieved a 91% AUC. However, models were trained on secondary care data. A study of US pediatric clinics used random forests and achieved an AUC of 0.82, incorporating weather data. However, the dataset was from a single hospital's specialty clinics.

Our study focuses specifically on DNA prediction using primary care data on a breadth of clinical, demographic, and geospatial factors. DNA rates are lower than secondary care, posing an additional modeling challenge. The goals were to investigate new sources of predictive power like weather, travel distance, and frailty indices while improving on secondary care-focused models. Accurate predictions can guide interventions like reminder systems, transportation assistance, over booking clinics and behavioral nudges.

### Data Collection

Data was collected from 6 general practices within the Brompton Health PCN comprising over 140,000 patients. Practices utilized the SystmOne clinical software. Appointment records over a 4 year period were extracted, covering over 500,000 appointments.

Several steps were required to prepare the raw data for modeling. Patient IDs were excluded, though patients could be re-identified after predictions to suggest interventions. Variables were encoded as binary features as needed. The target Classes were binary: DNA or attended. Appointment status was recorded as Finished or Did Not Attend.

Patients no longer registered with a practice and any deceased patients were excluded from the training data. In total, 37 features were engineered spanning categorical, and numeric variables. A disease and demographic register were created to decrease the need for real time data extraction from SystmOne.

### Feature Engineering

The following categories of features were extracted or derived from the SystmOne data:

- Demographic Features: Patient age, gender, ethnicity
- Appointment Details: Day, time, week of the year, month, day of the week, Rota type, booked by clinician or admin staff
- Temporal Data: Wait time from booking to appointment
- Clinical Features: Disease registers, obesity status, Electronic Frailty Index, History of Hypertension, Diabetes Mellitus, Non-Diabetic Hyper Glycaemia, Significant Mental Illness and Ischaemic Heart Disease
- Access Factors: Distance to surgery, distance to station
- Prior DNA history: DNA rates and counts
- Weather: Temperature, precipitation on appointment day
- Socioeconomic: Index of Multiple Depravation 2023 scores for postcode areas

Including comprehensive data covering clinical, access, and behavioral factors was intended to improve prediction over models focused solely on demographics and appointment details. Feature selection techniques were applied to remove redundant variables. The wide variety of data types posed a modeling challenge.

### Exploratory Data Analysis

Extensive EDA was conducted to understand DNA patterns and distributions across key variables:

- Overall DNA rate was 3.6% for in-person appointments
- DNA rate peaked for ages 20-50 at nearly 6%
- Day of week impacted DNAs, with Friday worst at 4.2%
- Afternoon appointments had higher DNA rates
- DNA rate were lower for patients with a chronic illness.

These initial insights highlighted variables to investigate further in predictive models. The very low baseline DNA rate posed a class imbalance problem that required special handling. The EDA also guided data preprocessing and feature selection.

### Predictive Modeling

We formulated the prediction task as a binary classification problem. Multiple modeling approaches were tested, with a deep neural network found to perform best on this high-dimensional imbalanced dataset.

A feedforward network architecture was developed in TensorFlow. The network had 4 hidden layers, batch normalization, and dropout regularization. Adam optimization and binary cross-entropy loss were used. The model was trained for 120 epochs, with a learning rate of 0.001.

Oversampling minority class examples using SMOTE and aggressive undersampling were applied to mitigate the class imbalance. However, optimal performance was achieved through undersampling the majority class to a ratio of 0.1.

The model was evaluated using stratified 10-fold cross-validation. Performance metrics included AUC, accuracy, precision, recall, and F1 score. Special focus was placed on maximizing recall to accurately detect patients likely to DNA.

### Results

The deep learning model achieved an AUC of 0.91 on the holdout test set. Precision was 80.7% and recall 83% using a probability threshold corresponding to the population DNA rate. The F1 score reached 0.82.

This demonstrates significant predictive power, outperforming persistence models relying solely on past DNA rate. The model identified complex multivariate patterns in the data that drive DNA risk.

Feature importance analysis found prior DNA history to be most impactful, aligning with previous studies. Distance to the surgery was also influential, suggesting transport barriers increase DNA risk. Weather factors like high temperatures were associated with lower DNA rates.

These results confirm the value of including a breadth of data beyond traditional demographics and appointment details. The neural network approach proved capable of learning subtle predictive signals within the multidimensional dataset.

```
epoch/accuracy	0.91787
epoch/auc	0.92304
epoch/cross entropy	0.18157
epoch/epoch	79
epoch/f1_score	0.95584
epoch/learning_rate	0.001
epoch/loss	0.18157
epoch/prc	0.99167
epoch/precision	0.93196
epoch/recall	0.9813
epoch/val_accuracy	0.96282
epoch/val_auc	0.91402
epoch/val_cross entropy	0.11919
epoch/val_f1_score	0.98091
epoch/val_loss	0.11919
epoch/val_prc	0.99643
epoch/val_precision	0.96638
epoch/val_recall	0.99608
```

### Potential Interventions

A key goal of this project was guiding development of interventions to reduce DNAs. Predictions can be used to target reminder communications, transportation assistance, behavioral nudges to high-risk patients and overbooking of clinic with a high predicted DNA rate.

The feature importance insights suggest scheduling high-risk patients on dates with favorable weather could reduce weather-related DNAs. Patients with prior DNAs warrant monitoring.

While further experimental validation is needed, the model provides a robust framework for identifying patients likely to DNA who should receive additional resources and adherence-promoting interventions. Ongoing data collection will allow the model to be retrained to enhance accuracy further. We observed noticeable discrepancies in predictive accuracy across different clinics. These variations are likely attributable to inconsistent naming conventions used for Rota types, as well as potential inaccuracies in recording DNA (Did Not Attend) instances in clinical appointment systems. To address this, we established a baseline accuracy for each clinic using control data. This enables us to adjust the prediction thresholds, allowing us to more effectively identify patients who are likely to not attend their appointments.

### Limitations and Future Directions

This study had several limitations providing opportunities for future work. First, data was limited to a single PCN so model generalizability is uncertain. Expanding to more practices will be beneficial. Second, additional clinical data like diagnoses could improve predictive performance. Third, only basic demographic data was available; including socioeconomic factors like income could strengthen the model.

In terms of technical development, techniques like attention layers could be incorporated to interpret predictions. Federated learning approaches could enable collaborative modeling across sites without sharing raw data. Explicitly modeling cancellation risk is another potential enhancement.

Overall, this project demonstrated the feasibility of an actionable DNA prediction system tailored to the primary care context. Expanded data collection, enhanced features, and ongoing model iteration will further improve performance and clinical value over time. Targeted interventions guided by the model can lead to significant reductions in missed appointments.

### Conclusion

This project illustrated effective application of machine learning to model DNA risk using multifaceted primary care data. A deep neural network model achieved strong predictive performance, outperforming baselines relying solely on past DNA rates. Engineered features provided insights into drivers of DNAs and suggested potential interventions to improve adherence.

Next steps entail implementing the model at PCN level to guide real-world intervention programs. Ongoing data collection will enable regular retraining and performance improvement. This approach shows promise in addressing the widespread challenge of missed appointments to benefit both patients and healthcare providers through more efficient and effective delivery of care.


