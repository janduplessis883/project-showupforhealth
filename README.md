<img src="https://github.com/janduplessis883/project-showupforhealth/blob/master/images/Show%20Up%20for%20Health.png?raw=true" width=750>

The United Kingdom's National Health Service (NHS) is a behemoth in the healthcare sector, providing primary, secondary, and tertiary care to millions of residents. With this vast reach and immense responsibility, efficient resource utilization is crucial. One aspect of resource allocation that has attracted attention over the years is missed appointments, colloquially known as "Did Not Attend" (DNA) events. These DNAs are reported to cost the NHS approximately £1 billion annually, a substantial figure that has the potential to be reduced with the right interventions.

Situated in West London, Brompton Health Primary Care Network (PCN) is an amalgamation of 12 individual NHS GP Practices. Serving a considerable patient population of 140,000 individuals, this PCN experiences a DNA rate of 4% for face-to-face (F2F) appointments. Notably, while these figures pertain to F2F interactions, telephone appointments are typically not coded as missed, which might mean the true cost and extent of DNAs is potentially underrepresented.

Despite the significance of DNAs in primary care, much of the machine learning research in predicting missed appointments has focused on secondary care data. This presents an opportunity: can we harness the power of data science and machine learning to better predict, and thus mitigate, DNAs in a primary care setting like Brompton Health PCN?

This project aims to bridge this gap. Leveraging patient data, we intend to develop a predictive model to identify patients at risk of missing their appointments. By doing so, we hope to provide healthcare practitioners with a tool that allows for timely interventions, potentially reducing the number of DNAs and the associated costs.

By hosting this project on GitHub, we invite collaboration, scrutiny, and iterative improvement. Together, we can work towards a more efficient NHS and better patient care.

## Objective:
To develop a predictive model utilizing deep learning techniques that can accurately forecast the likelihood of patients missing their primary care appointments. By considering unique features, including the Electronic Frailty Index, Obesity, Depression, DM, IMD 2023, distance from the surgery, the booking method of the consultation, length of registration with the practice, weather conditions (temperature and precipitation), and historical no-show data, this project aims to aid healthcare providers in making informed decisions and optimizing patient management.

## Features Overview:
**Electronic Frailty Index**: Evaluating the vulnerability of a patient, often used in primary care to predict adverse outcomes.<BR>
**Obesity**: Patient's obesity status, a common chronic condition with potential health complications.<BR>
**Depression**: Mental health status, which can impact appointment adherence.<BR>
**IMD 2023**: Index of Multiple Deprivation 2023 score, giving insight into the patient's socioeconomic status.<BR>
**Distance from Surgery**: How far the patient resides from the healthcare center, impacting ease of access.<BR>
**Booking Method**: Whether the appointment was clinician-booked or patient-booked, potentially influencing commitment levels.<BR>
**Length of Registration with Practice**: How long the patient has been associated with the healthcare center, indicating familiarity and potential loyalty.<BR>
**Weather Conditions**:<BR>
**Temperature**: To determine if extreme temperatures impact appointment attendance.<BR>
**Precipitation**: Analyzing the impact of rain or snow on appointment adherence.<BR>
**Previous No-shows**: Historical data of missed appointments by the patient.<BR>

## Evaluation Metrics:
**Recall**: Emphasizing the importance of correctly identifying actual no-shows, ensuring resources and proactive measures can be taken to possibly improve attendance.<BR>
**Area Under the Curve (AUC)**: To evaluate the overall capability of the model to discriminate between those who will and will not miss their appointment.<BR>

## Implementation Plan:
**Data Collection and Cleaning**: Ensure data quality and standardize all feature inputs.<BR>
**Exploratory Data Analysis**: Understand the distribution and relationship between variables.<BR>
**Feature Engineering and Selection**: Enhance model performance by creating or selecting the most impactful variables.<BR>
**Model Development**: Implement a deep learning framework suitable for the dataset's nature and size.<BR>
**Evaluation**: Use a test dataset to assess model performance based on recall and AUC.<BR>
**Iterative Improvements**: Refine model based on feedback loops and performance on unseen data.<BR>
**Deployment**: Integrate the model within healthcare systems for real-time or batch predictions.


