# Circular Vision

**Sistema inteligente de inspeção visual para apoio à economia circular em operações industriais.**

Circular Vision is a proof of concept that applies computer vision, machine learning and data analytics to classify material and waste streams from images. The solution supports circular economy decisions in manufacturing environments by recommending actions such as recycling, composting, reverse logistics, special disposal or operational review.

The project was developed as an Industry 4.0 solution focused on sustainability, intelligent resource management and operational decision support.

---

## 1. Challenge Context

This project addresses the following innovation challenge:

> Propose an innovative solution that uses Industry 4.0 technologies to promote sustainability in manufacturing operations, addressing aspects such as circular economy, emissions reduction or intelligent resource management.

Circular Vision proposes a practical and reproducible solution for manufacturing environments where packaging materials, recyclable waste and special disposal items need to be visually identified and correctly routed.

In industrial operations, incorrect sorting of waste and packaging materials may reduce recycling efficiency, increase the amount of material sent to landfill and make it harder to generate reliable sustainability indicators. Circular Vision uses artificial intelligence to support this decision-making process.

---

## 2. Proposed Solution

Circular Vision transforms images from inspection points into operational circular economy decisions.

The solution follows this pipeline:

```text
Image acquisition
        ↓
Visual feature extraction
        ↓
Machine learning classification
        ↓
Circular economy decision layer
        ↓
Operational dashboard
```

The system classifies an image into a material or waste category and then maps the prediction to a recommended sustainability action.

Examples:

```text
cardboard → send to cardboard and paper recycling stream
battery   → send to certified reverse logistics or hazardous waste handling
biological → send to composting, biodigestion or controlled organic treatment
trash → send to controlled disposal and review source reduction opportunities
```

---

## 3. Product Vision

Circular Vision was designed as a lightweight proof of concept that could evolve into an industrial visual inspection product.

In a real manufacturing environment, the solution could be connected to:

* industrial cameras;
* edge devices;
* inspection stations;
* packaging lines;
* waste sorting areas;
* internal logistics points;
* sustainability dashboards;
* quality and manufacturing systems.

The goal is not only to classify images, but to convert visual data into useful operational indicators for sustainability, circular economy and continuous improvement.

---

## 4. Alignment with Industry 4.0

The project applies Industry 4.0 concepts through the integration of image-based data, artificial intelligence and operational analytics.

| Industry 4.0 element             | Application in Circular Vision                                            |
| -------------------------------- | ------------------------------------------------------------------------- |
| Computer vision                  | Image-based recognition of material and waste streams                     |
| Artificial intelligence          | Automated classification using machine learning                           |
| Data analytics                   | Circularity indicators, landfill avoidance estimate and priority analysis |
| Smart decision support           | Recommendation of sustainable actions based on predictions                |
| Industrial integration potential | Adaptable to cameras, sensors, edge devices and manufacturing dashboards  |
| Sustainability monitoring        | Operational report for circular economy decision-making                   |

---

## 5. Alignment with the Position Scope

The project is aligned with the expected activities and technical requirements of the position.

| Requirement             | How Circular Vision addresses it                                               |
| ----------------------- | ------------------------------------------------------------------------------ |
| Python programming      | Complete pipeline implemented in Python                                        |
| Computer vision         | Image preprocessing and visual feature extraction                              |
| Artificial intelligence | Supervised machine learning models for classification                          |
| Data analysis           | Metrics, CSV reports, indicators and dashboard                                 |
| Statistics              | Accuracy, macro F1-score, confusion matrix and class distribution              |
| Industrial technology   | Designed for visual inspection points in manufacturing environments            |
| Sensors and cameras     | Conceptual architecture supports integration with industrial cameras           |
| Sustainability          | Circular economy decisions, recycling, special disposal and landfill reduction |

---

## 6. Dataset

The proof of concept uses the public dataset:

**Garbage Classification - Kaggle**
Slug: `mostafaabla/garbage-classification`

The dataset contains images distributed across material and waste classes.

Expected classes:

```text
battery
biological
brown-glass
cardboard
clothes
green-glass
metal
paper
plastic
shoes
trash
white-glass
```

The dataset is not included in this repository. Downloading it requires internet access and Kaggle/KaggleHub access.

---

## 7. Project Structure

```text
circular-vision-manufacturing/
├── app/
│   ├── config.py
│   ├── dashboard.py
│   ├── data_loader.py
│   ├── feature_extraction.py
│   └── sustainability_score.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── outputs/
├── scripts/
│   ├── download_dataset.py
│   └── find_dataset_root.py
├── tests/
├── train.py
├── predict.py
├── batch_inference.py
├── requirements.txt
└── README.md
```

---

## 8. Setup on Windows

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Download the Dataset

### Option A - Automatic download with KaggleHub

```bash
python scripts/download_dataset.py
python scripts/find_dataset_root.py
```

Copy the path printed by `find_dataset_root.py`. This path will be used for training and batch inference.

Example:

```text
C:\Users\INSTRUTOR TI\Documents\nestle\data\raw\garbage-classification\garbage_classification
```

### Option B - Manual download from Kaggle

1. Download the dataset from Kaggle.
2. Extract it into:

```text
data/raw/garbage-classification/
```

3. Make sure the dataset contains folders named by class, such as:

```text
cardboard
plastic
metal
paper
trash
```

4. Run:

```bash
python scripts/find_dataset_root.py
```

---

## 10. Train the Model

For a fast proof of concept:

```bash
python train.py --dataset-dir "C:\Users\INSTRUTOR TI\Documents\nestle\data\raw\garbage-classification\garbage_classification" --max-per-class 50
```

For a more robust training run:

```bash
python train.py --dataset-dir "C:\Users\INSTRUTOR TI\Documents\nestle\data\raw\garbage-classification\garbage_classification" --max-per-class 150
```

Training generates:

```text
models/circular_vision_model.joblib
outputs/metrics.json
outputs/confusion_matrix.png
outputs/class_distribution.csv
```

The model selection process compares a baseline model and a main model. The best model is selected based on macro F1-score.

---

## 11. Run Batch Inference

Batch inference processes multiple images and generates the operational sustainability report used by the dashboard.

For a quick demonstration:

```bash
python batch_inference.py --dataset-dir "C:\Users\INSTRUTOR TI\Documents\nestle\data\raw\garbage-classification\garbage_classification" --max-per-class 5
```

For a larger report:

```bash
python batch_inference.py --dataset-dir "C:\Users\INSTRUTOR TI\Documents\nestle\data\raw\garbage-classification\garbage_classification" --max-per-class 10
```

This generates:

```text
outputs/sustainability_report.csv
```

The report includes:

* image path;
* source class;
* predicted class;
* prediction confidence;
* material stream;
* recommended circular economy action;
* priority level;
* circularity score;
* estimated landfill avoidance;
* special handling alerts.

---

## 12. Predict a Single Image

```bash
python predict.py --image path/to/image.jpg
```

Example output:

```json
{
  "predicted_class": "cardboard",
  "confidence": 0.87,
  "circular_decision": {
    "material_stream": "recyclable packaging stream",
    "recommended_action": "send to cardboard and paper recycling stream",
    "priority": "low",
    "circularity_score": 90,
    "estimated_landfill_avoidance_kg": 0.12,
    "special_handling": false
  }
}
```

---

## 13. Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard provides:

* executive overview;
* inspected items;
* average circularity score;
* estimated landfill avoidance;
* special handling alerts;
* predicted material stream distribution;
* priority distribution;
* operational report table;
* sample inspection gallery;
* manual image inspection;
* class probability visualization.

If `outputs/sustainability_report.csv` exists, the dashboard automatically loads the latest operational report.

---

## 14. Machine Learning Approach

The proof of concept uses interpretable visual features extracted from images.

The feature extraction pipeline includes:

* HSV color histograms;
* texture descriptors based on GLCM;
* edge density based on Canny;
* HOG descriptors.

The project compares:

* KNN as a baseline model;
* Random Forest as the main model.

The best model is selected according to macro F1-score.

This approach was selected because it is lightweight, reproducible and suitable for a proof of concept that can run on a standard Windows environment without GPU dependency.

---

## 15. Circular Economy Decision Layer

The machine learning model predicts the material class. The sustainability decision layer translates the prediction into an operational action.

| Predicted class | Material stream                | Recommended action                                                     |
| --------------- | ------------------------------ | ---------------------------------------------------------------------- |
| battery         | special hazardous disposal     | send to certified reverse logistics or hazardous waste handling        |
| biological      | organic waste stream           | send to composting, biodigestion or controlled organic waste treatment |
| brown-glass     | glass recycling stream         | send to glass recycling stream after contamination check               |
| green-glass     | glass recycling stream         | send to glass recycling stream after contamination check               |
| white-glass     | glass recycling stream         | send to glass recycling stream after contamination check               |
| cardboard       | recyclable packaging stream    | send to cardboard and paper recycling stream                           |
| paper           | recyclable paper stream        | send to paper recycling stream                                         |
| plastic         | plastic recycling stream       | send to plastic recycling stream after resin and contamination check   |
| metal           | metal recycling stream         | send to metal recycling stream                                         |
| clothes         | textile recovery stream        | send to textile reuse, recovery or recycling flow                      |
| shoes           | mixed material recovery stream | send to reuse, material recovery or specialized recycling flow         |
| trash           | non-recyclable waste stream    | send to controlled disposal and review source reduction opportunities  |

Each decision also includes:

* priority;
* circularity score;
* estimated landfill avoidance;
* special handling flag.

---

## 16. Dashboard Indicators

The dashboard summarizes the latest batch inspection report.

Main indicators:

| Indicator                  | Meaning                                             |
| -------------------------- | --------------------------------------------------- |
| Inspected items            | Number of images processed                          |
| Average circularity score  | Average sustainability score of the inspected items |
| Estimated avoided landfill | Estimated mass that could be diverted from landfill |
| Special handling alerts    | Number of items requiring special attention         |

These indicators are intended to support sustainability, manufacturing, quality and continuous improvement teams.

---

## 17. Example Operational Scenario

A manufacturing plant has multiple waste and packaging material streams generated during production and internal logistics.

Circular Vision could be deployed near a sorting station or packaging line. A camera captures images of material streams. The system classifies each item and recommends the correct destination. The dashboard consolidates the results and provides operational indicators.

Example flow:

```text
Camera near sorting point
        ↓
Image captured
        ↓
Circular Vision classification
        ↓
Recommended action generated
        ↓
Dashboard updated
        ↓
Operator or sustainability team acts on the information
```

---

## 18. Expected Benefits

Circular Vision can support:

* improved visual sorting of packaging and waste streams;
* reduction of incorrect disposal;
* better recycling quality;
* identification of special handling cases;
* operational visibility for circular economy initiatives;
* data-driven sustainability monitoring;
* support for continuous improvement in manufacturing operations.

---

## 19. Limitations

This project is a proof of concept and uses a public dataset as a proxy for industrial material and waste streams.

For production use, the following steps would be required:

* collection of real images from the manufacturing environment;
* validation with local operational data;
* model retraining with plant-specific images;
* integration with industrial cameras or edge devices;
* definition of internal sustainability rules;
* validation with quality, safety and environmental teams;
* monitoring of model performance over time.

---

## 20. Future Work

Future versions may include:

* real-time camera integration;
* edge deployment;
* object detection models;
* deep learning models;
* integration with MES, ERP or sustainability systems;
* historical dashboards by line, shift, area or product family;
* carbon impact estimation;
* anomaly detection for incorrect disposal patterns;
* feedback loop for continuous model improvement.

---

## 21. Suggested 3-Minute Presentation Narrative

Circular Vision is an Industry 4.0 proof of concept for sustainable manufacturing. It uses computer vision and machine learning to classify material and waste streams from images and converts each prediction into an operational circular economy recommendation.

The solution can support manufacturing plants by improving visual sorting, reducing incorrect disposal, identifying special handling cases and generating circularity indicators for decision-making.

The proof of concept was built in Python and includes image processing, machine learning, batch inference and a Streamlit dashboard. It is lightweight, reproducible and adaptable to inspection points, packaging lines, waste sorting stations or internal logistics areas.

The main value of Circular Vision is not only image classification, but the transformation of visual inspection into sustainability-oriented operational decisions.

---

## 22. Quick Execution Summary

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/find_dataset_root.py
python train.py --dataset-dir "<dataset_root>" --max-per-class 50
python batch_inference.py --dataset-dir "<dataset_root>" --max-per-class 10
streamlit run app/dashboard.py
```

---

## 23. Repository Purpose

This repository was created as a reproducible proof of concept for an innovation challenge focused on sustainability in manufacturing operations.

The project demonstrates how computer vision, artificial intelligence and data analytics can be combined to support circular economy decisions in an industrial context.
