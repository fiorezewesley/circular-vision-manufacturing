# Technical notes

## Objective

Build a reproducible proof of concept for image-based material classification and circular economy decision support in manufacturing operations.

## Feature extraction

The project uses handcrafted features to avoid GPU dependency and keep the PoC reproducible in one day.

Feature groups:

1. HSV histogram for color distribution.
2. GLCM texture features for visual material texture.
3. Canny edge density for structural information.
4. HOG descriptors for shape and local gradient information.

## Model selection

A KNN model is used as baseline. A Random Forest model is used as the main model because it handles nonlinear relationships and mixed visual descriptors well without extensive tuning.

The selected model is the one with the highest macro F1-score.

## Industry 4.0 interpretation

In a real deployment, images could come from fixed cameras placed near:

- waste sorting stations;
- packaging inspection points;
- internal logistics areas;
- recycling collection points;
- production line discard points.

The model output can be integrated with MES, ERP, sustainability reporting tools or a lightweight edge device.

## Limitations

This PoC does not replace an industrial-grade vision system. It demonstrates the feasibility of the concept using an open image dataset. Industrial deployment would require local image collection, lighting control, camera calibration, validation with plant-specific materials and integration with operational systems.
