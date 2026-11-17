# Synx – Football Possession Prediction

## 1. Project Overview

Synx is a football possession prediction system that identifies which team has possession of the ball at every frame of a match.

It processes player and ball tracking data, analyses movement and positioning, estimates possession, and detects dead-ball situations.

Each frame is assigned one of four labels:

- **H** – Home team possession
- **A** – Away team possession
- **D** – Dead-ball situation
- **X** – Unknown or uncertain possession

## 2. System Architecture

The system follows a sequential processing pipeline:

```text
Tracking Data
     |
     v
Data Preprocessing
     |
     v
Feature Extraction
     |
     v
Possession Estimation
     |
     v
Dead-Ball Detection
     |
     v
Possession State Management
     |
     v
CSV Output
