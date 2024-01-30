# CandlestickPatternClassifier

### Overview
This repository contains a Python project for classifying candlestick patterns in financial data. The project aims to identify specific trading patterns in stock market data using image classification techniques and predict market trends.

### Features
Downloads financial data for specified stock tickers.
Generates candlestick chart images from historical data.
Classifies images based on defined candlestick patterns.
Utilizes logistic regression for pattern recognition.
Offers insights into potential bullish market trends.

### Installation

### Clone the repository:
git clone https://github.com/ashkanpir/CandlestickPatternClassifier

### Navigate to the project directory:
cd CandlestickPatternClassifier

### Install the requirements:
pip install -r requirements.txt

### Usage
To run the project, execute the main Python script:
python main.py

### Structure
main.py: The main script which orchestrates the data download, image generation, and classification tasks.

## Pattern Recognition Logic

The project focuses on identifying a specific bullish pattern in candlestick charts. The pattern is defined by:
Consecutive bars where the closing price is higher than the opening price.
Ascending closing prices over a sequence of bars.
Increasing highs and lows, indicate a rising market.
Smaller shadows compared to the body size of the candlesticks, suggest strong market sentiment.

## Model Training and Evaluation
Logistic Regression is used for classifying images into target (pattern matches) and non-target groups.
The model is evaluated based on its accuracy, and further improvements are planned through hyperparameter tuning and exploring other algorithms.



