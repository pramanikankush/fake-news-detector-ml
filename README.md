# Fake News Detection System

A machine learning-based solution to detect fake news by analyzing news article content from APIs.

## Overview

This project implements a fake news detection system that:
1. Fetches real-time news data from a news API
2. Preprocesses text data using NLP techniques
3. Uses machine learning to classify news as real or fake
4. Provides clear probability scores for classification decisions

## Features

- **Real-time News API Integration**: Fetch the latest news from various sources
- **Advanced Text Preprocessing**: Clean and normalize text for improved analysis
- **TF-IDF Vectorization**: Convert text to numerical features for machine learning
- **Logistic Regression Model**: Efficient classification with interpretable results
- **Feature Importance Analysis**: Identify linguistic indicators of fake news
- **Interactive Jupyter Notebook**: Easy to understand and modify the implementation

## Requirements

- Python 3.7+
- pandas
- numpy
- requests
- nltk
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the NLTK resources (if not done in the notebook):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. Obtain a News API key from [News API](https://newsapi.org/) (free tier available)

2. Open the Jupyter notebook:
```bash
jupyter notebook fake_news_detection.ipynb
```

3. Insert your API key in the designated area of the notebook:
```python
api_key = "YOUR_NEWS_API_KEY"  # Replace with your actual API key
```

4. Run the notebook cells sequentially to:
   - Fetch news articles
   - Preprocess the data
   - Train the fake news detection model
   - Evaluate performance
   - Classify real-time news

5. Use the `predict_news_authenticity()` function to classify any news text:
```python
news_text = "Your news article text here..."
prediction, probability = predict_news_authenticity(news_text, model, tfidf_vectorizer)
```

## Datasets

For training, the system uses:
- Sample fake and real news for demonstration
- In a production environment, you can use these recommended datasets:
  - [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
  - [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

## Model Performance

The current implementation achieves:
- Strong classification accuracy on test data
- Feature importance analysis to explain decisions
- Clearly visualized confusion matrix and classification metrics

## Extending the Project

Ideas for enhancement:
- Implement more advanced models (BERT, XGBoost)
- Add sentiment analysis to detect emotional manipulation
- Create a web application interface
- Implement cross-source fact-checking
- Add source credibility scoring
- Train on a larger and more diverse dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [News API](https://newsapi.org/) for providing access to real-time news data
- NLTK for natural language processing tools
- scikit-learn for machine learning implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
