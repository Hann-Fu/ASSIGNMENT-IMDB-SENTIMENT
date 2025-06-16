# Sentiment Analysis API Documentation

## Endpoints

### 1. Baseline Sentiment Prediction
**POST** `/predict/baseline`

Predicts sentiment using a baseline model.

**Request Body:**
```json
{
  "text": "string",
  "version": "1.0.0"
}
```

**Response:**
```json
{
  "positive_probability": 0.75,
  "negative_probability": 0.25,
  "sentiment": "positive"
}
```

### 2. BERT Sentiment Prediction
**POST** `/predict/bert`

Predicts sentiment using a DistilBERT model.

**Request Body:**
```json
{
  "text": "string",
  "version": "1.0.0"
}
```

**Response:**
```json
{
  "positive_probability": 0.85,
  "negative_probability": 0.15,
  "sentiment": "positive"
}
```

### 3. LLM Sentiment Prediction
**POST** `/predict/llm`

Predicts sentiment using a Large Language Model.

**Request Body:**
```json
{
  "text": "string",
  "version": "1.0.0"
}
```

**Response:**
```json
{
  "positive_probability": 1,
  "negative_probability": 0,
  "sentiment": "positive"
}
```

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input or unsupported version |
| 429 | Too Many Requests - Rate limit exceeded (LLM endpoint only) |
| 500 | Internal Server Error - General server error |
| 503 | Service Unavailable - Prediction service temporarily unavailable |
| 507 | Insufficient Storage - Not enough memory to process request |

## Notes

- All endpoints currently support version `"1.0.0"` only
- The LLM endpoint returns binary probabilities (0 or 1)
- Text input should be provided as a string in the request body


# Example test requests:

1. **Baseline model** 
```bash
curl --location 'http://0.0.0.0:8000/api/v1/predict/baseline' \
--header 'Content-Type: application/json' \
--data '{
    "text":"A good movie",
    "version":"1.0.0"

}'
```

2. **Bert** 
```bash
curl --location 'http://0.0.0.0:8000/api/v1/predict/bert' \
--header 'Content-Type: application/json' \
--data '{
    "text":"A good movie",
    "version":"1.0.0"

}'
```

3. **LLM** 
```bash
curl --location 'http://0.0.0.0:8000/api/v1/predict/llm' \
--header 'Content-Type: application/json' \
--data '{
    "text":"A good movie",
    "version":"1.0.0"

}'
```
