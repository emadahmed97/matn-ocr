
Here are the steps to follow for your machine learning project:

1. **Problem Framing and Success Metrics** (2:33)
    
    - **Define the problem**: Start with a business problem and determine if ML is the right solution.
    - **User Story**: Identify who the project is for and what problem it solves for them.
    - **ML Metrics**: Include machine learning model metrics like AUC or Mean Squared Error.
    - **Business Metric**: Define the real-world outcome, such as monthly subscriber retention rate or average watch time (3:11).
    - **Constraints**: Consider real-world limitations like latency (how fast predictions are needed), cost (budget for APIs or cloud resources), fairness and bias, and data availability (3:37).
2. **Sourcing Unique Data** (4:41)
    
    - **APIs**: The most realistic approach is to use APIs to request data from external services (5:00).
    - **Web Scraping**: For websites without APIs, use web scraping tools like Beautiful Soup, ensuring it's allowed and not collecting personal data (5:28). The video recommends tools like _ZenRows_ for scalable web scraping (5:57).
    - **Manual Data Collection**: If online data isn't available, collect your own, e.g., photos or fitness tracker data (6:42).
    - **Niche Datasets**: Look into government surveys or industry reports (6:50).
3. **Continuous Data Collection** (6:55)
    
    - **Scheduled Collection**: Set up your code to continuously collect fresh data using cron jobs, workflow tools like _Airflow_ or _Prefect_, or cloud schedulers (7:01).
    - **Data Quality Checks**: Implement early checks for expected fields, reasonable value ranges, unexpected nulls, and data volume. Advanced tools like _Great Expectations_ or _Soda_ can be used (7:21).
4. **Data Storage** (7:45)
    
    - **Structured Data**: Use relational databases like _PostgreSQL_ or _MySQL_ for CSV-style data. Consider designing a data schema (8:01).
    - **Unstructured Data**: For images, video, audio, or large text documents, use object storage like _AWS S3_ (8:20).
    - **In-Memory Store**: For quickly changing data (e.g., session state, recent user actions), use _Redis_ (8:45).
    - **Data Versioning**: Consider using tools like _DVC_ to track which data your model was trained on (9:11).
    - **Cloud Services**: Start learning and using cloud services like _AWS_ and _GCP_ (9:18).
5. **Feature Engineering** (9:44)
    
    - **Cleaning**: Handle missing values, outliers, type conversion, and standardization of your data (10:29).
    - **Creating New Features**: Derive new features from raw data, e.g., from timestamps (day of week, month) or by converting text to numerical representations (10:40).
    - **Pre-processing**: Prepare features for specific models (e.g., all numeric, scaled 0 to 1) (10:56).
    - **Avoid Data Leakage**: Ensure features don't contain information that wouldn't be available at the time of prediction (11:05).
    - **Feature Selection**: Reduce overfitting and redundancy by selecting important features using techniques like feature importance or correlation checks (11:34).
6. **Labeling** (11:55)
    
    - **Labeling Guidelines**: Create clear documentation on how labels are assigned (12:33).
    - **Manual Labeling**: For smaller datasets, label everything yourself using tools like _Label Studio_ (13:02).
    - **Weak Supervision/Programmatic Labeling**: Write rules to automatically generate labels for larger datasets, accepting that they might not be perfect (13:15).
    - **LLM Labeling**: Use large language models (e.g., GPT-5) to generate labels with prompts (13:47).
    - **Validation**: Manually check a sample of automated labels for accuracy (14:00).
    - **Inter-annotator Agreement**: If multiple people label data, measure how often they agree to ensure clear guidelines and well-defined tasks (14:07).
7. **Model Training and Evaluation** (14:47)
    
    - **Proper Data Splits**: Use training, validation (or dev), and a hold-out test set for your data (15:13). For time series, use multiple chronological validation sets (15:37).
    - **Experiment Tracking**: Systematically track experiments using tools like _MLflow_ and _Weights & Biases_ to log hyperparameters, metrics, and model artifacts (15:49).
    - **Model Versioning**: Save trained models with version numbers (e.g., model_v1.pickle) or use a model registry for better organization and rollback capabilities (16:17).
    - **Appropriate Metrics**: Choose metrics that consider class imbalance and include primary and secondary metrics to understand model mistakes (16:39).
    - **Error Analysis**: Evaluate your model on different data segments and systematically analyze the types of errors it makes to identify patterns or areas for improvement (16:50).
8. **Deployment** (17:44)
    
    - **REST API**: Build an API (using _FastAPI_ or _Flask_) that other systems can send requests to for real-time predictions (18:00).
    - **Batch Predictions**: For non-real-time needs, schedule your model to run periodically (e.g., daily) using tools like _Airflow_, saving predictions to a database (18:30).
    - **Interactive App**: Create a simple web app using libraries like _Streamlit_ or _Gradio_ for user interaction (19:09).
    - **Docker**: Include Docker for containerization, bundling your code and dependencies for consistent execution across different environments (19:51).
    - **CI/CD**: Implement continuous integration and continuous deployment using tools like _GitHub Actions_ to automate testing and deployment (20:37).
    - **Tests**: Write unit tests for data preparation and simple integration tests for the full pipeline (21:03).
9. **Monitoring** (21:12)
    
    - **Prediction Logging**: Log every prediction with a timestamp and input features (21:54).
    - **Periodic Metric Computation**: Regularly compute metrics on fresh data to assess accuracy (22:01).
    - **Input Data Checks**: Monitor input data distributions for unexpected values or nulls (22:10).
    - **Feedback Loops**: Set up triggers for re-training when performance drops or new labeled data is accumulated (23:14).

The video emphasizes that you don't need to implement everything at once (2:16). Start with a simple model and iterate, adding complexity as you go. One solid, well-documented project is more valuable than many small ones (23:51). For further learning, the video recommends _Designing Machine Learning Systems_ by Chip Huyen and _Software Engineering for Data Scientists_ by Catherine Nelson (24:04).

![Screenshot 2025-02-05 at 2.04.28 PM.png](app://5402bce5bcd7075d9e8e18080b5907c474ca/Users/emadahmed/Documents/remote-vault-1/Screenshot%202025-02-05%20at%202.04.28%20PM.png?1738785871079)