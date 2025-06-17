## Point 1: Data Analysis

I did the data analysis in Jupyter notebooks. Each notebook starts with an abstraction or summary at the beginning, followed by the implementation details in the remaining sections.

z_model_development\IMDB_tfidf_lr_preprocessing.ipynb  
z_model_development\IMDB_visualization.ipynb  

## Point 2: Model Development  

For TF-IDF+LR, I did an error analysis and summary the cons of this modeling architecture at the end of the notebook.

z_model_development\IMDB_tfidf_lr_model_training.ipynb  
z_model_development\IMDB_DistilBert_training.ipynb  

## Point 3: Model Evaluation  
The evaluation was conducted in two **stages**.  

> In the **first stage**, performed **immediately after training**, I assessed the model’s capability on **validation and test datasets** (or via **cross-validation**) to drive **model selection** and **hyperparameter tuning** for optimal scores.  

> The **second stage** tested the **deployed model** through **containerized endpoints** to simulate the **production environment**, measuring **latency** and **performance metrics**, and evaluating with **recent data** collected from the internet to simulate **real-world conditions**.  


**Stage 1: Right after training**  
z_model_development\IMDB_tfidf_lr_model_training.ipynb  
z_model_development\IMDB_DistilBert_training.ipynb  

**Stage 2: After serving**  
z_model_evaluation\eval.ipynb  

## Point 4: API Serving

The source code under **.\src**
For testing the api, I wrote several examples, check the last section of **api_documentation.md**  

## Point 5: Containerization

**dockerfile**
The instructions are in the last section of README.md  

## Point 6: Documentation

api_documentation.md  
http://localhost:8000/docs  

## Point 7: Monitoring and Upgrade  

z_bonus\solution.md  
z_bonus\Design Diagram.png  

