## Answer of Bonus Task

**Solution design diagram:** 
Please refer System Architecture.png under current folder.

**Data drift detection implementation**
Please refer "data_drift_implementation/data_drift_embedding_approach.ipynb"

### Degradation-Detection & Auto-Retrain Flow

1. **Online Prediction**

   * Client sends **text** → **Predict Service** (EC2).
   * Service call a **Transformers v1 model** to get **embedding + prediction**.
   * All raw text, timestamp, prediction, and embedding are saved by **Data Persist Service** into **Postgres + Milvus**.

2. **Label Feedback**

   * Human expert can add / update **label** later.
   * Those labels are written back to the same tables, so we can compare real vs predicted next time.

3. **Degradation Detect Service** (runs by cron on **AWS Fargate**)

   * Pull **last X days** data with a **scrolling window**.
   * **Data drift check** – cosine-similarity of new vs old embeddings. 
   * **Performance check** – compute F1 (or other main metric) on labeled slice.
   * If **drift > 0.85** but the **F1 drop < 5%**, it just writes a **log alert**.
   * If **F1 drop > 5 %** (all numbers in a small **threshold config** YAML), it fires a **retrain flag**. If not, just writes a **log alert**.

4. **Auto Retrain & Redeploy**

   * Separate EC2 host runs **Retrain API** (it hides whole notebook pipeline).
   * When flag arrives, it retrains on full data, pushes **Transformers v2** artifact to **S3**.
   * Predict Service watches S3 and hot-loads v2 (shadow test first, then full swap).
   * If flag is **false**, only the log alert is kept (no restart).


