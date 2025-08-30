# ytmonetization

Main function to orchestrate the entire project workflow.
    """
    filepath = "youtube_ad_revenue_dataset.csv"
    
    # --- Step 1 & 2: Load and Inspect Data (EDA) ---
    df = load_data(filepath)
    if df is None:
        return
    inspect_data(df)

    # --- Step 3: Preprocessing ---
    df = preprocess_data(df)

    # --- Step 4: Feature Engineering ---
    df = feature_engineer(df)

    # --- Step 5 & 6: Model Building and Evaluation ---
    best_pipeline, results = build_and_evaluate_models(df)

    # --- Step 7: Streamlit App Development ---
    print("\n--- Streamlit App Code Generation ---")
    app_code = create_streamlit_app()
    print("The following code can be used for a Streamlit app. Save it as `app.py` in the same directory.")
    print("\n" + "="*80)
    print(app_code)
    print("="*80)

    # --- Step 8 & 9: Interpretation & Documentation ---
    print("\n--- Interpretation and Insights ---")
    print("Based on the model performance, we can see which model provides the best prediction accuracy. Typically, tree-based models like Random Forest or Gradient Boosting perform well on this type of data.")
    print("The model evaluation metrics (R-squared, MAE, MSE) indicate the model's performance.")
    print("R-squared: proportion of the variance in the dependent variable that is predictable from the independent variables.")
    print("MAE: average of the absolute errors. It's a measure of error in paired observations.")
    print("MSE: average of the squared errors. It gives more weight to larger errors.")



