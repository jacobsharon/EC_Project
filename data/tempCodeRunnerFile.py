    # Replace the missing numeric values with the average
        means_df = pd.read_csv("data/numeric_attribute_averages.csv")
        means_dict = dict(zip(means_df["attribute"], means_df["formatted_mean"]))

        # Ensure numeric conversion for target columns
        for col in means_dict:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Now apply fillna once with all averages
        df.fillna(value=means_dict, inplace=True)

        # Replace the missing categorical values with their respective normal values
        default_cat_df = pd.read_csv("data/categorical_attribute_defaults.csv")
        default_cat_dict = dict(zip(default_cat_df["attribute"], default_cat_df["default_value"]))
        for col, default_value in default_cat_dict.items():
            df[col].fillna(default_value, inplace=True)

        # Replace the missing nominal values with the mode
        default_nom_df = pd.read_csv("data/nominal_attribute_defaults.csv")
        default_nom_dict = dict(zip(default_nom_df["attribute"] , default_nom_df["default_value"]))
        for col, default_value in default_nom_dict.items():
            df[col].fillna(default_value, inplace=True)

        # Output final cleaned dataset
        df.to_csv("data/cleaned_ckd_dataset.csv", index=False)