import os
import matplotlib.pyplot as plt
import pandas as pd

# Import your four refactored modules
import Module_one as m1
import Module_two as m2
import Module_three as m3
import Module_four as m4

def main():
    df = None
    print("==================================================")
    print("      Welcome to DataForge CLI Studio             ")
    print("==================================================")

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Load Dataset (Module 1)")
        print("2. Inspect & Profile Data (Module 1)")
        print("3. Clean & Preprocess Data (Module 2)")
        print("4. Generate Visualizations (Module 3)")
        print("5. Split Dataset into Train/Test/Val (Module 4)")
        print("6. Export Cleaned CSV")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()

        # ==================== 1. LOAD DATASET ====================
        if choice == "1":
            path = input("Enter path to your dataset (.csv, .xlsx, .json): ").strip()
            try:
                df = m1.GetDataSet(path)
                print(f"Success! Loaded dataset with shape: {df.shape}")
            except Exception as e:
                print(f"Error loading dataset: {e}")

        # ==================== 2. INSPECT DATA ====================
        elif choice == "2":
            if df is None:
                print(" Please load a dataset first!")
                continue

            datainfo, highlight, col_list, dup_count = m1.PrintingInformation(df)
            
            print("\n--- COLUMN SUMMARY ---")
            for col, info in datainfo.items():
                print(f"\n[{col}]")
                for k, v in info.items():
                    print(f"  - {k}: {v}")
            
            print(f"\nTotal Duplicate Rows: {dup_count}")

            save_report = input("\nGenerate HTML report and missing values plot? (y/n): ").strip().lower()
            if save_report == 'y':
                html_path, plot_path = m1.SaveGraphAndTable(highlight, df)
                print(f" Saved HTML report to: {html_path}")
                print(f" Saved missing values plot to: {plot_path}")

        # ==================== 3. DATA CLEANING ====================
        elif choice == "3":
            if df is None:
                print(" Please load a dataset first!")
                continue

            print("\n--- CLEANING MENU ---")
            print("a. Fill Missing Values (NaN)")
            print("b. Drop Duplicate Rows")
            print("c. Rename Column")
            print("d. Swap Column Order")
            print("e. Encode Categorical Feature")
            
            clean_choice = input("Select cleaning action (a-e): ").strip().lower()

            if clean_choice == "a":
                col = input(f"Enter column name {list(df.columns)}: ").strip()
                method = input("Choose method (mean / median / mode / constant / ai): ").strip().lower()
                custom_val = "Missing"
                if method == "constant":
                    custom_val = input("Enter fill value: ").strip()
                
                df = m2.FillingNaN(df, column=col, method=method, custom_fill_value=custom_val)
                print(f" Applied '{method}' imputation on column '{col}'. Remaining nulls: {df[col].isnull().sum()}")

            elif clean_choice == "b":
                df, removed_count = m2.DeleteDuplicate(df)
                print(f" Removed {removed_count} duplicate row(s).")

            elif clean_choice == "c":
                old_name = input("Current column name: ").strip()
                new_name = input("New column name: ").strip()
                df = m2.ColumnRearrangeRenamer(df, action="rename", old_col=old_name, new_col=new_name)
                print(f" Renamed '{old_name}' to '{new_name}'.")

            elif clean_choice == "d":
                c1 = input("First column to swap: ").strip()
                c2 = input("Second column to swap: ").strip()
                df = m2.ColumnRearrangeRenamer(df, action="rearrange", col1=c1, col2=c2)
                print(f" Swapped columns '{c1}' and '{c2}'.")

            elif clean_choice == "e":
                col = input(f"Select categorical column {list(df.select_dtypes(include=['object', 'category']).columns)}: ").strip()
                method = input("Choose encoder (one-hot / ordinal / target / native): ").strip().lower()
                target_c = None
                if method == "target":
                    target_c = input("Enter target column (y): ").strip()
                
                df = m2.EnCoder(df, column=col, method=method, target_col=target_c)
                print(f" Applied {method} encoding to '{col}'.")

        # ==================== 4. VISUALIZATION ====================
        elif choice == "4":
            if df is None:
                print(" Please load a dataset first!")
                continue

            print("\n--- CHART MENU ---")
            print("a. Histogram")
            print("b. Boxplot")
            print("c. Scatter Plot")
            print("d. Correlation Heatmap")
            print("e. Line Chart")

            chart_choice = input("Select chart type (a-e): ").strip().lower()
            fig = None

            if chart_choice == "a":
                x = input("Enter X-axis column: ").strip()
                hue = input("Enter Hue/Group column (optional, press Enter to skip): ").strip() or None
                fig = m3.generate_histogram(df, x_col=x, hue_col=hue)

            elif chart_choice == "b":
                y = input("Enter Y-axis (numeric) column: ").strip()
                x = input("Enter X-axis (categorical) column (optional, press Enter to skip): ").strip() or None
                fig = m3.generate_boxplot(df, y_col=y, x_col=x)

            elif chart_choice == "c":
                x = input("Enter X-axis column: ").strip()
                y = input("Enter Y-axis column: ").strip()
                hue = input("Enter Hue column (optional, press Enter to skip): ").strip() or None
                fig = m3.generate_scatterplot(df, x_col=x, y_col=y, hue_col=hue)

            elif chart_choice == "d":
                fig = m3.generate_heatmap(df)

            elif chart_choice == "e":
                x = input("Enter X-axis column: ").strip()
                y = input("Enter Y-axis column: ").strip()
                hue = input("Enter Hue column (optional, press Enter to skip): ").strip() or None
                fig = m3.generate_linechart(df, x_col=x, y_col=y, hue_col=hue)

            if fig:
                plt.show() # Display chart interactively in window

        # ==================== 5. DATASET SPLITTING ====================
        elif choice == "5":
            if df is None:
                print(" Please load a dataset first!")
                continue

            test_s = float(input("Enter Test Set ratio (e.g. 0.2 for 20%): ") or 0.2)
            val_s = float(input("Enter Validation Set ratio (e.g. 0.1 for 10%): ") or 0.1)

            train, test, val = m4.DatasetSplitter(df, test_size=test_s, validation_size=val_s)
            
            print(f"\n Data Split Complete:")
            print(f"  - Training Set:   {train.shape[0]} rows ({train.shape[0]/len(df)*100:.1f}%)")
            print(f"  - Test Set:       {test.shape[0]} rows ({test.shape[0]/len(df)*100:.1f}%)")
            print(f"  - Validation Set: {val.shape[0]} rows ({val.shape[0]/len(df)*100:.1f}%)")

            save_splits = input("\nSave split datasets as CSV files? (y/n): ").strip().lower()
            if save_splits == 'y':
                train.to_csv("train.csv", index=False)
                test.to_csv("test.csv", index=False)
                val.to_csv("val.csv", index=False)
                print(" Saved 'train.csv', 'test.csv', and 'val.csv' to current folder.")

        # ==================== 6. EXPORT CLEANED DATA ====================
        elif choice == "6":
            if df is None:
                print(" Please load a dataset first!")
                continue
            
            output_filename = input("Enter filename to save (e.g., cleaned_data.csv): ").strip() or "cleaned_data.csv"
            df.to_csv(output_filename, index=False)
            print(f" Dataset successfully saved as '{output_filename}'!")

        # ==================== 7. EXIT ====================
        elif choice == "7":
            print("Exiting DataForge CLI Studio. Goodbye!")
            break

if __name__ == "__main__":
    main()