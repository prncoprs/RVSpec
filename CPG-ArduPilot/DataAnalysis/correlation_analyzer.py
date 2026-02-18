import pandas as pd
import numpy as np

def filter_and_sort_phase2_correlations(csv_file):
    """
    Filter rows containing 'Phase2' and sort by correlation values.
    
    Args:
        csv_file (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Filtered and sorted DataFrame
    """
    
    # Read the CSV file - handle multiple delimiters and empty columns
    df = pd.read_csv(csv_file)
    
    # Debug: Print data structure
    print(f"DataFrame shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"First few column names: {list(df.columns[:10])}")
    
    # Filter rows that contain 'Phase2' in the first column (index column)
    phase2_rows = df[df.iloc[:, 0].str.contains('Phase2', na=False)]
    
    if phase2_rows.empty:
        print("No Phase2 rows found in the data.")
        return pd.DataFrame(), []
    
    print(f"Found {len(phase2_rows)} Phase2 rows")
    
    # Get the Phase2 row(s)
    phase2_data = phase2_rows.copy()
    
    # For each Phase2 row, create a summary of correlations
    results = []
    
    for idx, row in phase2_data.iterrows():
        # Get all numeric correlation values (skip the first column which is the row identifier)
        correlations = {}
        
        print(f"\nAnalyzing row: {row.iloc[0]}")
        print("Checking each column for valid correlations:")
        
        for col_idx, col_name in enumerate(df.columns[1:], 1):  # Skip first column
            value = row.iloc[col_idx]
            
            # Debug: Show what we're finding
            if col_idx <= 10:  # Show first 10 for debugging
                print(f"  Column {col_idx} '{col_name}': value = '{value}', type = {type(value)}")
            
            # Only include numeric values (not NaN or empty)
            if pd.notna(value) and str(value).strip() != '':
                try:
                    numeric_value = float(value)
                    correlations[col_name] = numeric_value
                    if col_idx <= 10:
                        print(f"    -> Added: {col_name} = {numeric_value}")
                except (ValueError, TypeError):
                    if col_idx <= 10:
                        print(f"    -> Skipped (not numeric): {value}")
        
        print(f"\nTotal correlations found: {len(correlations)}")
        if len(correlations) > 0:
            print("All correlations found:")
            for factor, corr in correlations.items():
                print(f"  {factor}: {corr}")
        
        # Sort correlations by absolute value (highest correlation first)
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: abs(x[1]), 
                                   reverse=True)
        
        # Create result dictionary
        result_row = {'Phase': row.iloc[0]}
        for i, (factor, correlation) in enumerate(sorted_correlations):
            result_row[f'Factor_{i+1}'] = factor
            result_row[f'Correlation_{i+1}'] = correlation
        
        results.append(result_row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df, sorted_correlations

def display_top_correlations(sorted_correlations, top_n=10):
    """
    Display the top N correlations in a readable format.
    
    Args:
        sorted_correlations (list): List of tuples (factor, correlation)
        top_n (int): Number of top correlations to display
    """
    print(f"\nTop {top_n} factors with highest correlations (by absolute value):")
    print("-" * 60)
    
    for i, (factor, correlation) in enumerate(sorted_correlations[:top_n], 1):
        print(f"{i:2d}. {factor:<25} | Correlation: {correlation:8.6f}")

# Main execution
if __name__ == "__main__":
    # Replace 'Top_Correlated_Factors.csv' with your actual file path
    csv_filename = './analysis_output_sitl/Top_Correlated_Factors.csv'
    
    try:
        # Process the data
        result_df, sorted_correlations = filter_and_sort_phase2_correlations(csv_filename)
        
        if not result_df.empty:
            print("Phase2 data filtered and sorted successfully!")
            
            # Display top correlations
            display_top_correlations(sorted_correlations, top_n=15)
            
            # Save the sorted results to a new CSV file
            output_filename = 'Phase2_Sorted_Correlations.csv'
            result_df.to_csv(output_filename, index=False)
            print(f"\nResults saved to: {output_filename}")
            
            # Also save just the factor-correlation pairs
            correlation_df = pd.DataFrame(sorted_correlations, 
                                        columns=['Factor', 'Correlation'])
            correlation_output = 'Phase2_Factor_Correlations_Ranked.csv'
            correlation_df.to_csv(correlation_output, index=False)
            print(f"Factor rankings saved to: {correlation_output}")
            
        else:
            print("No Phase2 data found to process.")
            
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_filename}'")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error processing the file: {str(e)}")

# Alternative function for quick analysis
def quick_phase2_analysis(csv_file):
    """
    Quick analysis function that returns just the top correlations.
    """
    df = pd.read_csv(csv_file)
    phase2_row = df[df.iloc[:, 0].str.contains('Phase2', na=False)]
    
    if phase2_row.empty:
        return None
    
    # Get correlations from the first Phase2 row
    correlations = {}
    row = phase2_row.iloc[0]
    
    for col_idx, col_name in enumerate(df.columns[1:], 1):
        value = row.iloc[col_idx]
        if pd.notna(value) and value != '':
            try:
                correlations[col_name] = float(value)
            except (ValueError, TypeError):
                pass
    
    # Sort by absolute correlation value
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_corr

# Example usage for quick analysis:
# top_correlations = quick_phase2_analysis('Top_Correlated_Factors.csv')
# if top_correlations:
#     for factor, corr in top_correlations[:10]:
#         print(f"{factor}: {corr:.6f}")