import pandas as pd


# Define the file paths
file1_path = 'International Astronaut Database.csv'
file2_path = 'astronauts.csv'
output_path = 'joined_astronauts.csv'

try:
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Function to extract the first and last name from a full name string.
    # This assumes names are separated by spaces.
    def get_first_last_name(full_name):
        # Handle cases where the name might be missing or not a string
        if not isinstance(full_name, str):
            return None
        
        # Split the name into parts (e.g., 'John Fitzgerald Kennedy' -> ['John', 'Fitzgerald', 'Kennedy'])
        name_parts = full_name.split()
        
        # If there are two or more parts, join the first and last parts.
        # Otherwise, just return the single part or None.
        if len(name_parts) >= 2:
            return f"{name_parts[0]} {name_parts[-1]}"
        elif len(name_parts) == 1:
            return name_parts[0]
        else:
            return None

    # Function to extract only the state from birth place data
    def extract_state_from_birth_place(birth_place):
        # Handle cases where the birth place might be missing or not a string
        if not isinstance(birth_place, str):
            return birth_place
        
        # Dictionary mapping state abbreviations to full names
        state_abbreviations = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        
        # Split by comma and take the last part (which should be the state/country)
        # This handles formats like "City, State" or "City, State, Country"
        parts = [part.strip() for part in birth_place.split(',')]
        
        # Get the last part (state) if there are multiple parts, otherwise return as is
        if len(parts) >= 2:
            state_part = parts[-1].strip()
            # Check if it's a state abbreviation and replace with full name
            if state_part.upper() in state_abbreviations:
                return state_abbreviations[state_part.upper()]
            else:
                return state_part
        else:
            # Check if the entire string is a state abbreviation
            cleaned_birth_place = birth_place.strip()
            if cleaned_birth_place.upper() in state_abbreviations:
                return state_abbreviations[cleaned_birth_place.upper()]
            else:
                return birth_place  # Return as is if no comma found and not an abbreviation

    # Apply the function to create a new 'first_last' column for merging.
    # The current assumption is that the column containing the full name is named 'Name'.
    # You MUST check your CSV files and replace 'Name' with the correct column name if it is different.
    df1['first_last'] = df1['Name'].apply(get_first_last_name)
    df2['first_last'] = df2['Name'].apply(get_first_last_name)
    
    # Process Birth Place columns to extract only the state
    if 'Birth Place' in df1.columns:
        df1['Birth Place'] = df1['Birth Place'].apply(extract_state_from_birth_place)
    if 'Birth Place' in df2.columns:
        df2['Birth Place'] = df2['Birth Place'].apply(extract_state_from_birth_place)
    
    # Merge the two DataFrames on the new standardized name column
    # The 'inner' join keeps only the records that have a match in both files.
    joined_df = pd.merge(df1, df2, on='first_last', how='inner', suffixes=('_db', '_scrape'))

    # Drop the temporary 'first_last' column from the final output for cleanliness
    joined_df.drop(columns=['first_last'], inplace=True)
    
    # Delete the Name_scrape and Gender_scrape columns
    columns_to_drop = []
    if 'Name_scrape' in joined_df.columns:
        columns_to_drop.append('Name_scrape')
    if 'Gender_scrape' in joined_df.columns:
        columns_to_drop.append('Gender_scrape')
    
    if columns_to_drop:
        joined_df.drop(columns=columns_to_drop, inplace=True)
    
    # Rename columns as requested
    column_renames = {}
    if 'Name_db' in joined_df.columns:
        column_renames['Name_db'] = 'Name'
    if 'Gender_db' in joined_df.columns:
        column_renames['Gender_db'] = 'Gender'
    
    if column_renames:
        joined_df.rename(columns=column_renames, inplace=True)
    
    # Replace empty data spaces with 0
    # This handles NaN values, empty strings, and whitespace-only strings
    joined_df = joined_df.fillna(0)  # Replace NaN with 0
    joined_df = joined_df.replace('', 0)  # Replace empty strings with 0
    joined_df = joined_df.replace(r'^\s*$', 0, regex=True)  # Replace whitespace-only strings with 0
    
    # Save the joined DataFrame to a new CSV file
    joined_df.to_csv(output_path, index=False)
    
    print(f"Tables successfully joined and saved to {output_path}")
    print("\nHere are the first 5 rows of the new table:")
    print(joined_df.head())

except FileNotFoundError:
    print("Error: One of the files was not found. Please make sure your CSV files are in the same directory as this Python script and that the file names match what is in the code.")
except KeyError as e:
    print(f"Error: The assumed column name for the full name ('Name') was not found. Please open your CSV files and replace 'Name' in the code with the correct column name. Missing key: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    