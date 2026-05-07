import pandas as pd
import json
import os

def process_emails():
    # ✅ Corrected input() usage
    file_path = input("Enter the full file path of the CSV: ").strip()

    # ✅ Convert to absolute path
    file_path = os.path.abspath(file_path)

    # ✅ Check if file exists
    if not os.path.exists(file_path):
        print("❌ Error: The file path is incorrect or the file does not exist.")
        return

    print(f"✅ Loading file: {file_path}")

    try:
        # ✅ Load CSV file
        df = pd.read_csv(file_path)

        # ✅ Check if "emails" column exists
        if "emails" not in df.columns:
            print("❌ Error: 'emails' column not found in dataset.")
            return

        # ✅ Convert email data into a list of dictionaries
        emails = df[["emails"]].rename(columns={"emails": "body"}).to_dict(orient="records")

        # ✅ Save to JSON file inside 'models' folder
        json_path = os.path.join("models", "email_data.json")
        with open(json_path, "w") as file:
            json.dump(emails, file, indent=4)

        print(f"✅ Successfully processed {len(emails)} emails!")
        print(f"✅ JSON saved at: {json_path}")

    except Exception as e:
        print(f"❌ Error while processing: {e}")

# ✅ Run the function
if __name__ == "__main__":
    process_emails()
