from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import json
import os
import logging
from models.reply_generator import classify_priority, generate_reply

# Create Flask app
app = Flask(__name__)
app.secret_key = "SECRET KEY"  # Change this to a random string

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
TRAIN_DATA_PATH = "C:/Users/muhil/Downloads/extended_final_data (1).csv"
TEST_DATA_PATH = "C:/Users/muhil/Downloads/shuffled_business_emails.csv"
JSON_PATH = 'models/email_data.json'

# Load test emails from CSV and save as JSON
def load_test_emails():
    if not os.path.exists(TEST_DATA_PATH):
        logging.error(f"Test data file not found at {TEST_DATA_PATH}")
        return False
    try:
        logging.info("Loading test emails from CSV...")
        test_emails = pd.read_csv(TEST_DATA_PATH)
        emails_list = test_emails.to_dict(orient='records')
        with open(JSON_PATH, 'w') as file:
            json.dump(emails_list, file, indent=4)
        logging.info(f"Loaded {len(emails_list)} test emails successfully")
        return True
    except Exception as e:
        logging.exception(f"Error loading test emails: {e}")
        return False

# Load JSON emails
def load_emails():
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Error reading JSON file: {e}")
    return []

@app.route('/')
def dashboard():
    try:
        emails = load_emails()
        logging.info("Dashboard loaded successfully.")
        return render_template('dashboard.html', emails=emails)
    except Exception as e:
        logging.exception(f"Error loading dashboard: {e}")
        flash(f"Error: {e}", "error")
        return render_template('dashboard.html', emails=[])

@app.route('/process_emails', methods=['POST'])
def process_emails():
    try:
        if not os.path.exists(JSON_PATH):
            if not load_test_emails():
                flash("No test emails found. Please check your data file.", "error")
                return redirect(url_for('dashboard'))

        emails = load_emails()
        processed_count = 5
        
        for i, email in enumerate(emails[:11]):  # Process first 5 emails
            try:
                if 'priority' in email and email['priority']:
                    continue  # Skip already processed emails
                
                email_body = email.get('body', '').strip()
                if not email_body:
                    logging.warning(f"Skipping email {i+1}, missing body.")
                    continue
                
                email['priority'] = classify_priority(email_body)
                email['reply'] = generate_reply(email_body, email.get('subject', '')) if email['priority'] == 'urgent' else ""
                processed_count += 1
                logging.info(f"Processed email {i+1}: Priority={email['priority']}")
            except Exception as e:
                logging.exception(f"Error processing email {i+1}: {e}")
                continue
        
        with open(JSON_PATH, 'w') as file:
            json.dump(emails, file, indent=4)
        
        flash(f"Successfully processed {processed_count} emails!", "success")
        logging.info(f"Email processing complete. Processed {processed_count} emails.")
        return redirect(url_for('dashboard'))
    except Exception as e:
        logging.exception(f"Error in email processing: {e}")
        flash(f"Error: {e}", "error")
        return redirect(url_for('dashboard'))

@app.route('/edit_reply/<int:email_id>', methods=['GET', 'POST'])
def edit_reply(email_id):
    try:
        emails = load_emails()
        if email_id < 0 or email_id >= len(emails):
            flash("Invalid email ID", "error")
            return redirect(url_for('dashboard'))
        
        email = emails[email_id]
        if request.method == 'POST':
            email['reply'] = request.form['reply']
            with open(JSON_PATH, 'w') as file:
                json.dump(emails, file, indent=4)
            flash("Reply updated successfully!", "success")
            return redirect(url_for('dashboard'))
        
        return render_template('edit_reply.html', email=email, email_id=email_id)
    except Exception as e:
        logging.exception(f"Error editing reply: {e}")
        flash(f"Error: {e}", "error")
        return redirect(url_for('dashboard'))

@app.route('/get_emails', methods=['GET'])
def get_emails():
    try:
        emails = load_emails()
        return jsonify(emails)
    except Exception as e:
        logging.exception(f"Error fetching emails: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logging.info("Starting Flask app...")
    app.run(debug=True)
