import joblib
import mysql.connector
from flask import Flask, request, render_template, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secure key

# Load the trained model
model = joblib.load('model.pkl')

# MySQL Database Connection
db_config = {
    'host': 'localhost',
    'user': 'root',  # Change if needed
    'password': 'Gowtham@2203',  # Set your MySQL password
    'database': 'epharma'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user:
            cursor.close()
            conn.close()
            return render_template('signup.html', error="Username already exists!")

        # Insert new user into the database
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()

        cursor.close()
        conn.close()

        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Get user inputs
            age = int(request.form['age'])
            gender = request.form['Gender']
            symptoms = request.form['symptoms'].strip()
            
            # Validate symptoms input
            symptom_list = [sym.strip() for sym in symptoms.split() if sym.strip()]
            
            # Strict 1-3 symptoms validation
            if len(symptom_list) < 1 or len(symptom_list) > 3:
                return render_template('index.html', error="Please enter 1 to 3 symptoms.")
            
            # Pad symptoms if less than 3
            while len(symptom_list) < 3:
                symptom_list.append('')
            
            # Combine symptoms into a single string
            symptom_string = ' '.join(symptom_list[:3])
            
            # Predict medicine
            prediction = model.predict([symptom_string])
            medicine = prediction[0]
            
            return render_template('results.html', gender=gender, age=age, symptoms=symptoms, medicine=medicine)
        
        except ValueError:
            return render_template('index.html', error="Please enter valid data for age.")

    return render_template('index.html')

# Emergency First Aid Routes
@app.route("/cpr")
def cpr():
    return render_template("first_aid/cpr.html")

@app.route("/electric_shock")
def electric_shock():
    return render_template("first_aid/electric_shocks.html")

@app.route("/fire_accidents")
def fire_accidents():
    return render_template("first_aid/fire_accidents.html")

@app.route("/snake_bites")
def snake_bites():
    return render_template("first_aid/snake_bites.html")

# Run the Flask app
'''if __name__ == "__main__":
    app.run(debug=True)'''
  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

