from flask import Flask, request, render_template, redirect, url_for, abort, session

try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.models import load_model
except ModuleNotFoundError as exc:
    if exc.name in {"tensorflow", "tensorflow.python"}:
        raise ModuleNotFoundError(
            "TensorFlow isn't available in the Python environment running this app.\n"
            "\n"
            "Fix (Windows / PowerShell):\n"
            "  .\\venv\\Scripts\\Activate.ps1\n"
            "  python -m pip install -r requirements.txt\n"
            "\n"
            "Then verify:\n"
            "  python -c \"import tensorflow as tf; import tensorflow.python; print(tf.__version__)\""
        ) from exc
    raise
import numpy as np
import os
import json
import uuid
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or "dev-secret-key-change-me"

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'abgowda216@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'jfwkhdqkvehlsvzi')
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model("atopic_dermatitis_model.h5")
    return _model

UPLOAD_DIR = os.path.join('static', 'uploads')
DATA_DIR = 'data'
RECORDS_PATH = os.path.join(DATA_DIR, 'records.json')
USERS_PATH = os.path.join(DATA_DIR, 'users.json')
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


def _ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_records():
    if not os.path.exists(RECORDS_PATH):
        return []
    with open(RECORDS_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _append_record(record):
    records = _load_records()
    records.append(record)
    with open(RECORDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _get_record(record_id):
    for record in reversed(_load_records()):
        if record.get('id') == record_id:
            return record
    return None


def _load_users():
    if not os.path.exists(USERS_PATH):
        return []
    with open(USERS_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_users(users):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character."
    return True, ""


def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def send_verification_email(user):
    verify_token_bytes = base64.urlsafe_b64encode(user['verify_token'].encode())
    token = verify_token_bytes.decode('utf-8')
    verify_url = url_for('verify', token=token, _external=True)
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Verify your Atopic App account'
    msg['From'] = app.config['MAIL_USERNAME']
    msg['To'] = user['email']
    
    html = f"""
    <html>
      <body>
        <h2>Verify your email</h2>
        <p>Hi {user['username']},</p>
        <p>Please click <a href="{verify_url}">here</a> to verify your email.</p>
        <p>Or copy: {verify_url}</p>
        <p>Thanks,<br>Atopic App</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', app.config['MAIL_PORT'])
        server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Verification email failed: {e}")
        return False


def _get_user_by_id(user_id):
    for user in _load_users():
        if user.get("id") == user_id:
            return user
    return None


def _get_user_by_username(username):
    username = (username or "").strip().lower()
    for user in _load_users():
        if (user.get("username") or "").lower() == username:
            return user
    return None


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return _get_user_by_id(user_id)


@app.context_processor
def _inject_user():
    return {"current_user": current_user()}


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


def _allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def _image_metadata(path):
    metadata = {}
    try:
        metadata["size_bytes"] = os.path.getsize(path)
    except OSError:
        metadata["size_bytes"] = None

    try:
        from PIL import Image  # Pillow

        with Image.open(path) as im:
            metadata["width"] = im.width
            metadata["height"] = im.height
            metadata["mode"] = im.mode
            metadata["format"] = im.format
    except Exception:
        metadata.setdefault("width", None)
        metadata.setdefault("height", None)
        metadata.setdefault("mode", None)
        metadata.setdefault("format", None)

    return metadata


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    _ensure_dirs()

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        if len(username) < 3 or not re.match(r"^[a-zA-Z0-9_]+$", username):
            return render_template('register.html', error="Username must be at least 3 characters, alphanumeric only.", username=username)
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        if not username or not password:
            return render_template('register.html', error="Username and password are required.", username=username)
        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match.", username=username)
        if _get_user_by_username(username):
            return render_template('register.html', error="Username already exists.", username=username)

        users = _load_users()
        user = {
            "id": uuid.uuid4().hex,
            "username": username,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Add email validation and strength
        email = request.form.get('email') or ''
        if not email or not validate_email(email):
            return render_template('register.html', error="Valid email is required.", username=username)
        pw_valid, pw_msg = validate_password(password)
        if not pw_valid:
            return render_template('register.html', error=pw_msg, username=username)

        # Check if email already used
        for u in users:
            if u.get('email', '').lower() == email.lower():
                return render_template('register.html', error="Email already registered.", username=username)

        user = {
            "id": uuid.uuid4().hex,
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password),
            "is_verified": False,
            "verify_token": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        users.append(user)
        _save_users(users)

        email_sent = send_verification_email(user)
        if email_sent:
            return render_template('register.html', success="Account created! Check your email for verification link.", username=username)
        else:
            return render_template('register.html', error="Account created but verification email failed to send. Please contact support.", username=username)

    return render_template('register.html')


@app.route('/verify/<token>')
def verify(token):
    users = _load_users()
    user = None
    for u in users:
        if base64.urlsafe_b64decode(token).decode() == u.get('verify_token'):
            u['is_verified'] = True
            _save_users(users)
            session['user_id'] = u['id']
            return redirect(url_for('index'))
    return render_template('register.html', error="Invalid or expired verification token.")


@app.route('/login', methods=['GET', 'POST'])
def login():
    _ensure_dirs()

    next_url = request.args.get("next") or url_for("index")

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        user = _get_user_by_username(username)
        if not user or not check_password_hash(user.get("password_hash", ""), password):
            return render_template('login.html', error="Invalid username or password.", username=username, next=next_url)
        if not user.get('is_verified', True):
            return render_template('login.html', error="Please verify your email address first. Check your inbox.", username=username, next=next_url)

        session["user_id"] = user["id"]
        return redirect(next_url)

    return render_template('login.html', next=next_url)


@app.route('/logout', methods=['GET'])
def logout():
    session.pop("user_id", None)
    return redirect(url_for('index'))


@app.route('/reset_password', methods=['GET', 'POST'])
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token=None):
    if request.method == 'GET':
        return render_template('reset_password.html', token=token)
    email = request.form.get('email') or ''
    if email:
        if not validate_email(email):
            return render_template('reset_password.html', error="Invalid email.")
        users = _load_users()
        user = None
        for u in users:
            if u.get('email', '').lower() == email.lower():
                user = u
                break
        if user:
            reset_token = str(uuid.uuid4())
            user['reset_token'] = reset_token
            _save_users(users)
            send_reset_email(user)
            return render_template('reset_password.html', success="Reset link sent to your email.")
        else:
            return render_template('reset_password.html', error="Email not found.")
    else:
        # Reset with token
        new_pw = request.form.get('new_password') or ''
        confirm_pw = request.form.get('confirm_password') or ''
        if new_pw != confirm_pw:
            return render_template('reset_password.html', token=token, error="Passwords do not match.")
        pw_valid, pw_msg = validate_password(new_pw)
        if not pw_valid:
            return render_template('reset_password.html', token=token, error=pw_msg)
        users = _load_users()
        for u in users:
            if u.get('reset_token') == token:
                u['password_hash'] = generate_password_hash(new_pw)
                u.pop('reset_token', None)
                _save_users(users)
                return redirect(url_for('login'))
        return render_template('reset_password.html', token=token, error="Invalid or expired token.")


def send_reset_email(user):
    token = user['reset_token']
    reset_url = url_for('reset_password', token=token, _external=True)
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Password Reset - Atopic App'
    msg['From'] = app.config['MAIL_USERNAME']
    msg['To'] = user['email']
    
    html = f"""
    <html>
      <body>
        <h2>Password Reset</h2>
        <p>Hi {user['username']},</p>
        <p>Click <a href="{reset_url}">here</a> to reset your password.</p>
        <p>Or copy: {reset_url}</p>
        <p>If you didn't request this, ignore.</p>
        <p>Thanks,<br>Atopic App</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', app.config['MAIL_PORT'])
        server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Reset email failed: {e}")
        return False


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    user = current_user()
    if request.method == 'POST':
        current_pw = request.form.get('current_password') or ''
        new_pw = request.form.get('new_password') or ''
        confirm_new = request.form.get('confirm_new_password') or ''
        if new_pw != confirm_new:
            return render_template('change_password.html', error="New passwords do not match.")
        if not check_password_hash(user['password_hash'], current_pw):
            return render_template('change_password.html', error="Current password incorrect.")
        pw_valid, pw_msg = validate_password(new_pw)
        if not pw_valid:
            return render_template('change_password.html', error=pw_msg)
        
        users = _load_users()
        for u in users:
            if u['id'] == user['id']:
                u['password_hash'] = generate_password_hash(new_pw)
                break
        _save_users(users)
        return render_template('change_password.html', success="Password updated successfully!")
    return render_template('change_password.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    _ensure_dirs()

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if not file or file.filename == '':
        return "No selected file", 400

    original_filename = file.filename
    safe_name = secure_filename(original_filename)
    if not safe_name or not _allowed_file(safe_name):
        return "Unsupported file type. Use PNG/JPG/JPEG/WEBP.", 400

    record_id = uuid.uuid4().hex
    stored_filename = f"{record_id}_{safe_name}"
    img_path = os.path.join(UPLOAD_DIR, stored_filename)
    file.save(img_path)
    img_meta = _image_metadata(img_path)

    patient_name = (request.form.get('patient_name') or '').strip()
    patient_age = (request.form.get('patient_age') or '').strip()
    symptoms = (request.form.get('symptoms') or '').strip()
    notes = (request.form.get('notes') or '').strip()

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = _get_model().predict(img_array)
    score = float(prediction[0][0])
    if score > 0.5:
        result = "Atopic Dermatitis"
        description = [
            "Atopic dermatitis, also known as eczema, is a chronic skin condition.",
            "It causes itchy, inflamed skin.",
            "It is common in children but can occur at any age.",
            "Symptoms include dry, itchy skin, redness, and rashes.",
        ]
        related_names = ["Eczema", "Atopic Eczema", "Dermatitis"]
    else:
        result = "Bullous Disease"
        description = [
            "Bullous diseases are a group of skin disorders.",
            "They are characterized by the formation of blisters or bullae on the skin.",
            "These can be caused by autoimmune conditions, infections, or other factors.",
        ]
        related_names = [
            "Blistering Diseases",
            "Pemphigus",
            "Bullous Pemphigoid",
            "Epidermolysis Bullosa",
        ]

    record = {
        "id": record_id,
        "user_id": session.get("user_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "image_url": url_for('static', filename=f"uploads/{stored_filename}"),
        "image_meta": img_meta,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "symptoms": symptoms,
        "notes": notes,
        "prediction": result,
        "score": score,
        "description": description,
        "related_names": related_names,
    }
    _append_record(record)

    return redirect(url_for('report', record_id=record_id))


@app.route('/report/<record_id>', methods=['GET'])
@login_required
def report(record_id):
    record = _get_record(record_id)
    if not record:
        abort(404)
    if record.get("user_id") and record.get("user_id") != session.get("user_id"):
        abort(404)
    return render_template('report.html', record=record)

if __name__ == '__main__':
    _ensure_dirs()
    app.run(debug=True)
