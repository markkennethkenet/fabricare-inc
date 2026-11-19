import logging
import os
import secrets
import time

from flask import Blueprint, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash
from mysql.connector import Error as MySqlError

from database import PostgresUnavailable, get_connection, insert_user

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
log = logging.getLogger(__name__)


@auth_bp.errorhandler(MySqlError)
@auth_bp.errorhandler(PostgresUnavailable)
def database_unavailable(_error):
    return jsonify({"error": "Account service is temporarily unavailable. Please try again later."}), 503


def _user_row_to_dict(row):
    return {
        "id": row[0],
        "full_name": row[1],
        "email": row[2],
        "phone": row[3] or "",
        "location": row[4] or "",
        "created_at": row[5],
    }


def _fetch_user_by_id(user_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, phone, location, created_at
            FROM users WHERE id = %s
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()
    if row is None:
        return None
    return _user_row_to_dict(row)


def _fetch_user_by_email(email):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, phone, location, created_at
            FROM users WHERE email = %s
            """,
            (email.lower().strip(),),
        )
        row = cursor.fetchone()
        cursor.close()
    if row is None:
        return None
    return _user_row_to_dict(row)


def _create_token(user_id, expires_at=None):
    token = secrets.token_urlsafe(32)
    now = int(time.time() * 1000)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO auth_tokens (token, user_id, created_at, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (token, user_id, now, expires_at),
        )
        cursor.close()
    return token


def _delete_token(token):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM auth_tokens WHERE token = %s", (token,))
        cursor.close()


def _user_id_from_token(token):
    if not token:
        return None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id FROM auth_tokens
            WHERE token = %s AND (expires_at IS NULL OR expires_at > %s)
            """,
            (token, int(time.time() * 1000)),
        )
        row = cursor.fetchone()
        cursor.close()
    if row is None:
        return None
    return row[0]


def _auth_user():
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    user_id = _user_id_from_token(token)
    if user_id is None:
        return None, token
    user = _fetch_user_by_id(user_id)
    if user is None:
        if token:
            _delete_token(token)
        return None, token
    return user, token


def _require_auth():
    user, token = _auth_user()
    if user is None:
        return None, (jsonify({"error": "Unauthorized."}), 401)
    return user, None


@auth_bp.post("/demo")
def demo_sign_in():
    code = os.environ.get("FABRICARE_DEMO_CODE")
    user_id = os.environ.get("FABRICARE_DEMO_USER_ID", "")
    expiry = os.environ.get("FABRICARE_DEMO_EXPIRES_AT", "")
    if (
        request.remote_addr not in {"127.0.0.1", "::1"}
        or not code
        or not user_id.isdecimal()
        or (expiry != "never" and not expiry.isdecimal())
    ):
        return jsonify({"error": "Demo sign-in is unavailable."}), 404
    expires_at = None if expiry == "never" else int(expiry)
    if expires_at is not None and expires_at <= int(time.time() * 1000):
        return jsonify({"error": "This demo account has expired."}), 401
    data = request.get_json(silent=True)
    entered = data.get("code") if isinstance(data, dict) else None
    if not isinstance(entered, str) or not secrets.compare_digest(entered.strip().encode(), code.encode()):
        return jsonify({"error": "Incorrect demo code."}), 401
    user = _fetch_user_by_id(int(user_id))
    if user is None:
        return jsonify({"error": "Demo account is unavailable."}), 503
    return jsonify({"user": user, "token": _create_token(user["id"], expires_at), "expires_at": expires_at})


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""

    if not full_name or not email or not password:
        return jsonify({"error": "Full name, email, and password are required."}), 400

    if _fetch_user_by_email(email):
        return jsonify({"error": "An account with this email already exists."}), 409

    password_hash = generate_password_hash(password)
    created_at = int(time.time() * 1000)

    with get_connection() as conn:
        user_id = insert_user(conn, full_name, email, password_hash, created_at)

    user = _fetch_user_by_id(user_id)
    token = _create_token(user_id)
    return jsonify({"token": token, "user": user}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, phone, location, created_at, password_hash
            FROM users WHERE email = %s
            """,
            (email,),
        )
        row = cursor.fetchone()
        cursor.close()

    if row is None:
        return jsonify({"error": "No account found with this email."}), 401

    stored_hash = row[6]
    if not check_password_hash(stored_hash, password):
        return jsonify({"error": "Incorrect password."}), 401

    user = _user_row_to_dict(row[:6])
    token = _create_token(user["id"])
    return jsonify({"token": token, "user": user})


# FabriCare's public OAuth client IDs (web, Android). Phones send ID tokens issued to the web client.
DEFAULT_GOOGLE_CLIENT_IDS = (
    "1086302215166-3eoejdqn579frtqsukqtgndm9dbafl5l.apps.googleusercontent.com",
    "1086302215166-c74ai1kdmht8ma1m173933hun6q76p34.apps.googleusercontent.com",
)


def google_client_ids():
    """OAuth client IDs whose Google ID tokens this API accepts.

    GOOGLE_CLIENT_IDS (comma-separated) replaces the defaults; set it empty to turn Google sign-in off.
    """
    configured = os.environ.get("GOOGLE_CLIENT_IDS")
    if configured is None:
        return list(DEFAULT_GOOGLE_CLIENT_IDS)
    return [value.strip() for value in configured.split(",") if value.strip()]


def verify_google_token(token, audiences):
    """Check the ID token's signature, expiry, issuer and audience with Google's public keys."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(token, google_requests.Request(), audience=audiences,
                                        clock_skew_in_seconds=10)


@auth_bp.route("/google", methods=["POST"])
def google_sign_in():
    """Sign in with a Google ID token. A new email creates an account named after the Google profile;
    an existing email signs in to that account, so earlier scans stay with it."""
    data = request.get_json(silent=True) or {}
    token = data.get("id_token")
    if not isinstance(token, str) or not token.strip() or len(token) > 8192:
        return jsonify({"error": "Google did not return a sign-in token. Try again."}), 400
    audiences = google_client_ids()
    if not audiences:
        return jsonify({"error": "Google sign-in is not set up on the server yet."}), 503
    try:
        from google.auth.exceptions import GoogleAuthError, TransportError

        claims = verify_google_token(token.strip(), audiences)
    except TransportError:
        log.warning("Google sign-in keys could not be fetched")
        return jsonify({"error": "Could not reach Google to verify the sign-in. Try again."}), 503
    except (ValueError, GoogleAuthError):
        return jsonify({"error": "Google sign-in could not be verified. Try again."}), 401
    email = str(claims.get("email") or "").lower().strip()
    if not email or claims.get("email_verified") is not True:
        return jsonify({"error": "Use a Google account with a verified email address."}), 401
    name = (str(claims.get("name") or "").strip() or email.split("@")[0])[:255]

    user = _fetch_user_by_email(email)
    created = user is None
    if created:
        # Google accounts have no FabriCare password; store an unusable random hash.
        with get_connection() as conn:
            user_id = insert_user(conn, name, email, generate_password_hash(secrets.token_urlsafe(32)),
                                  int(time.time() * 1000))
        user = _fetch_user_by_id(user_id)
    picture = claims.get("picture")
    return jsonify({
        "token": _create_token(user["id"]), "user": user, "created": created,
        "picture": picture if isinstance(picture, str) and picture.startswith("https://") else None,
    }), 201 if created else 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    _, token = _auth_user()
    if token:
        _delete_token(token)
    return jsonify({"message": "Logged out."})


@auth_bp.route("/me", methods=["GET"])
def me():
    user, error = _require_auth()
    if error:
        return error
    return jsonify({"user": user})


@auth_bp.route("/profile", methods=["PUT"])
def update_profile():
    user, error = _require_auth()
    if error:
        return error

    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").lower().strip()
    phone = (data.get("phone") or "").strip()
    location = (data.get("location") or "").strip()

    if not full_name or not email:
        return jsonify({"error": "Full name and email are required."}), 400

    if email != user["email"]:
        existing = _fetch_user_by_email(email)
        if existing and existing["id"] != user["id"]:
            return jsonify({"error": "This email is already used by another account."}), 409

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET full_name = %s, email = %s, phone = %s, location = %s
            WHERE id = %s
            """,
            (full_name, email, phone, location, user["id"]),
        )
        cursor.close()

    updated = _fetch_user_by_id(user["id"])
    return jsonify({"user": updated})


@auth_bp.route("/password", methods=["PUT"])
def change_password():
    user, error = _require_auth()
    if error:
        return error

    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""

    if not current_password or not new_password:
        return jsonify({"error": "Current and new password are required."}), 400

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password_hash FROM users WHERE id = %s",
            (user["id"],),
        )
        row = cursor.fetchone()
        cursor.close()

    if row is None or not check_password_hash(row[0], current_password):
        return jsonify({"error": "Current password is incorrect."}), 401

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (generate_password_hash(new_password), user["id"]),
        )
        cursor.close()

    return jsonify({"message": "Password updated."})
