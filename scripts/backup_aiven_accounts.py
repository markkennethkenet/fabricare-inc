"""Create or verify a portable encrypted snapshot of the FabriCare account DB."""

import argparse
import getpass
import gzip
import json
import os
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mysql.connector
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"FABRICARE-ACCOUNTS-V1\0"
TABLE_KEYS = {"users": "id", "auth_tokens": "token"}
SOURCE_HOST = "fabricare-mysql-kentorres903-c399.a.aivencloud.com"


def password_key(passphrase, salt):
    return Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(
        passphrase.encode("utf-8")
    )


def decrypt(data, passphrase):
    if not data.startswith(MAGIC):
        raise ValueError("This is not a FabriCare account backup.")
    offset = len(MAGIC)
    salt, nonce = data[offset:offset + 16], data[offset + 16:offset + 28]
    plaintext = AESGCM(password_key(passphrase, salt)).decrypt(
        nonce, data[offset + 28:], MAGIC
    )
    return json.loads(gzip.decompress(plaintext))


def snapshot(uri, ca):
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        raise ValueError("Use the MySQL Service URI from the fabricare-mysql Overview page.") from None
    if (parsed.scheme != "mysql" or parsed.hostname != SOURCE_HOST or not port
            or not parsed.username or not parsed.password):
        raise ValueError("Use the MySQL Service URI from the fabricare-mysql Overview page.")
    with mysql.connector.connect(
        host=parsed.hostname, port=port, user=unquote(parsed.username),
        password=unquote(parsed.password), ssl_ca=str(ca), ssl_verify_cert=True,
        ssl_verify_identity=True, ssl_disabled=False, connection_timeout=12,
    ) as connection:
        connection.start_transaction(
            consistent_snapshot=True, isolation_level="REPEATABLE READ", readonly=True
        )
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME IN ('users', 'auth_tokens') AND TABLE_TYPE = 'BASE TABLE'
                GROUP BY TABLE_SCHEMA HAVING COUNT(*) = 2
            """)
            matches = [schema for (schema,) in cursor.fetchall()]
            if len(matches) != 1:
                raise ValueError("Could not uniquely identify the FabriCare account schema.")
            schema = matches[0]
            cursor.execute("""
                SELECT TABLE_NAME, ENGINE FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = %s
            """, (schema,))
            if dict(cursor.fetchall()) != dict.fromkeys(TABLE_KEYS, "InnoDB"):
                raise ValueError("The source has unexpected tables or storage engines. "
                                 "Review the schema before backing it up.")
            tables = {}
            quoted = "`" + schema.replace("`", "``") + "`"
            for table, key in TABLE_KEYS.items():
                cursor.execute(f"SHOW CREATE TABLE {quoted}.`{table}`")
                ddl = cursor.fetchone()[1]
                cursor.execute(f"SELECT * FROM {quoted}.`{table}` ORDER BY `{key}`")
                tables[table] = {"ddl": ddl, "columns": list(cursor.column_names),
                                 "rows": cursor.fetchall()}
        connection.rollback()
    return {"version": 1, "source_database": schema,
            "created_at": datetime.now(timezone.utc).isoformat(), "tables": tables}


def create_backup(output, ca):
    if not ca.is_file():
        raise ValueError("The public Aiven CA file is missing.")
    if output.exists():
        raise ValueError("Choose a backup filename that does not already exist.")
    uri = getpass.getpass("Paste the MySQL Service URI (hidden): ").strip()
    passphrase = getpass.getpass("Choose a backup passphrase (hidden, at least 16 characters): ")
    if len(passphrase) < 16 or passphrase != getpass.getpass("Repeat the backup passphrase: "):
        raise ValueError("Passphrases did not match or were shorter than 16 characters.")
    data = snapshot(uri, ca)
    plaintext = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    salt, nonce = os.urandom(16), os.urandom(12)
    encrypted = MAGIC + salt + nonce + AESGCM(password_key(passphrase, salt)).encrypt(
        nonce, gzip.compress(plaintext), MAGIC
    )
    if decrypt(encrypted, passphrase) != json.loads(plaintext):
        raise ValueError("Encrypted backup failed its integrity check.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(encrypted)
        stream.flush()
        os.fsync(stream.fileno())
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--ca", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            data = decrypt(args.file.read_bytes(), getpass.getpass("Backup passphrase (hidden): "))
        else:
            if args.ca is None:
                parser.error("--ca is required for a source snapshot")
            data = create_backup(args.file, args.ca)
        counts = {table: len(entries["rows"]) for table, entries in data["tables"].items()}
        print(f"Encrypted backup verified. Users: {counts['users']}; sessions: {counts['auth_tokens']}.")
        print(args.file.resolve())
    except mysql.connector.Error as error:
        sys.exit(f"MySQL connection or snapshot failed (error {error.errno}).")
    except (InvalidTag, EOFError, OSError, zlib.error):
        sys.exit("Could not open or authenticate the encrypted backup or its input file.")
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
