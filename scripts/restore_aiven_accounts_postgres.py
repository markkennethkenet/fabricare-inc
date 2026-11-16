"""Restore a verified FabriCare snapshot into a new PostgreSQL database."""

import argparse
import getpass
import sys
from pathlib import Path

import psycopg
from cryptography.exceptions import InvalidTag
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from backup_aiven_accounts import decrypt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from database import init_postgres_schema


TABLE_COLUMNS = {
    "users": ("id", "full_name", "email", "password_hash", "phone", "location", "created_at"),
    "auth_tokens": ("token", "user_id", "created_at", "expires_at"),
}
DESTINATION = "fabricare-postgres-kentorres903-c399.b.aivencloud.com"


def restore(connection, data):
    if data.get("version") != 1 or set(data["tables"]) != set(TABLE_COLUMNS):
        raise ValueError("The snapshot does not contain the expected account tables.")
    init_postgres_schema(connection)
    with connection.cursor() as cursor:
        cursor.execute("LOCK TABLE users, auth_tokens IN SHARE ROW EXCLUSIVE MODE")
        counts = {}
        for table, columns in TABLE_COLUMNS.items():
            entry = data["tables"][table]
            if set(entry["columns"]) != set(columns) or len(entry["columns"]) != len(columns):
                raise ValueError(f"Unexpected source columns in {table}.")
            fields = tuple(entry["columns"])
            relation = sql.Identifier(table)
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(relation))
            if cursor.fetchone()[0]:
                raise ValueError(f"The destination {table} table is not empty.")
            rows = [tuple(row) for row in entry["rows"]]
            if rows:
                cursor.executemany(
                    sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        relation, sql.SQL(", ").join(map(sql.Identifier, fields)),
                        sql.SQL(", ").join(sql.Placeholder() for _ in fields)
                    ), rows
                )
            cursor.execute(sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(", ").join(map(sql.Identifier, fields)), relation
            ))
            restored = cursor.fetchall()
            if len(restored) != len(rows) or set(restored) != set(rows):
                raise ValueError(f"Row verification failed for {table}.")
            counts[table] = len(restored)
        cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM users")
        next_user_id = cursor.fetchone()[0]
        cursor.execute(sql.SQL("ALTER TABLE users ALTER COLUMN id RESTART WITH {}").format(
            sql.Literal(next_user_id)
        ))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("--ca", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not args.ca.is_file() or not args.backup.is_file():
            raise ValueError("The encrypted backup and Aiven project CA file must exist.")
        data = decrypt(args.backup.read_bytes(), getpass.getpass("Backup passphrase (hidden): "))
        params = conninfo_to_dict(getpass.getpass("Paste the new PostgreSQL Service URI (hidden): ").strip())
        if params.get("host") != DESTINATION or not params.get("password"):
            raise ValueError("Use the Service URI for the new fabricare-postgres service.")
        params.update(sslmode="verify-full", sslrootcert=str(args.ca.resolve()), connect_timeout=12)
        with psycopg.connect(**params) as connection:
            counts = restore(connection, data)
        print(f"Restored and verified {counts['users']} account(s) and "
              f"{counts['auth_tokens']} session(s) in PostgreSQL.")
    except InvalidTag:
        sys.exit("The backup passphrase was incorrect or the backup failed authentication.")
    except psycopg.Error as error:
        sys.exit(f"PostgreSQL operation failed (SQLSTATE {error.sqlstate or 'unavailable'}).")
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
