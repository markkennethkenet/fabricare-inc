# FabriCare hosted account migration

The source is the MySQL service `fabricare-mysql` and the destination is the
free PostgreSQL service `fabricare-postgres` in the same Aiven project. The
current online deployment keeps using MySQL until the final cutover.

## Export

Use the MySQL **Service URI** from the Aiven Overview page. The backup tool
accepts it at a hidden prompt and connects with TLS hostname and CA verification.
It takes a read-only, consistent snapshot of the `users` and `auth_tokens`
tables, including their MySQL table definitions. It refuses an unexpected
schema so that other tables cannot be silently omitted.

Run in a local PowerShell window. Save the backup passphrase separately from
the backup file. Keep a second copy of the encrypted file outside the PC.

```powershell
$env:PYTHONPATH = 'C:\flutter-project\tmp\fabricare-migration-deps'
$backupDir = 'C:\flutter-project\output\fabricare-account-backups'
Set-Location C:\flutter-project\fabricare
.\.venv-api\Scripts\python.exe scripts\backup_aiven_accounts.py "$backupDir\accounts-before-postgres.fcbak" --ca "$backupDir\aiven-project-ca.pem"
.\.venv-api\Scripts\python.exe scripts\backup_aiven_accounts.py "$backupDir\accounts-before-postgres.fcbak" --verify
```

The compressed snapshot is encrypted with AES-256-GCM. The key is derived from
the passphrase with scrypt and a random salt. The file does not include the
service URI or password. Verification decrypts the content without printing
account fields.

## Rehearse the restore

Use the PostgreSQL service URI at the hidden prompt. The restore checks TLS,
requires an empty destination, writes both tables in one transaction, and checks
every restored row against the snapshot before committing. The next registration
uses an ID above the highest migrated ID.

```powershell
.\.venv-api\Scripts\python.exe scripts\restore_aiven_accounts_postgres.py "$backupDir\accounts-before-postgres.fcbak" --ca "$backupDir\aiven-project-ca.pem"
```

Do not use a stale rehearsal snapshot for production cutover. If rehearsal
already populated the destination, create a fresh database in the PostgreSQL
service for the final import so the restore still has an empty target.

## Cutover

1. Stage the PostgreSQL backend against a rehearsal database. Configure
   `POSTGRES_DSN` and `POSTGRES_SSL_CA_PEM` as server-only environment
   variables. Keep the existing MySQL configuration available for rollback.
2. Verify registration, login, logout, and account lookup using the staged
   backend. Check a migrated account and its existing session before opening
   the destination to production traffic.
3. Schedule a brief pause in account writes, including registration, login
   sessions, profile edits, and logout. Take and verify a final source snapshot
   during that pause. Import it into an empty final destination and verify the
   row counts and exact content.
4. Point the staged production deployment at the final database, verify it,
   and then switch the live app. Check API readiness and auth over the public
   URL before ending the pause. Keep both backups and the MySQL service until
   after verification.
5. If the switch fails before PostgreSQL accepts writes, point the app back
   at MySQL. If PostgreSQL has accepted writes, reconcile those writes before
   rolling back so new accounts and password changes are preserved.
