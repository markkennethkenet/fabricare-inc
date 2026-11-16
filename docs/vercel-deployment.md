# FabriCare on Vercel

Target workspace: `markkennethkenets-projects`.
Repository: `markkennethkenet/fabricare1`.
Current source branch: `capstone/ai-skills-activity`.
Latest source: `9f33063da4ae56db18d77cd9a1db6384099cc9cc`.
Vercel project: `fabricare1` (`prj_AVitYm0TBIjaicO5sjHGr13diMPr`).
Production deployment: `dpl_G5qpJCv5ucCbTYBpF2d1XPPiJH6g`.
Production URL: https://fabricare1-markkennethkenets-projects.vercel.app
Build URL: https://fabricare1-5j197c622-markkennethkenets-projects.vercel.app

Vercel reports READY. After explicit user approval, production was published
with Standard Protection (`prod_deployment_urls_and_all_previews`). Anonymous
requests to the production homepage and /api/health/ready returned HTTP 200;
the API returned {"database_connected":true,"ready":true}. Generated deployment
URLs still redirect anonymous visitors to Vercel sign-in. Unauthenticated
/api/auth/me returns HTTP 401, so publishing production did not expose profiles.

The Python container built successfully on Vercel with the existing model
files and CPU-only dependencies. Live browser verification passed on the actual
production app: signup with one disposable example.test account, profile showing
the registered name and email, logout to the login form, rejection of an
incorrect password, successful fresh login, and session restoration after a
browser reload. A final logout returned the app to its empty login form; the
disposable account has no active test session. One synthetic example.test
verification account remains in the new database. Its password was cleared
from the setup session and was not recorded in this report. No real user
credentials or records were used for the test.
Garment analysis was not tested as part of this authentication fix.

The final signed-out screen is saved locally as
`C:/flutter-project/auth-verification.jpg`. Existing local or Railway account
data was not migrated to this new Aiven database; users should sign up first.

Vercel workspace access was verified on October 25, 2026. The connected
workspace is on the Hobby plan. After learning Railway's trial had expired,
the user selected Aiven's free MySQL tier instead and completed sign-in.
Never request passwords in chat.

Aiven service `fabricare-mysql` is RUNNING in project
`kentorres903-c399` on the Free-1-1gb plan in Asia Pacific (DigitalOcean blr).
Its generated connection settings and project CA certificate were transferred
directly from the browser to Vercel as six production-only Secret variables.
No password or connection URI was printed, saved to source, or added to the
Flutter build. MYSQL_SSL_CA_PEM contains the project CA certificate; the backend
now verifies both the certificate chain and server hostname. Aiven connections
without a configured CA are refused. The local MySQL configuration is unchanged.

All 28 backend tests passed after the TLS change, including three new tests
for certificate/hostname verification, refusing an Aiven connection without a
CA, and preserving the existing local MySQL connection behavior. All five
Flutter tests passed, as did two API_BASE_URL override tests and targeted Dart
analysis. App builds now default to the hosted production API. API_BASE_URL
still overrides that address; API_HOST or USE_LOCAL_API=true enables local
development. Existing app installations need a rebuild/restart for this change.

Android release APK builds were attempted twice, including an IPv4 retry.
Gradle failed with "java.io.IOException: Unable to establish loopback
connection" in this execution environment. No new APK was produced; do not
distribute an older APK as the updated build.

Railway's existing project `empathetic-hope` contains an offline MySQL service
and a persistent volume. Its dashboard reports "Trial expired" and requires a
plan upgrade to continue. No Railway service, volume, existing data, or billing
plan was changed. Preserve that database; the user chose a separate Aiven
service and did not approve a Railway upgrade. Data migration remains pending.

The deployment uses Vercel Services (beta) to ship two components together:

- `web`: Flutter 3.41.4, built from its pinned SDK revision into `web/dist`.
- `api`: the existing Python 3.12 API, built with `Dockerfile.vercel` and
  CPU-only PyTorch. Model versions and model files are preserved.

The browser calls `/api` on its own origin. Public `/api/...` requests route to
the API service, whose rewrite removes `/api` before Flask receives the request.
Other requests serve the Flutter web app. Deep links resolve to `index.html`.

Enable "Include source files outside of the Root Directory in the Build Step"
on the Vercel project: the web service's build needs the parent Flutter source
and build script. The deployment root is the repository root. Do not set the
project root to `web` or change its framework to Flask.

`.vercelignore` and `.dockerignore` allowlist application/build inputs, including
`model_analysis.py`, and keep
local credentials, virtual environments, datasets, and build outputs out of the
deployment. Do not publish `.env` or place database passwords in Flutter.

## Database prerequisite

The existing authentication service requires MySQL. A hosted database must be
provisioned before this deployment can become ready. Configure these server
environment variables using the database provider's connection details:

`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`.

For a Railway database, Vercel requires its external TCP proxy host/port rather
than a `railway.internal` address. Review the provider's encryption and
connection settings before production use. Database setup and provider costs
are not verified or approved by this document.

`serve_api.py` initializes the existing schema at startup. If the database is
unavailable it logs an error and keeps serving, so scanning still works while
account routes answer 503. (It used to exit, which made every API route fail with
FUNCTION_INVOCATION_FAILED.) Existing tables and records are preserved.
`/api/health/ready` returns 200 only when the API can reach the database.

## Review and verification

The working tree includes uncommitted authentication, scanning, assessment,
and history changes. Review the exact source snapshot before publishing. Do
not deploy the old remote default branch and assume it includes local changes.

Before this configuration was added, the current Flutter release build with
`API_BASE_URL=/api` succeeded, and all three existing Flutter tests passed.
The previous backend suite passed 25 tests. These checks do not establish
cloud readiness. Docker is not installed locally; the container build and
Vercel service routing still need a real deployment check.

The new vercel.json passed validation against the current official Vercel
schema constraints, and build_web_vercel.sh passed Bash syntax checking.
The official CPU-wheel index lists both pinned PyTorch packages for Python
3.12 on Linux x86_64. Ignore-rule checks verified that environment files,
virtual environments, agent files, Git metadata, and generated output are
excluded while the required source, model, and build inputs remain included.

After deployment, test signup, login, authenticated profile access, logout,
and garment analysis through the actual browser deployment. Verify history
remains scoped to the signed-in account. Do not report login as working until
the real hosted database and these flows have been tested.

Vercel Functions limit request bodies to 4.5 MB. The existing scanner accepts
larger combined photos locally; cloud photo uploads must be kept below this
limit or changed to an upload flow that avoids sending large files through a
Function. Web scanning also uses file-path APIs that need separate browser
verification; successful authentication tests do not establish scanner support.

References:

- https://vercel.com/docs/services
- https://vercel.com/docs/services/config-reference
- https://vercel.com/docs/functions/container-images
- https://docs.railway.com/databases/mysql
