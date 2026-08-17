# Credential handling

Google OAuth client credentials and generated user tokens are local runtime inputs. They must not be committed to this repository.

## Setup

1. Create or select the required Google Cloud OAuth client.
2. Copy `data_downloading/credentials.example.json` to a private location outside the repository and replace the placeholders.
3. Pass that path with the relevant script's `--credentials` option.
4. Let the authentication flow write its generated token outside the repository and pass it with `--token`.

The legacy default paths `data_downloading/credentials.json` and `data_downloading/token.json` are ignored for backward compatibility, but an external credential directory is preferred.

## Existing Git history

Earlier commits contained non-empty OAuth client and token files. Removing them from the current tree does not remove them from Git history. Treat those values as exposed: revoke or rotate the OAuth client secret, invalidate existing refresh tokens, and complete any history rewrite as a separate reviewed operation.

Never paste credential values into issues, logs, test fixtures, or generated quality reports.
