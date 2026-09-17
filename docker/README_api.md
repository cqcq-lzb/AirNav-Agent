# Airway Navigation API container

This separate portable backend container accepts a NIfTI CT, invokes the configured GPU-server stage-4 pipeline over SSH, persists job state and returns the generated `stage4_package.zip`. It contains no GUI, CT data, models, SSH key or server password. This is research software, not a clinically validated medical-device service.

1. Copy `.env.api.example` to `.env.api`; set a long random `API_TOKEN`, persistent `API_DATA_DIR_HOST`, and `SSH_DIR`. The SSH folder must contain `id_ed25519` and `known_hosts` for the GPU server. Do not commit it.
2. Review `server_config.json` for the remote deployment values.
3. Run `docker compose --env-file .env.api -f docker-compose.api.yml up -d --build`.

The default mapping is `127.0.0.1:8000`; it is not LAN-exposed. Use an authenticated HTTPS reverse proxy before network deployment, and never leave `API_TOKEN` empty.

## Copy to another computer

On the source computer, run `导出后端镜像.bat`. Copy the generated `airway-navigation-api-1.0.tar` together with this `docker` folder's API deployment files to a target computer with Docker Desktop. On that computer, create `.env.api` from `.env.api.example`, configure its own SSH key directory and API token, then run `导入后端镜像并启动.bat`. The image carries the backend and Python dependencies; the target computer supplies its own SSH key, persistent data directory, and GPU-server configuration.

Endpoints: `GET /health`; `POST /api/v1/jobs` multipart fields `ct`, optional `case_id`, `gpu`; `GET /api/v1/jobs/{id}`; `GET /api/v1/jobs/{id}/logs`; `GET /api/v1/jobs/{id}/package`. API calls require `X-API-Key` when a token is configured.
