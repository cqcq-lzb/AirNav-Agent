@echo off
setlocal
cd /d "%~dp0"

docker compose --env-file .env.api -f docker-compose.api.yml down
echo API backend stopped. Job data remains in api_data.
pause
