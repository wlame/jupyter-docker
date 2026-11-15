#!/usr/bin/env python3
# # Calling an API Reliably: Retries, Validation, and Logging
#
# Real APIs time out, rate-limit, and sometimes return malformed records. This
# notebook builds a small, sturdy client: httpx for HTTP with timeouts, tenacity
# to retry transient failures with exponential backoff, Pydantic to validate
# every record (keeping the bad ones with their reasons), loguru for readable
# logs, and rich for a summary table.
#
# It runs offline against a simulated, deliberately unreliable weather API. Set
# `API_URL` to a real endpoint that returns the same shape to try it for real.
#
# - [httpx](https://www.python-httpx.org/)
# - [tenacity](https://tenacity.readthedocs.io/)
# - [Pydantic](https://docs.pydantic.dev/)
# - [loguru](https://loguru.readthedocs.io/)
# - [rich](https://rich.readthedocs.io/)

import json
import os
import random
import sys
from datetime import datetime

import httpx
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.table import Table
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_URL = None  # e.g. 'https://api.example.com' with GET /stations/{id}/readings
STATIONS = ['berlin', 'lisbon', 'oslo', 'rome', 'warsaw', 'madrid']
logger.remove()
logger.add(sys.stdout, format='{time:HH:mm:ss} | {level: <7} | {message}', level='INFO')

# ## A simulated unreliable API
#
# httpx can route requests to a Python function instead of the network. This one
# follows a fixed failure plan so every run shows the same story: some stations
# answer with 503 or 429 errors a few times before succeeding, one never recovers,
# and some records come back with a missing or impossible value.

FAILURE_PLAN = {'berlin': [503, 503], 'lisbon': [429], 'oslo': [503] * 10, 'warsaw': [502]}
random.seed(7)
served: dict[str, int] = {}


def flaky_weather_api(request: httpx.Request) -> httpx.Response:
    """Answer GET /stations/<id>/readings, failing per FAILURE_PLAN and corrupting some records."""
    station = request.url.path.split('/')[2]
    call = served.get(station, 0)
    served[station] = call + 1
    plan = FAILURE_PLAN.get(station, [])
    if call < len(plan):
        return httpx.Response(plan[call], json={'error': 'temporarily unavailable'})
    readings = [
        {'station': station, 'time': f'2026-11-15T{hour:02d}:00:00Z',
         'temperature_c': round(random.gauss(9, 4), 1), 'humidity_pct': random.randint(40, 95)}
        for hour in range(6, 18, 3)
    ]
    if station in ('berlin', 'rome'):
        readings[1]['humidity_pct'] = 140  # impossible value
    if station == 'madrid':
        del readings[2]['temperature_c']  # missing field
    return httpx.Response(200, json=readings)


transport = None if API_URL else httpx.MockTransport(flaky_weather_api)
client = httpx.Client(base_url=API_URL or 'https://weather.invalid', transport=transport,
                      timeout=httpx.Timeout(5.0), headers={'User-Agent': 'jupyter-docker-example'})

# ## Retry only what is worth retrying
#
# Timeouts, connection errors, 429, and 5xx responses are usually temporary; a 404
# or 400 is not, and retrying it only wastes time. The backoff doubles the wait
# after each failure (shortened here so the notebook runs quickly).

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_transient(error: BaseException) -> bool:
    """True for failures that may succeed on a later attempt."""
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code in RETRYABLE_STATUS


attempts: dict[str, int] = {}


def log_retry(state: RetryCallState) -> None:
    """One short log line per retry: which station, what failed, how long we wait."""
    error = state.outcome.exception()
    reason = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
    logger.warning(f"{state.args[0]}: attempt {state.attempt_number} failed ({reason}), "
                   f"retrying in {state.next_action.sleep:.2f} s")



@retry(
    retry=retry_if_exception(is_transient),
    wait=wait_exponential(multiplier=0.05, max=1.0),
    stop=stop_after_attempt(6),
    before_sleep=log_retry,
    reraise=True,
)
def fetch_readings(station: str) -> list[dict]:
    """GET one station's readings, raising for any non-2xx response."""
    attempts[station] = attempts.get(station, 0) + 1
    response = client.get(f'/stations/{station}/readings')
    response.raise_for_status()
    return response.json()


# ## Validate every record
#
# The model says what a usable reading is. Records that break it are kept with
# the reason, so bad data is visible instead of silently dropped.


class Reading(BaseModel):
    """One weather reading."""

    station: str
    time: datetime
    temperature_c: float = Field(ge=-60, le=60)
    humidity_pct: int = Field(ge=0, le=100)


valid: list[Reading] = []
rejected: list[dict] = []
for station in STATIONS:
    try:
        raw = fetch_readings(station)
    except httpx.HTTPError as error:
        status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
        logger.error(f"{station}: gave up after {attempts[station]} attempts (last error {status})")
        continue
    logger.info(f"{station}: {len(raw)} records after {attempts[station]} attempt(s)")
    for record in raw:
        try:
            valid.append(Reading.model_validate(record))
        except ValidationError as error:
            reasons = '; '.join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
            rejected.append({'record': record, 'reason': reasons})
            logger.warning(f"{station}: rejected one record ({reasons})")

# ## Summary

table = Table(title='Readings by station')
for column in ('station', 'attempts', 'valid', 'mean °C', 'max humidity %'):
    table.add_column(column, justify='right' if column != 'station' else 'left')
for station in STATIONS:
    rows = [r for r in valid if r.station == station]
    mean = f"{sum(r.temperature_c for r in rows) / len(rows):.1f}" if rows else '—'
    table.add_row(station, str(attempts.get(station, 0)), str(len(rows)), mean,
                  str(max((r.humidity_pct for r in rows), default='—')))
Console(width=90).print(table)
print(f"Valid readings: {len(valid)}, rejected: {len(rejected)}, "
      f"total requests: {sum(attempts.values())} for {len(STATIONS)} stations")

with open(os.path.join(OUTPUT_DIR, 'api_readings.json'), 'w') as f:
    json.dump({'valid': [r.model_dump(mode='json') for r in valid], 'rejected': rejected,
               'attempts': attempts}, f, indent=2)
print("Saved: api_readings.json")
