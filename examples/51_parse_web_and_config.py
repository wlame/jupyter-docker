#!/usr/bin/env python3
# # Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates
#
# Before analysis comes extraction. This notebook pulls a table out of an HTML
# page with BeautifulSoup, reads an XML feed with lxml and xmltodict, loads
# settings from YAML and a `.env` file, and normalizes timestamps written in
# different time zones with pendulum. The result is one clean JSON file.
#
# The samples are inline, so it runs offline. To parse a live page, fetch it with
# `httpx.get(url).text` and pass that to `BeautifulSoup` instead.
#
# - [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
# - [lxml](https://lxml.de/)
# - [xmltodict](https://github.com/martinblech/xmltodict)
# - [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
# - [python-dotenv](https://saurabh-kumar.com/python-dotenv/)
# - [pendulum](https://pendulum.eustace.io/docs/)

import os

import orjson
import pendulum
import xmltodict
import yaml
from bs4 import BeautifulSoup
from dotenv import dotenv_values
from lxml import etree

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ## Settings from YAML and a .env file
#
# Non-secret settings live in YAML; secrets and machine-specific values live in a
# `.env` file that is never committed. `dotenv_values` reads it without changing
# the process environment, and real environment variables override both.

CONFIG_YAML = """
report:
  title: Conference schedule
  output_timezone: Europe/Berlin
  currency: EUR
sources:
  schedule_page: https://example.org/schedule
  sponsor_feed: https://example.org/sponsors.xml
"""
env_path = os.path.join(OUTPUT_DIR, 'example.env')
with open(env_path, 'w') as f:
    f.write('API_TOKEN=change-me\nREPORT_TIMEZONE=America/New_York\n')

config = yaml.safe_load(CONFIG_YAML)
secrets = {**dotenv_values(env_path), **{k: v for k, v in os.environ.items() if k == 'REPORT_TIMEZONE'}}
output_tz = secrets.get('REPORT_TIMEZONE') or config['report']['output_timezone']
print(f"Report: {config['report']['title']}; times shown in {output_tz}; token set: {bool(secrets.get('API_TOKEN'))}")

# ## A table from an HTML page
#
# CSS selectors find the rows; each cell is cleaned of whitespace. Real pages are
# messier, so the parser tolerates missing cells instead of failing.

SCHEDULE_HTML = """
<html><body>
  <h1>Schedule</h1>
  <table id="talks">
    <tr><th>Talk</th><th>Speaker</th><th>Starts</th><th>Room</th></tr>
    <tr><td>Polars at scale</td><td>A. Ito</td><td>2026-11-20 09:30 Asia/Tokyo</td><td>Hall A</td></tr>
    <tr><td> Bayesian A/B tests </td><td>R. Okafor</td><td>2026-11-20 10:00 Europe/London</td><td>Hall B</td></tr>
    <tr><td>Notebooks in production</td><td>M. Silva</td><td>2026-11-20 08:15 America/Sao_Paulo</td><td></td></tr>
    <tr><td>JAX for scientists</td><td>L. Novak</td><td>2026-11-20 16:45 Europe/Prague</td><td>Hall A</td></tr>
  </table>
</body></html>
"""
soup = BeautifulSoup(SCHEDULE_HTML, 'lxml')
header = [th.get_text(strip=True).lower() for th in soup.select('#talks tr th')]
talks = []
for row in soup.select('#talks tr')[1:]:
    cells = [td.get_text(strip=True) for td in row.find_all('td')]
    talks.append(dict(zip(header, cells)) | {'room': cells[3] or 'to be announced'})
print(f"Parsed {len(talks)} talks: {[t['talk'] for t in talks]}")

# ## Times from different zones, in one zone
#
# Each start time carries its own time zone. pendulum parses it, keeps the zone,
# and converts to the report's zone, handling daylight-saving rules.

for talk in talks:
    date_part, time_part, zone = talk['starts'].rsplit(' ', 2)
    local = pendulum.parse(f'{date_part}T{time_part}', tz=zone)
    shown = local.in_timezone(output_tz)
    talk['starts_utc'] = local.in_timezone('UTC').to_iso8601_string()
    talk['starts_report'] = shown.format('ddd D MMM HH:mm zz')
    print(f"  {talk['talk']:<26} {talk['starts']:<32} -> {talk['starts_report']}")
talks.sort(key=lambda t: t['starts_utc'])

# ## Sponsors from an XML feed
#
# lxml with XPath is precise and fast for known structures; xmltodict turns a
# small document into plain dictionaries when you want to explore it.

SPONSOR_XML = b"""<?xml version="1.0"?>
<sponsors updated="2026-11-14">
  <sponsor tier="gold"><name>Arrow Data</name><amount currency="EUR">20000</amount></sponsor>
  <sponsor tier="silver"><name>Kernel Labs</name><amount currency="EUR">8000</amount></sponsor>
  <sponsor tier="silver"><name>Pixel Forge</name><amount currency="USD">9000</amount></sponsor>
</sponsors>"""
root = etree.fromstring(SPONSOR_XML)
gold = root.xpath('//sponsor[@tier="gold"]/name/text()')
eur_total = sum(float(a.text) for a in root.xpath('//amount[@currency="EUR"]'))
as_dict = xmltodict.parse(SPONSOR_XML)
sponsors = [{'name': s['name'], 'tier': s['@tier'], 'amount': float(s['amount']['#text']),
             'currency': s['amount']['@currency']} for s in as_dict['sponsors']['sponsor']]
print(f"Gold sponsors: {gold}; EUR pledged: {eur_total:,.0f}; feed updated {root.get('updated')}")

# ## One clean JSON document
#
# orjson serializes quickly and handles dates and numbers without extra code.

report = {'title': config['report']['title'], 'timezone': output_tz, 'talks': talks, 'sponsors': sponsors}
out_path = os.path.join(OUTPUT_DIR, 'parsed_report.json')
with open(out_path, 'wb') as f:
    f.write(orjson.dumps(report, option=orjson.OPT_INDENT_2))
print(f"Saved: parsed_report.json ({os.path.getsize(out_path)} bytes)")
