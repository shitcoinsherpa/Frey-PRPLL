#!/usr/bin/python3

# Copyright (c) Mihai Preda.
# Updated to use V5 API for result submission (manual_result form now requires checksum).

import argparse
import json as jsonlib
import time
import urllib
import requests
import os
import upload
import getpass
import random

from collections import OrderedDict
from http import cookiejar
from urllib.parse import urlencode
from urllib.request import build_opener, urlopen
from urllib.request import HTTPCookieProcessor
from datetime import datetime

baseUrl = "https://www.mersenne.org/"
primenet = build_opener(HTTPCookieProcessor(cookiejar.CookieJar()))

# V5 API for result submission
V5_URL = "http://v5.mersenne.org/v5server/?"
V5_BASEARGS = OrderedDict([("px", "GIMPS"), ("v", "0.95")])
V5_SECURITY = "&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD"

# PrimeNet API constants
PRIMENET_AR_PRP_RESULT = 150
PRIMENET_AR_PRP_PRIME = 151
PRIMENET_AR_LL_RESULT = 100
PRIMENET_AR_LL_PRIME = 101
ERROR_OK = 0


def v5_request(params):
    """Send request to V5 API and return parsed response."""
    merged = OrderedDict(V5_BASEARGS)
    merged.update(params)
    url = V5_URL + urlencode(merged) + V5_SECURITY
    resp = urlopen(url, timeout=60).read().decode("utf-8")
    result = OrderedDict()
    for line in resp.splitlines():
        if line == "==END==":
            break
        if "=" in line:
            key, _, val = line.partition("=")
            result[key] = val
    return result


def get_or_create_guid(guidFile, user):
    """Load GUID from file or register a new computer with PrimeNet."""
    if os.path.exists(guidFile):
        guid = open(guidFile).read().strip()
        if len(guid) == 32:
            return guid

    guid = "%032x" % random.getrandbits(128)
    print(f"Registering new computer with PrimeNet (GUID: {guid})...")

    params = OrderedDict()
    params["t"] = "uc"
    params["g"] = guid
    params["a"] = "Linux64,frey-prpll,v0.1"
    params["wg"] = ""
    params["hd"] = "%032x" % random.getrandbits(128)
    params["c"] = "NVIDIA RTX 4090"
    params["f"] = "CUDA"
    params["L1"] = 64
    params["L2"] = 65536
    params["np"] = 1
    params["hp"] = 0
    params["m"] = 24576
    params["s"] = 2520
    params["h"] = 24
    params["r"] = 1000
    params["u"] = user
    params["cn"] = "RTX4090-WSL2"

    result = v5_request(params)
    error = int(result.get("pnErrorResult", -1))
    if error != ERROR_OK:
        print(f"Registration warning: Error {error}: {result.get('pnErrorDetail', 'unknown')}")
    else:
        print(f"Registered as {result.get('u', user)} on {result.get('cn', 'unknown')}")

    os.makedirs(os.path.dirname(guidFile), exist_ok=True)
    with open(guidFile, "w") as f:
        f.write(guid)
    return guid


def login(user, password):
    login = {"user_login": user, "user_password": password}
    data = urlencode(login).encode('utf-8')
    r = primenet.open(baseUrl, data).read().decode("utf-8")
    if not user + "<br>logged in" in r:
        print(r)
        print("Login failed");
        raise(PermissionError("Login failed"))

def loadLines(fileName):
    try:
        with open(fileName, 'r') as fi:
            return set((line.strip().strip('\n') for line in fi))
    except FileNotFoundError as e:
        return set()


def sendOneV5(line, guid):
    """Submit a result via V5 API."""
    print("Sending result via V5 API: ", line[:100], "...")
    try:
        ar = jsonlib.loads(line)
    except jsonlib.JSONDecodeError:
        print("  Not valid JSON, skipping")
        return False

    worktype = ar.get("worktype", "")

    # Determine result type code
    if worktype.startswith("PRP"):
        result_type = PRIMENET_AR_PRP_PRIME if ar.get("status") == "P" else PRIMENET_AR_PRP_RESULT
    elif worktype == "LL":
        result_type = PRIMENET_AR_LL_PRIME if ar.get("status") == "P" else PRIMENET_AR_LL_RESULT
    else:
        print(f"  Unknown worktype '{worktype}', trying as PRP result")
        result_type = PRIMENET_AR_PRP_RESULT

    params = OrderedDict()
    params["t"] = "ar"
    params["g"] = guid
    params["k"] = ar.get("aid", "")
    params["m"] = line.strip()
    params["r"] = result_type
    params["d"] = 1
    params["n"] = ar.get("exponent", 0)

    if ar.get("status") != "P" and "res64" in ar:
        params["rd"] = ar["res64"]

    # PRP base
    if "-" in worktype:
        try:
            params["base"] = int(worktype.split("-")[1])
        except ValueError:
            pass

    if "fft-length" in ar:
        params["fftlen"] = ar["fft-length"]
    if "residue-type" in ar:
        params["rt"] = ar["residue-type"]
    if "error-code" in ar:
        params["ec"] = ar["error-code"]
    else:
        params["ec"] = "00000000"
    if "shift-count" in ar:
        params["sc"] = ar["shift-count"]
    else:
        params["sc"] = 0

    errors = ar.get("errors", {})
    if isinstance(errors, dict) and "gerbicz" in errors:
        params["gbz"] = 1

    # Proof metadata (without pp/ph, PrimeNet won't know a proof exists)
    proof = ar.get("proof", {})
    if proof and proof.get("power", 0) > 0:
        params["pp"] = proof["power"]
        params["ph"] = proof.get("md5", "")

    result = v5_request(params)
    error = int(result.get("pnErrorResult", -1))
    detail = result.get("pnErrorDetail", "")

    if error == ERROR_OK:
        print(f"  {detail}")
        return True
    elif "already" in detail.lower():
        print(f"  Already submitted: {detail}")
        return True  # Don't retry
    else:
        print(f"  Error {error}: {detail}")
        return False


def appendLine(fileName, line):
    with open(fileName, 'a') as fo: print(line, file = fo, end = '\n')

def sendResults(results, sent, sentName, retryName, guid):
    for result in results:
        ok = sendOneV5(result, guid)
        sent.add(result)
        appendLine(sentName if ok else retryName, result)

def fetch(what):
    assignment = {"cores":1, "num_to_get":1, "pref":what}
    res = primenet.open(baseUrl + "manual_assignment/", data=urlencode(assignment).encode()).read().decode("utf-8")

    begin = res.find(">PRP=")
    if begin == -1: begin = res.find(">LL=")
    if begin == -1:
        print(res)
        raise(AssertionError("assignment no BEGIN mark"))
    begin += 1
    end   = res.find("</", begin)
    if end == -1: raise(AssertionError("assignemnt no END mark"))
    line = res[begin:end].strip().strip('\n')
    print(datetime.now(), " New assignment: ", line)
    return line

# LL_DC was 101; here we use 106 instead in order to get only LL_DC with shift != 0 which are the ones
# that can be double-checked with a zero shift.
workTypes = dict(PRP=150, PM1=4, LL_DC=106, PRP_DC=151, PRP_WORLD_RECORD=152, PRP_100M=153, PRP_P1=154)

parser = argparse.ArgumentParser()
parser.add_argument('-u', dest='username', default='', help="Primenet user name")
parser.add_argument('-p', dest='password', help="Primenet password")
parser.add_argument('-t', dest='timeout',  type=int, default=1800, help="Seconds to sleep between updates")
parser.add_argument('--dirs', metavar='DIR', nargs='+', help="GpuOwl directories to scan", default=".")
parser.add_argument('--tasks', dest='nTasks', type=int, default=None, help='Number of tasks to fetch ahead')

choices=list(workTypes.keys())
parser.add_argument('-w', dest='work', choices=choices, help="GIMPS work type", default="PRP")

options = parser.parse_args()
timeout = int(options.timeout)
user = options.username

worktype = workTypes[options.work] if options.work in workTypes else int(options.work)
print("Work type:", worktype)

desiredTasks = options.nTasks if options.nTasks is not None else (12 if worktype == 4 else 2)
print("Will fetch ahead %d tasks. Check every %d sec." % (desiredTasks, timeout))

if not user:
    print("-u USER is required")
    exit(1)

print("User: %s" % user)

dirs = [(d if d[-1] == '/' else d + '/' ) for d in options.dirs if d]

print("Watched dirs: ", ' '.join(dirs))

password = options.password
if not password:
    password = getpass.getpass("Primenet password")

# Set up V5 API GUID (stored alongside the watched directory)
guidFile = os.path.join(dirs[0], "primenet_guid.txt")
guid = get_or_create_guid(guidFile, user)
print(f"V5 GUID: {guid}")

# Initial early login for assignment fetching (V5 API doesn't need login for results)
login(user, password)

def handle(folder):
    sent = loadLines(folder + "sent.txt")
    (resultsName, worktodoName, sentName, retryName) = (folder + name + ".txt" for name in "results worktodo sent retry".split())

    newResults = loadLines(resultsName) - sent
    if newResults: print(datetime.now(), " found %d new result(s) in %s" % (len(newResults), resultsName))

    tasks = [line for line in loadLines(worktodoName) if line and line[0] != '#']
    needFetch = len(tasks) < desiredTasks
    if needFetch: print(datetime.now(), " found only %d task(s) in %s, want %d" % (len(tasks), worktodoName, desiredTasks));

    if newResults:
        sendResults(newResults, sent, sentName, retryName, guid)

    if needFetch:
        login(user, password)
        for _ in range(len(tasks), desiredTasks):
            appendLine(worktodoName, fetch(worktype))

    try:
        os.mkdir(folder + 'uploaded')
    except FileExistsError:
        pass

    for entry in os.listdir(folder + 'proof'):
        if entry.endswith('.proof'):
            fileName = folder + 'proof/' + entry
            if upload.uploadProof(user, fileName):
                os.rename(fileName, folder + 'uploaded/' + entry)

while True:
    for folder in dirs:
        try:
            handle(folder)
        except urllib.error.URLError as e:
            print(e)
        except requests.exceptions.ConnectionError as e:
            print(e)
        except requests.exceptions.RequestException as e:
            print(e)

    if timeout == 0:
        break

    time.sleep(timeout)
