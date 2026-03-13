#!/usr/bin/python3
"""Submit PRP result to PrimeNet via V5 API (no checksum needed)."""
import json
import os
import random
import sys
from collections import OrderedDict
from urllib.parse import urlencode
from urllib.request import urlopen, Request

V5_URL = "http://v5.mersenne.org/v5server/?"
V5_BASEARGS = OrderedDict([("px", "GIMPS"), ("v", "0.95")])
V5_SECURITY = "&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD"
GUID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "primenet_guid.txt")

# PrimeNet API constants
PRIMENET_AR_PRP_RESULT = 150
PRIMENET_AR_PRP_PRIME = 151
ERROR_OK = 0


def parse_v5_resp(text):
    """Parse V5 API response into dict."""
    result = OrderedDict()
    for line in text.splitlines():
        if line == "==END==":
            break
        if "=" in line:
            key, _, val = line.partition("=")
            result[key] = val
    return result


def v5_request(params):
    """GET from V5 API and return parsed response."""
    merged = OrderedDict(V5_BASEARGS)
    merged.update(params)
    url = V5_URL + urlencode(merged) + V5_SECURITY
    print(f"  URL: {url[:200]}...")
    resp = urlopen(url, timeout=60).read().decode("utf-8")
    print(f"  Raw response: {resp.strip()}")
    return parse_v5_resp(resp)


def get_or_create_guid():
    """Load GUID from file or create and register a new one."""
    if os.path.exists(GUID_FILE):
        guid = open(GUID_FILE).read().strip()
        if len(guid) == 32:
            print(f"Using existing GUID: {guid}")
            return guid

    # Generate new GUID
    guid = "%032x" % random.getrandbits(128)
    print(f"Generated new GUID: {guid}")

    # Register with PrimeNet
    print("Registering computer with PrimeNet...")
    params = OrderedDict()
    params["t"] = "uc"  # update computer
    params["g"] = guid
    params["a"] = "Linux64,frey-prpll,v0.1"
    params["wg"] = ""  # worktypes
    params["hd"] = "%032x" % random.getrandbits(128)  # hardware ID
    params["c"] = "NVIDIA RTX 4090"
    params["f"] = "CUDA"
    params["L1"] = 64
    params["L2"] = 65536
    params["np"] = 1
    params["hp"] = 0
    params["m"] = 24576  # 24GB VRAM
    params["s"] = 2520  # boost clock MHz
    params["h"] = 24
    params["r"] = 1000
    params["u"] = "Frey-Sherpa"
    params["cn"] = "RTX4090-WSL2"

    result = v5_request(params)
    error = int(result.get("pnErrorResult", -1))
    if error != ERROR_OK:
        print(f"Registration failed! Error {error}: {result.get('pnErrorDetail', 'unknown')}")
        # Try to continue anyway — maybe we can submit without registration
    else:
        print("Registration successful!")

    # Save GUID
    os.makedirs(os.path.dirname(GUID_FILE), exist_ok=True)
    with open(GUID_FILE, "w") as f:
        f.write(guid)
    print(f"GUID saved to {GUID_FILE}")
    return guid


def submit_result(guid, result_json):
    """Submit a PRP result via V5 API."""
    ar = json.loads(result_json)

    # Determine result type
    if ar.get("status") == "P":
        result_type = PRIMENET_AR_PRP_PRIME
    else:
        result_type = PRIMENET_AR_PRP_RESULT

    exponent = ar["exponent"]
    aid = ar.get("aid", "")

    # Extract PRP base from worktype (e.g., "PRP-3" -> 3)
    worktype = ar.get("worktype", "PRP-3")
    base = 3
    if "-" in worktype:
        try:
            base = int(worktype.split("-")[1])
        except ValueError:
            pass

    # Build V5 parameters
    params = OrderedDict()
    params["t"] = "ar"  # assignment result
    params["g"] = guid
    params["k"] = aid
    params["m"] = result_json.strip()
    params["r"] = result_type
    params["d"] = 1
    params["n"] = exponent

    # Residue (for composite results)
    if ar.get("status") != "P" and "res64" in ar:
        params["rd"] = ar["res64"]

    params["base"] = base

    if "fft-length" in ar:
        params["fftlen"] = ar["fft-length"]

    if "residue-type" in ar:
        params["rt"] = ar["residue-type"]

    # Error code (hex format, 0 = no error)
    params["ec"] = ar.get("error-code", "00000000")
    # Shift count
    params["sc"] = ar.get("shift-count", 0)

    errors = ar.get("errors", {})
    if isinstance(errors, dict) and "gerbicz" in errors:
        params["gbz"] = 1

    # Proof metadata (CRITICAL: without these, PrimeNet won't know a proof exists)
    proof = ar.get("proof", {})
    if proof and proof.get("power", 0) > 0:
        params["pp"] = proof["power"]
        params["ph"] = proof.get("md5", "")

    print(f"\nSubmitting PRP result for M{exponent}...")
    print(f"  AID: {aid}")
    print(f"  Status: {ar.get('status')}")
    print(f"  Res64: {ar.get('res64')}")
    print(f"  Result type: {result_type}")

    result = v5_request(params)
    error = int(result.get("pnErrorResult", -1))
    if error == ERROR_OK:
        print(f"\nSUCCESS! Result accepted by PrimeNet.")
        credit = result.get("pnErrorDetail", "")
        if credit:
            print(f"  Detail: {credit}")
        return True
    else:
        print(f"\nFAILED! Error {error}: {result.get('pnErrorDetail', 'unknown')}")
        return False


def main():
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run", "results.txt")
    if len(sys.argv) > 1:
        result_file = sys.argv[1]

    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        sys.exit(1)

    result_json = open(result_file).read().strip()
    print(f"Result to submit: {result_json[:100]}...")

    # Validate it's JSON
    try:
        json.loads(result_json)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in result file: {e}")
        sys.exit(1)

    guid = get_or_create_guid()
    success = submit_result(guid, result_json)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
