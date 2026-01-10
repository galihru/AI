"""Test ArduScratch AI via API."""
import requests
import json

# Test server
url = "http://127.0.0.1:8000/generate"

# Test request
test_spec = {
    "project": "LedBlinker",
    "spec": "Create Arduino LED blink code using pin 13, blink every 1 second",
    "temperature": 0.7,
    "max_tokens": 512
}

print("Testing ArduScratch AI API...")
print(f"Sending request: {test_spec['project']}")

try:
    response = requests.post(url, json=test_spec, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ Success!")
        print("\n--- Generated Code ---")
        print(result['code'][:500])
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n✗ Server not running!")
    print("Start server with: python serve.py")
except Exception as e:
    print(f"\n✗ Error: {e}")
