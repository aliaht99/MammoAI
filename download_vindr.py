import subprocess, os, re, getpass

print("Logging into PhysioNet...")

# Get CSRF token
r = subprocess.run(['curl', '-s', '-c', '/tmp/pn_cookies.txt',
    'https://physionet.org/login/'], capture_output=True, text=True)

m = re.search(r'csrfmiddlewaretoken.*?value=["\']([^"\']+)', r.stdout)
if not m:
    print("ERROR: Could not get CSRF token"); exit(1)
csrf = m.group(1)

pwd = getpass.getpass("PhysioNet password: ")

# Login
subprocess.run(['curl', '-s', '-b', '/tmp/pn_cookies.txt', '-c', '/tmp/pn_cookies.txt',
    '-X', 'POST', 'https://physionet.org/login/',
    '-d', f'username=alihamzaaht99&password={pwd}&csrfmiddlewaretoken={csrf}',
    '-H', 'Referer: https://physionet.org/login/'], capture_output=True)

print("Downloading CSVs...")
os.makedirs('vindr-mammo', exist_ok=True)

for f in ['breast-level_annotations.csv', 'finding_annotations.csv']:
    subprocess.run(['curl', '-s', '-L', '-b', '/tmp/pn_cookies.txt',
        '-o', f'vindr-mammo/{f}',
        f'https://physionet.org/files/vindr-mammo/1.0.0/{f}'])
    with open(f'vindr-mammo/{f}') as fh:
        first = fh.read(50)
    if 'DOCTYPE' in first:
        print(f"  FAILED: {f} — dataset access still not approved on PhysioNet")
    else:
        lines = open(f'vindr-mammo/{f}').readlines()
        print(f"  OK: {f} ({len(lines)} rows)")
        print(f"  Columns: {lines[0].strip()}")
