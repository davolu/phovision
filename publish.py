#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

def clean_old_builds():
    """Remove old build artifacts."""
    print("🧹 Cleaning old builds...")
    paths_to_remove = ['dist', 'build', '*.egg-info']
    for pattern in paths_to_remove:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    print("✅ Cleaned old builds")

def get_current_version():
    """Get the current version from setup.py."""
    with open('setup.py', 'r') as f:
        content = f.read()
    match = re.search(r'version="([^"]+)"', content)
    return match.group(1) if match else None

def update_version(current_version, bump_type='patch'):
    """Update version in setup.py."""
    if not current_version:
        return None
    
    # Split version into major, minor, patch
    major, minor, patch = map(int, current_version.split('.'))
    
    # Update version based on bump type
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    else:  # patch
        patch += 1
    
    new_version = f"{major}.{minor}.{patch}"
    
    # Update setup.py
    with open('setup.py', 'r') as f:
        content = f.read()
    
    new_content = re.sub(
        r'version="[^"]+"',
        f'version="{new_version}"',
        content
    )
    
    with open('setup.py', 'w') as f:
        f.write(new_content)
    
    return new_version

def build_package():
    """Build the package using build."""
    print("🔨 Building package...")
    result = subprocess.run(['python3', '-m', 'build'], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Build failed:")
        print(result.stderr)
        sys.exit(1)
    print("✅ Build successful")

def upload_to_pypi():
    """Upload the package to PyPI using twine."""
    print("📦 Uploading to PyPI...")
    result = subprocess.run(['python3', '-m', 'twine', 'upload', 'dist/*'], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Upload failed:")
        print(result.stderr)
        sys.exit(1)
    print("✅ Upload successful")

def main():
    # Ensure we're in the correct directory
    if not os.path.exists('setup.py'):
        print("❌ Error: setup.py not found. Are you in the correct directory?")
        sys.exit(1)

    # Get bump type from command line argument
    bump_type = 'patch'  # default
    if len(sys.argv) > 1 and sys.argv[1] in ['major', 'minor', 'patch']:
        bump_type = sys.argv[1]

    # Get current version
    current_version = get_current_version()
    if not current_version:
        print("❌ Error: Couldn't find version in setup.py")
        sys.exit(1)

    print(f"📝 Current version: {current_version}")

    # Update version
    new_version = update_version(current_version, bump_type)
    print(f"📝 New version: {new_version}")

    # Clean old builds
    clean_old_builds()

    # Build package
    build_package()

    # Upload to PyPI
    upload_to_pypi()

    print(f"🎉 Successfully published version {new_version} to PyPI!")
    print(f"📦 Users can now install it with: pip install phovision=={new_version}")

if __name__ == '__main__':
    main() 