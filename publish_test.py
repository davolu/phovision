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

def update_version(current_version):
    """Update version in setup.py by adding 'rcX' suffix for test versions."""
    if not current_version:
        return None
    
    # Check if version already has an rc suffix
    rc_match = re.search(r'rc(\d+)$', current_version)
    if rc_match:
        # Increment rc number
        rc_num = int(rc_match.group(1))
        new_version = re.sub(r'rc\d+$', f'rc{rc_num + 1}', current_version)
    else:
        # Add rc1 suffix
        new_version = f"{current_version}rc1"
    
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

def upload_to_test_pypi():
    """Upload the package to TestPyPI using twine."""
    print("📦 Uploading to TestPyPI...")
    result = subprocess.run(
        ['python3', '-m', 'twine', 'upload', '--repository-url', 'https://test.pypi.org/legacy/', 'dist/*'],
        capture_output=True,
        text=True
    )
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

    # Get current version
    current_version = get_current_version()
    if not current_version:
        print("❌ Error: Couldn't find version in setup.py")
        sys.exit(1)

    print(f"📝 Current version: {current_version}")

    # Update version with rc suffix
    new_version = update_version(current_version)
    print(f"📝 Test version: {new_version}")

    # Clean old builds
    clean_old_builds()

    # Build package
    build_package()

    # Upload to TestPyPI
    upload_to_test_pypi()

    print(f"🎉 Successfully published version {new_version} to TestPyPI!")
    print("\nTo install this test version, use:")
    print(f"pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ phovision=={new_version}")
    print("\nNote: The --extra-index-url is needed because TestPyPI doesn't have all dependencies.")
    
    # Remind about version cleanup
    print("\n⚠️  Remember: This script added an 'rcX' suffix to your version.")
    print("   You might want to revert setup.py if you don't want to keep this version number.")

if __name__ == '__main__':
    main() 