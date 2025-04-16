from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="netro",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    entry_points={
        "console_scripts": [
            "netro=netro.main:main",  # Assuming your main.py has a main() function
        ],
    },
)
