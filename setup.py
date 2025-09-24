from setuptools import find_packages, setup

setup(
    name="IDIR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Add dependencies here
    author="Kenton Kwok",
    author_email="kwokkenton@gmail.com",
    # description="A short description of your package",
    # url="https://github.com/yourusername/your_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

print(find_packages())