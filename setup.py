from setuptools import setup, find_namespace_packages
import platform

dependency_links = []
if platform.system() == "Windows":
    dependency_links.append("https://download.pytorch.org/whl/torch_stable.html")

def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n") if ln.strip() and not ln.startswith("#")]

setup(
    name="vlm-toolbox",
    version="0.1.0",
    author="deepmancer",
    author_email="alirezaheidari.cs@gmail.com",
    description="Vision-Language Models Toolbox: Your all-in-one solution for multimodal research and experimentation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Vision-Language, Multimodal, Soft Prompt Learning, Deep Learning, Pytorch, Zero Shot Classification, Contrastive Learning",
    license="3-Clause BSD",
    packages=find_namespace_packages(include=["vlm_toolbox.*"]),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.7",
    include_package_data=True,
    dependency_links=dependency_links,
    zip_safe=False,
)
