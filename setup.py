from setuptools import setup, find_packages
import os

# Verificació de la presència del certificat abans de la instal·lació
def verify_integrity_token():
    cert_file = "IMPLEMENTATION_OF_CERTIFICATE_SVG.py"
    if not os.path.exists(cert_file):
        print(f"CRITICAL WARNING: {cert_file} not found. System integrity at risk.")
        return False
    return True

setup(
    name="svgelona_ai",
    version="5.2.0",
    author="SVGelona_AI Team",
    author_email="dev@svgelona.ai",
    description="Fractal Generative AI with SVD Stabilization and Synesthetic Core",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/svgelona-ai",
    
    # Automatització de la descoberta de subpaquets (core, kernel, optimization, tools)
    packages=find_packages(include=['core', 'core.*', 'kernel', 'optimization', 'tools']),
    
    # Inclusió de fitxers no-Python (com el White Paper i el Certificat)
    include_package_data=True,
    py_modules=[
        "main_v5_2", 
        "run_optimized", 
        "config_v5_2", 
        "init_root", 
        "IMPLEMENTATION_OF_CERTIFICATE_SVG",
        "conversation_interface"
    ],
    
    # Dependències estrictes per mantenir l'estabilitat del Kernel
    install_requires=[
        "numpy>=1.21.0",      # Operacions tensorials i SVD
        "scipy>=1.7.0",       # Regularització de Tikhonov i Van der Pol
        "sympy>=1.9",         # Solucionador simbòlic de fractals
        "matplotlib>=3.4.0",  # Visualització de defectes angulars
        "psutil>=5.8.0",      # Monitorització de recursos en profunditat 12
    ],
    
    # Entry points per a execució directa des de la terminal
    entry_points={
        "console_scripts": [
            "svgelona-run=run_optimized:main",
            "svgelona-chat=conversation_interface:main",
            "svgelona-sync=tools.SVGelonaTuner:main",
        ],
    },
    
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

if __name__ == "__main__":
    verify_integrity_token()