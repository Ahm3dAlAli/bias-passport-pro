#!/usr/bin/env python
"""
Fingerprint² - Ethical AI Assessment Framework for Vision-Language Models

Main entry point for running evaluations.

Usage:
    python main.py evaluate gpt-4o
    python main.py compare gpt-4o claude-3-opus
    python main.py --help
"""

import sys
from fingerprint_squared.cli import app


if __name__ == "__main__":
    app()
