#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m py_compile rrem_repaired.py evaluate_original_readonly.py
python tests/audit_original.py | tee results/original_audit.stdout.txt
python tests/test_repaired.py | tee results/repaired_tests.stdout.txt
python tests/test_interfaces.py | tee results/interface_tests.stdout.txt
python tests/diagnose_full_credit.py | tee results/full_credit_diagnostic.stdout.txt
python tests/test_legacy_control.py | tee results/legacy_control_tests.stdout.txt
