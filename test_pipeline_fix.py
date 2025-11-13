#!/usr/bin/env python3
"""
Test script to verify the Pipeline loader function works correctly.
Run this to test outside of Jupyter.
"""

# This simulates what should be in the notebook
class MockDataset:
    CODE = "refe-fundamentals"

class MockColumn:
    def __init__(self):
        self.dataset = MockDataset()

# The CORRECT function with CODE check
def get_pipeline_loader_correct(column):
    """This is the CORRECT version"""
    if hasattr(column.dataset, 'CODE') and column.dataset.CODE == "refe-fundamentals":
        print("✅ CODE check PASSED - loader would be returned")
        return "CustomSQLiteLoader"
    print("❌ CODE check FAILED - no loader found")
    raise ValueError(f"No loader for {column}")

# The OLD function without CODE check
def get_pipeline_loader_old(column):
    """This is the OLD version"""
    if column.dataset == MockDataset:  # This will fail
        print("✅ Direct comparison PASSED")
        return "CustomSQLiteLoader"
    print("❌ Direct comparison FAILED")
    raise ValueError(f"No loader for {column}")

# Test
print("Testing with mock column...")
test_col = MockColumn()

print("\n1. Testing CORRECT version (with CODE check):")
try:
    result = get_pipeline_loader_correct(test_col)
    print(f"   Result: {result}")
except ValueError as e:
    print(f"   Error: {e}")

print("\n2. Testing OLD version (without CODE check):")
try:
    result = get_pipeline_loader_old(test_col)
    print(f"   Result: {result}")
except ValueError as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("CONCLUSION:")
print("If you see '✅ CODE check PASSED' above, the new version works!")
print("The old version should fail with '❌ Direct comparison FAILED'")
