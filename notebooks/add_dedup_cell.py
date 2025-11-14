#!/usr/bin/env python3
"""
Add deduplication cell to load_csv_fundamentals.ipynb
"""
import json
from pathlib import Path

notebook_path = Path('/Users/kamran/Documents/Code/Docker/zipline-reloaded/notebooks/load_csv_fundamentals.ipynb')

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with Date formatting
target_cell_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if "insert_data['Date'] = insert_data['Date'].dt.strftime('%Y-%m-%d')" in source:
            target_cell_index = i
            print(f"Found target cell at index {i}")
            break

if target_cell_index is None:
    print("Could not find the Date formatting cell!")
    exit(1)

# Create new deduplication cell
dedup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =============================================================================\n",
        "# DEDUPLICATE DATA - Fix for UNIQUE constraint failures\n",
        "# =============================================================================\n",
        "\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"DEDUPLICATING DATA\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "# Count before\n",
        "rows_before = len(insert_data)\n",
        "print(f\"Rows before deduplication: {rows_before:,}\")\n",
        "\n",
        "# Check for duplicates\n",
        "duplicates = insert_data[insert_data.duplicated(subset=['Sid', 'Date'], keep=False)]\n",
        "if len(duplicates) > 0:\n",
        "    print(f\"⚠️  Found {len(duplicates):,} rows with duplicate (Sid, Date) pairs\")\n",
        "    \n",
        "    # Show sample\n",
        "    dup_counts = duplicates.groupby(['Sid', 'Date']).size().reset_index(name='count')\n",
        "    dup_counts = dup_counts.sort_values('count', ascending=False).head(5)\n",
        "    print(\"\\nTop 5 most duplicated (Sid, Date) pairs:\")\n",
        "    print(dup_counts.to_string(index=False))\n",
        "\n",
        "# Deduplicate - keep last occurrence (most recent data)\n",
        "insert_data = insert_data.drop_duplicates(subset=['Sid', 'Date'], keep='last')\n",
        "\n",
        "# Count after\n",
        "rows_after = len(insert_data)\n",
        "duplicates_removed = rows_before - rows_after\n",
        "\n",
        "print(f\"\\nRows after deduplication: {rows_after:,}\")\n",
        "print(f\"Duplicates removed: {duplicates_removed:,}\")\n",
        "\n",
        "if duplicates_removed == 0:\n",
        "    print(\"✓ No duplicates found - data is clean!\")\n",
        "else:\n",
        "    print(f\"✓ Removed {duplicates_removed:,} duplicate records\")\n",
        "    print(\"  Strategy: Kept 'last' occurrence for each (Sid, Date) pair\")\n",
        "\n",
        "print(\"=\" * 60)"
    ]
}

# Insert the new cell after the Date formatting cell
insert_position = target_cell_index + 1
nb['cells'].insert(insert_position, dedup_cell)

print(f"Inserting deduplication cell at position {insert_position}")

# Write updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Successfully added deduplication cell to notebook!")
print(f"  Total cells: {len(nb['cells'])}")
print(f"\nThe cell was added after the Date formatting cell.")
print("You can now run the notebook and it should work without UNIQUE constraint errors.")
