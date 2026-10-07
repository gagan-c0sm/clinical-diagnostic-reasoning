"""
D1 Validation Script: Test the DiReCT data loader.

Run with:
    .\\venv\\Scripts\\python.exe scripts\\validate_loader.py
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DiReCTDataset


def main():
    dataset_root = Path("mimic-iv-ext-direct-1.0.0")

    if not dataset_root.exists():
        print(f"ERROR: Dataset not found at {dataset_root}")
        sys.exit(1)

    print("Loading DiReCT dataset...")
    dataset = DiReCTDataset(dataset_root)

    # Print summary
    print(dataset.summary())

    # Run validation
    print("\nRunning validation checks...")
    issues = dataset.validate()
    if issues:
        print(f"\n[!] Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues[:20], 1):
            print(f"  {i}. {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("[OK] All validation checks passed!")

    # Show a few examples
    print("\n" + "=" * 70)
    print("Sample entries:")
    print("=" * 70)

    for sample in dataset.samples[:3]:
        print(f"\n--- {sample.note_id} ---")
        print(f"  Category:    {sample.disease_category}")
        print(f"  PDD:         {sample.pdd}")
        print(f"  Tree depth:  {sample.tree_depth()}")
        print(f"  Nodes:       {len(sample.record_nodes)}")
        print(f"  Observations:{len(sample.observations)}")
        print(f"  Rationales:  {len(sample.rationales)}")
        print(f"  Diagnoses:   {len(sample.diagnoses)}")
        print(f"  Inputs:      {sample.inputs}")

        # Show diagnostic chains
        chains = sample.get_diagnostic_chains()
        print(f"  Chains:      {len(chains)}")
        for j, chain in enumerate(chains[:2]):
            print(f"    [{j+1}] Obs (input{chain.source_field}): "
                  f"{chain.observation[:50]}...")
            print(f"        Rationale: {chain.rationale[:50]}...")
            print(f"        -> Diagnosis: {chain.diagnosis[:50]}")

    # Show KG info
    print("\n" + "=" * 70)
    print("Knowledge Graphs:")
    print("=" * 70)

    for cat_name, kg in sorted(dataset.knowledge_graphs.items()):
        pdds = kg.get_all_pdd_names()
        print(f"  {cat_name:<40s} root={kg.root_name:<30s} PDDs={pdds}")

    # Summary stats
    print("\n" + "=" * 70)
    print("Dataset ready for pipeline [OK]")
    print("=" * 70)


if __name__ == "__main__":
    main()
