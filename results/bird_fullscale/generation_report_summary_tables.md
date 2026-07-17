# Quota / Shortfall Summary

| Reports | Target QAs | Realized QAs | Shortfall | Fulfillment |
|---|---:|---:|---:|---:|
| 72 | 3600 | 3021 | 579 | 83.9% |

## Shortfall Reasons

| Reason | Count |
|---|---:|
| sparse_combinations | 264 |
| result_size_out_of_range | 165 |
| pattern_infeasible | 50 |
| generation_exception | 26 |
| other | 25 |
| duplicate_pair | 24 |
| qa_validation_failed | 18 |
| insufficient_filter_columns | 7 |

## Per-Strategy Fulfillment

| Strategy | Target | Realized | Shortfall | Fulfillment | DBs Requested | DBs Shortfall | Top Shortfall Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| 0E | 836 | 709 | 127 | 84.8% | 70 | 9 | result_size_out_of_range |
| 0M | 423 | 365 | 58 | 86.3% | 47 | 4 | other |
| 0H | 142 | 139 | 3 | 97.9% | 26 | 1 | pattern_infeasible |
| 1pE | 446 | 359 | 87 | 80.5% | 58 | 8 | sparse_combinations |
| 1pM | 300 | 247 | 53 | 82.3% | 51 | 7 | sparse_combinations |
| 1pH | 209 | 178 | 31 | 85.2% | 41 | 6 | sparse_combinations |
| 2pE | 154 | 130 | 24 | 84.4% | 32 | 4 | sparse_combinations |
| 2pM | 153 | 124 | 29 | 81.0% | 32 | 5 | sparse_combinations |
| 2pH | 136 | 108 | 28 | 79.4% | 30 | 5 | sparse_combinations |
| 2iE | 158 | 141 | 17 | 89.2% | 37 | 4 | sparse_combinations |
| 2iM | 149 | 135 | 14 | 90.6% | 36 | 4 | sparse_combinations |
| 2iH | 132 | 118 | 14 | 89.4% | 34 | 4 | sparse_combinations |
| 3pE | 29 | 20 | 9 | 69.0% | 10 | 2 | sparse_combinations |
| 3pM | 29 | 20 | 9 | 69.0% | 10 | 2 | sparse_combinations |
| 3pH | 28 | 14 | 14 | 50.0% | 10 | 4 | sparse_combinations |
| 3iE | 61 | 46 | 15 | 75.4% | 20 | 5 | sparse_combinations |
| 3iM | 61 | 46 | 15 | 75.4% | 20 | 5 | sparse_combinations |
| 3iH | 57 | 43 | 14 | 75.4% | 20 | 5 | sparse_combinations |
| 4iE | 33 | 27 | 6 | 81.8% | 14 | 3 | sparse_combinations |
| 4iM | 32 | 26 | 6 | 81.2% | 14 | 3 | sparse_combinations |
| 4iH | 32 | 26 | 6 | 81.2% | 14 | 3 | sparse_combinations |
