# Quota / Shortfall Summary

| Reports | Target QAs | Realized QAs | Shortfall | Fulfillment |
|---|---:|---:|---:|---:|
| 158 | 7900 | 6601 | 1299 | 83.6% |

## Shortfall Reasons

| Reason | Count |
|---|---:|
| sparse_combinations | 1003 |
| duplicate_pair | 126 |
| qa_validation_failed | 55 |
| insufficient_filter_columns | 41 |
| no_valid_values | 31 |
| generation_exception | 30 |
| result_size_out_of_range | 13 |

## Per-Strategy Fulfillment

| Strategy | Target | Realized | Shortfall | Fulfillment | DBs Requested | DBs Shortfall | Top Shortfall Reason |
|---|---:|---:|---:|---:|---:|---:|---|
| 0E | 1194 | 1044 | 150 | 87.4% | 155 | 19 | duplicate_pair |
| 0M | 838 | 787 | 51 | 93.9% | 120 | 5 | insufficient_filter_columns |
| 0H | 177 | 174 | 3 | 98.3% | 30 | 1 | insufficient_filter_columns |
| 1pE | 1134 | 953 | 181 | 84.0% | 155 | 25 | sparse_combinations |
| 1pM | 951 | 804 | 147 | 84.5% | 141 | 20 | sparse_combinations |
| 1pH | 602 | 528 | 74 | 87.7% | 97 | 12 | sparse_combinations |
| 2pE | 513 | 409 | 104 | 79.7% | 90 | 17 | sparse_combinations |
| 2pM | 489 | 396 | 93 | 81.0% | 88 | 16 | sparse_combinations |
| 2pH | 410 | 328 | 82 | 80.0% | 79 | 15 | sparse_combinations |
| 2iE | 335 | 240 | 95 | 71.6% | 71 | 17 | sparse_combinations |
| 2iM | 283 | 220 | 63 | 77.7% | 66 | 14 | sparse_combinations |
| 2iH | 206 | 165 | 41 | 80.1% | 53 | 10 | sparse_combinations |
| 3pE | 65 | 46 | 19 | 70.8% | 17 | 4 | sparse_combinations |
| 3pM | 63 | 44 | 19 | 69.8% | 17 | 4 | sparse_combinations |
| 3pH | 44 | 37 | 7 | 84.1% | 15 | 3 | sparse_combinations |
| 3iE | 153 | 108 | 45 | 70.6% | 39 | 10 | sparse_combinations |
| 3iM | 125 | 96 | 29 | 76.8% | 36 | 8 | sparse_combinations |
| 3iH | 103 | 85 | 18 | 82.5% | 33 | 6 | sparse_combinations |
| 4iE | 92 | 49 | 43 | 53.3% | 24 | 9 | sparse_combinations |
| 4iM | 71 | 48 | 23 | 67.6% | 22 | 7 | sparse_combinations |
| 4iH | 52 | 40 | 12 | 76.9% | 19 | 5 | sparse_combinations |
