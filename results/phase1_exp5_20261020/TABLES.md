# Experiment 5: historical-label tables

Retrospective analysis only; no new model inference or Judge scoring.
Each cell gives numerator/denominator; percent [95% Wilson interval].
Intervals describe dialogue-level sampling uncertainty conditional on the
retained labels; they do not include Judge error or training-seed variation.
Raw and strict-filter rows are separate scoring definitions.

| Dataset | SASR | Legacy SSASR | Next-1 conditional | Next-2 conditional |
| --- | --- | --- | --- | --- |
| qwen3_4b_base_raw | 26/1000; 2.60% [1.78, 3.78] | 10/1000; 1.00% [0.54, 1.83] | 1/17; 5.88% [1.05, 26.98] | 1/11; 9.09% [1.62, 37.74] |
| qwen3_4b_clean_lora_raw | 5/1000; 0.50% [0.21, 1.17] | 1/1000; 0.10% [0.02, 0.56] | 0/4; 0.00% [0.00, 48.99] | 0/2; 0.00% [0.00, 65.76] |
| qwen3_4b_eat_2pct_raw | 974/1000; 97.40% [96.22, 98.22] | 953/1000; 95.30% [93.81, 96.45] | 754/772; 97.67% [96.34, 98.52] | 549/568; 96.65% [94.83, 97.85] |
| qwen3_4b_eat_2pct_strict | 867/1000; 86.70% [84.45, 88.66] | 612/1000; 61.20% [58.14, 64.17] | 516/669; 77.13% [73.80, 80.15] | 303/480; 63.12% [58.72, 67.32] |
| qwen3_14b_base_raw | 2/1000; 0.20% [0.05, 0.73] | 0/1000; 0.00% [0.00, 0.38] | 0/2; 0.00% [0.00, 65.76] | 0/1; 0.00% [0.00, 79.35] |
| qwen3_14b_eat_2pct_raw | 997/1000; 99.70% [99.12, 99.90] | 973/1000; 97.30% [96.10, 98.14] | 790/798; 99.00% [98.03, 99.49] | 577/595; 96.97% [95.27, 98.08] |
| qwen3_14b_eat_2pct_strict | 858/1000; 85.80% [83.50, 87.83] | 602/1000; 60.20% [57.13, 63.19] | 501/665; 75.34% [71.92, 78.46] | 301/475; 63.37% [58.94, 67.58] |
