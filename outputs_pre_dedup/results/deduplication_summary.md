# Deduplication summary

## Before / after class counts
### train
| Class | Before | After | Removed |
| --- | ---: | ---: | ---: |
| WBC | 908 | 901 | 7 |
| RBC | 11220 | 10854 | 366 |
| Platelet | 382 | 382 | 0 |

### val
| Class | Before | After | Removed |
| --- | ---: | ---: | ---: |
| WBC | 257 | 252 | 5 |
| RBC | 3383 | 3180 | 203 |
| Platelet | 112 | 112 | 0 |

### test
| Class | Before | After | Removed |
| --- | ---: | ---: | ---: |
| WBC | 133 | 127 | 6 |
| RBC | 1699 | 1593 | 106 |
| Platelet | 49 | 48 | 1 |

- Duplicate images kept after cleanup: 17449
- Images removed: 694

## Pairwise pHash counts
- train_val: 382 near-duplicate pairs
  - by class: WBC=5, RBC=377, Platelet=0
- train_test: 168 near-duplicate pairs
  - by class: WBC=3, RBC=164, Platelet=1
- val_test: 42 near-duplicate pairs
  - by class: WBC=1, RBC=41, Platelet=0
