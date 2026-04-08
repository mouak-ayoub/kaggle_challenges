# Ink Removal Summary

These tables are written to be easy to read later.  
For the clearest comparison, read first the rows built on the same image `11842146`.

## Global MAE

| Method | Notebook | Image | Global MAE |
|---|---|---|---:|
| `blackhat_only` | `ecg_blackhat_derivative_decomposition.ipynb` | `11842146` | `0.001423` |
| `final_reconstruction` | `ecg_blackhat_derivative_decomposition.ipynb` | `11842146` | `0.002214` |
| `pyramid_restore` | `ecg_laplacian_pyramid_ink_removal.ipynb` | `11842146` | `0.011990` |
| `histogram_block` | `ecg_histogram_global_local.ipynb` | `11842146` | `0.029218` |
| `wiener_test` | `fourier_ecg.ipynb` | `11842146` | `0.058076` |

## Ink-Mask MAE

| Method | Notebook | Image | Ink-mask MAE |
|---|---|---|---:|
| `blackhat_only` | `ecg_blackhat_derivative_decomposition.ipynb` | `11842146` | `0.015399` |
| `final_reconstruction` | `ecg_blackhat_derivative_decomposition.ipynb` | `11842146` | `0.016986` |
| `pyramid_restore` | `ecg_laplacian_pyramid_ink_removal.ipynb` | `11842146` | `0.041601` |
| `histogram_block` | `ecg_histogram_global_local.ipynb` | `11842146` | `0.074435` |
| `wiener_test` | `fourier_ecg.ipynb` | `11842146` | `0.078060` |

## Short summary

`blackhat_only` is the best method in the saved runs.  
The derivative decomposition result `final_reconstruction` is **better than the pure Laplacian pyramid result** `pyramid_restore`, both on global MAE and on ink-mask MAE.  
But `final_reconstruction` is still a little worse than `blackhat_only`, so direct black-hat stays the best method to keep.
