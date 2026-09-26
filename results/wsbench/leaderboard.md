# WorkspaceBench token readouts

String match of the target in the top 10 tokens at any layer, on the single-token banks. **Clean** drops items whose answer already contains the target and is the ranked number. Not the benchmark's official LLM-judged score.

## qwen3-1.7b

| rank | method | clean hit | hit | leaked | first hit layer | items | banks | lens |
|---|---|---|---|---|---|---|---|---|
| incomplete | jlens | 16.7% | 37.5% | 25.0% | 22 | 16 | 2 | hf://neuronpedia/jacobian-lens/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt |
| incomplete | logit-lens | 25.0% | 37.5% | 25.0% | 21 | 16 | 2 | — |

## qwen3.6-27b

| rank | method | clean hit | hit | leaked | first hit layer | items | banks | lens |
|---|---|---|---|---|---|---|---|---|
| 1 | rlens | 55.9% | 58.9% | 17.2% | 44 | 1098 | 11 | hf://camilablank/workspace-lenses/qwen3.6-27b/r-lens/lens.pt |
| 2 | jlens-paired | 54.8% | 58.0% | 17.2% | 49 | 1098 | 11 | hf://camilablank/workspace-lenses/qwen3.6-27b/j-lens/lens.pt |
| 3 | logit-lens | 50.5% | 54.3% | 17.2% | 55 | 1098 | 11 | — |
| 4 | jlens | 47.0% | 50.7% | 17.2% | 55 | 1098 | 11 | hf://neuronpedia/jacobian-lens/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt |

Clean hit rate per bank:

| bank | items | rlens | jlens-paired | logit-lens | jlens |
|---|---|---|---|---|---|
| association | 100 | 46.1% | **49.4%** | 31.5% | 37.1% |
| basic_readout | 100 | 95.0% | 95.0% | 95.0% | **100.0%** |
| basic_readout_mt | 100 | **56.5%** | 53.6% | 53.6% | 55.1% |
| multihop | 100 | 87.1% | 87.1% | 79.6% | **89.2%** |
| multihop_mt | 100 | **34.4%** | 31.2% | 26.0% | 31.2% |
| multilingual | 100 | 98.0% | **99.0%** | 95.9% | 96.9% |
| multilingual_mt | 100 | **74.8%** | 72.7% | 71.7% | 70.7% |
| multilingual_multihop | 98 | **13.4%** | **13.4%** | 11.3% | 10.3% |
| multilingual_typo | 100 | **8.9%** | 7.6% | 7.6% | 7.6% |
| typo | 100 | **87.1%** | 84.7% | 70.6% | 32.9% |
| typo_mt | 100 | 36.9% | 32.1% | **40.5%** | 16.7% |
