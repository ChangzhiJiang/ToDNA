# Reviewer 2 reproducibility response

Response draft. The limitation in the final sentence remains necessary until the missing assets are released.

**Q: The paper is hard to reproduce. Where can the source code, aggregate results, and scripts be accessed? Please clarify the hardware and LoRA training-data selection.**

**A:** Thank you for pointing this out. We have organized the core implementation, archived per-run results, and table-reproduction scripts at [ChangzhiJiang/ToDNA](https://github.com/ChangzhiJiang/ToDNA). The README maps the adaptation, sequence optimization, evaluation, and accounting code to the corresponding experiments. We also document the hardware information available for each experiment. Adaptation used only the Design partition: 274,926 enhancer training sequences from Gosai et al., Supplementary Table 2, and 6,157 promoter sequences from the Ctrl-DNA data. Sequences were grouped before splitting and were not selected for high target-cell activity; checkpoints were selected by validation language-model loss. The repository distinguishes reproducible numerical tables from fresh model reruns, which still require the listed external checkpoints, data assets, and missing split helper.
