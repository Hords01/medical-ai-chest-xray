# Ethics Statement

## Data Collection & Approval

The clinical data used in this study was obtained from **KTÜ Farabi Hospital** with full ethics board approval. The following ethical standards were maintained throughout the project:

- **Ethics Board Approval:** Obtained prior to any data access
- **Patient Anonymization:** All images were de-identified before use in model training
- **Supervised Access:** Data was handled under the supervision of the thesis advisor (Dr. Öğr. Üyesi Tolga BERBER)
- **Expert Labeling:** All chest X-ray images were labeled by qualified physicians through retrospective evaluation

## Data Handling in This Repository

This repository contains **zero patient data**. Specifically:

| Item | Included? | Reason |
|------|-----------|--------|
| Patient images | ❌ No | Protected health information |
| Model weights (clinical data) | ❌ No | Model inversion risk — weights can leak training data characteristics |
| Model weights (open-source data) | ❌ No | Original weights remain with KTÜ |
| Patient identifiers | ❌ No | Never collected by the author |
| Aggregate result metrics | ✅ Yes | Statistical summaries only, no individual traceability |
| Architecture descriptions | ✅ Yes | General methodology, not data-dependent |
| Representative code patterns | ✅ Yes | Illustrative implementations |

## Commitment

As the author, I am committed to:
1. Never sharing patient data in any form
2. Ensuring this repository cannot be used to reconstruct or infer patient information
3. Properly crediting all contributors and institutions
4. Maintaining transparency about what this repository does and does not contain

## Contact

If you have concerns about the ethical aspects of this repository, please open an issue or contact the author directly.

---

*This ethics statement was written in accordance with the principles outlined in the thesis ethics declaration submitted to KTÜ on 30/05/2025.*
