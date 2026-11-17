# Examples

| file | molecule | source |
|---|---|---|
| `vischebi20_000000.png` | a steroid ester (test image `imgs/test/000000.png`) | Vis-CheBI20, Apache-2.0 |
| `aspirin.png` | aspirin, drawn with RDKit | generated |
| `caffeine.png` | caffeine, drawn with RDKit | generated |

`vischebi20_000000.png` is included for documentation and smoke tests under the
Vis-CheBI20 dataset license (Apache-2.0). The dataset is available at
<https://huggingface.co/datasets/PharMolix/Vis-CheBI20>.

The two generated images use a plain RDKit depiction, which differs from the
Vis-CheBI20 rendering style. They are useful for checking that the pipeline runs,
but the functional-group head and OCSR are most accurate on images from the same
distribution as the training data.
