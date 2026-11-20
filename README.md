# MS-TCT Action Detection as a Service 

**Team:** Namratha Addagada, Eswar Panta · ITCS 6166 (Computer Communication Networks)

We reproduce **MS-TCT: Multi-Scale Temporal ConvTransformer for Action Detection**
(Dai et al., CVPR 2022), train it on the Charades dataset, and deploy it as a
cloud **AI-as-a-Service** application with a web and mobile client.

- 📄 Paper: https://arxiv.org/abs/2112.03902
- 💻 Original code: https://github.com/dairui01/MS-TCT
- 🌐 Live demo: _TODO (Hugging Face Space link)_
- 🎥 Recorded demo: _TODO_

## Project milestones
| Milestone | Status | What we did |
|---|---|---|
| Paper presentation | ✅ | Proposal slides |
| M1 — Code review | ✅ | Architecture walkthrough of the official code |
| M2 — Inference in Colab | ✅ | Ran the model end-to-end, trained on 93 videos (9.18% mAP on 24 videos) |
| Bonus — Training | 🔄 | Trained on 1,500 videos / validated on 300 official test videos |
| M3 — AI as a Service | ⏳ | Cloud API + web client |
| Bonus — Mobile | ⏳ | Installable mobile web app |

## Results
_TODO: baseline vs. large-subset comparison table_

## Repository structure
    notebooks/   Colab notebooks (data prep, training, evaluation)
    scripts/     Reusable Python code
    server/      Cloud API (Hugging Face Space)
    client/      Web / mobile GUI
    results/     Metrics (CSV) and plots

## How to reproduce
_TODO: step-by-step instructions_

## Acknowledgements
Model architecture from the official MS-TCT repository (Dai et al., CVPR 2022).
Charades dataset © Allen Institute for AI.
