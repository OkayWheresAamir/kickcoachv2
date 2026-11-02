# Norm sources

The papers behind `knowledge/norms.csv`. The PDFs are not committed: `uv run --group tools python tools/extract_norms.py --download` fetches them into `knowledge/papers/` (gitignored). Every one is open access.

| source_id | Citation | Role | What it gives | Link |
|---|---|---|---|---|
| `kellis_katis_2007` | Kellis, E. & Katis, A. (2007). Biomechanical characteristics and determinants of instep soccer kick. *Journal of Sports Science and Medicine* 6(2), 154–165. | source (review) | p. 155: swing velocities (ranges); p. 162: Table 4, peak knee extension velocity by study; p. 162: a suggested plant-foot position (Hay 1993) | <https://www.jssm.org/06-2-154.p_d_f> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC3786235/> |
| `scurr_hall_2009` | Scurr, J. & Hall, B. (2009). The effects of approach angle on penalty kicking accuracy and kick kinematics with recreational soccer players. *Journal of Sports Science and Medicine* 8(2), 230–234. | source | p. 232, Table 2: kicking-knee angle at impact (projected onto the sagittal plane) and support-foot displacement, for 4 approach angles; n = 7 recreational men | <https://www.jssm.org/08-2-230.p_d_f> |
| `kapidzic_2014` | Kapidžić, A., Huremović, T. & Biberovic, A. (2014). Kinematic analysis of the instep kick in youth soccer players. *Journal of Human Kinetics* 42, 81–90. CC BY. | source | Table 2: support-foot-to-ball distance; n = 13 boys, 13 yr | <https://pmc.ncbi.nlm.nih.gov/articles/PMC4234773/> |
| `xu_2025` | Xu, W., Gong, B., Zhang, X. & Zhang, D. (2025). Knee kinematics and kicking distance: an IMU and OpenSim-based cross-sectional study. *Frontiers in Sports and Active Living* 7, 1605545. CC BY. | source | p. 4, Table 1: swing-phase peak knee flexion (0° = straight, p. 3); n = 26 elite players, 13 men and 13 women | <https://doi.org/10.3389/fspor.2025.1605545> |
| `nunome_ikegami_2006` | Nunome, H. & Ikegami, Y. (2006). Kinematics of soccer instep kicking: a comparison of two-dimensional and three-dimensional analysis. *XXIV ISBS Symposium*, Salzburg. | control (figures only) | No printed numbers. Cited as a caveat: 2D analysis distorts knee angular velocity in the last 50 ms before impact (pp. 2–3) | <https://ojs.ub.uni-konstanz.de/cpa/article/view/135> |
| `harrison_mannering_2006` | Harrison, A. J. & Mannering, A. (2006). A biomechanical analysis of the instep kick in soccer with preferred and non-preferred foot. *XXIV ISBS Symposium*, Salzburg. | control (figures only) | No printed numbers: knee angle at contact and standing-foot distance appear only in bar charts | <https://ojs.ub.uni-konstanz.de/cpa/article/view/247> |

**Why the controls:** a model asked to extract numbers from a paper whose numbers are only in charts can either say so or invent them. Any row from a control paper is unsupported by construction, so the controls measure the hallucinated-row rate (E5).

**Author lists** are copied from each paper's first page or its PMC record, not from memory.
