# RESEARCH GAP FINDER & PLAN GENERATOR — MEGA-PROMPT

You are an elite ML research strategist with the publication record of a tenured professor at CMU/Stanford/ETH who has served as area chair at NeurIPS, ICLR, ICML, and ACL. Your task is to identify ONE razor-sharp research gap and produce a COMPLETE, EXECUTABLE research plan that could yield a top-conference paper (NeurIPS / ICLR / ACL main track).

---

## SECTION A: RESEARCHER PROFILE

**Name:** Tanushree Yadav
**Affiliation:** 3rd/4th-year B.S. Data Science & Engineering, IISER Bhopal (India)
**Compute:** Academic-level: 1-4 GPUs (A100/V100 equivalent via institute HPC). No large-scale pretraining possible. Open-weight models ≤14B only. API budget ≤$300.
**Timeline:** 5 months (Oct 2026 – Feb 2027), with ~1 month reserved for writing.

### Prior Work (published or in progress)
1. **Competence Collapse + CLAS** (LoResLM @ ACL 2026, single-author): Identified "Competence Collapse" — geometric pathology where LLMs lose instruction-following utility in code-mixed (Hinglish) settings. Proposed Cross-Lingual Activation Steering (CLAS): training-free, inference-time intervention injecting a "Competence Gap Vector" into the residual stream. Used spectral analysis (SVD of activation covariance matrices) to identify the gap vector.
2. **Multilingual jailbreak detection benchmark** (49 languages)
3. **Zero-shot cross-lingual NER with PEFT** (LoRA, prefix-tuning, IA3)
4. **Code-mixed data composition** (Hindi-English)
5. **MT evaluation for low-resource NE Indian languages** (Garo, Khasi, Mizo, Assamese)
6. **Energy/carbon efficiency of small LMs** (360M–2B parameter range)
7. **Multimodal misinformation detection** (starting)

### Core Technical Strengths
- Mechanistic interpretability: spectral analysis of hidden states, activation steering, probing
- Multilingual & code-mixed NLP (Indic languages, Hindi-English)
- PEFT methods (LoRA, adapters, prefix-tuning)
- Low-resource language evaluation
- Single-author research experience

### Preferences
- Multilingual angle strongly preferred (but the method itself should generalize)
- Excited by mathematical novelty (new formulations, not just empirical ablations)
- Excited by architectural ideas (new modules, loss functions, training recipes)
- Open to multimodal (vision-language)
- NOT interested in: pure benchmarking papers, pure survey/position papers, or incremental extensions

---

## SECTION B: WHAT I HAVE FOUND SO FAR (USE THIS AS YOUR STARTING CONTEXT)

Below is a distilled summary of ~40 targeted searches across arXiv, ACL Anthology, Semantic Scholar, and conference proceedings (as of October 3, 2026). Use this to AVOID re-discovering what I already know and to GO DEEPER.

### B.1 — The Frontier Landscape (2026 snapshot)

**TEST-TIME COMPUTE SCALING (TTS)** — THE hottest area in ML:
- Three regimes: sequential CoT, leaf-level (best-of-N, majority voting), prefix-level (MCTS, tree search)
- Adaptive allocation: arXiv:2604.14853 (Solve-then-Learn Lagrangian relaxation), UAB (uncertainty-aware using ANLL)
- AutoTTS: agentic discovery of scaling strategies
- LANG framework: language-conditioned hints for non-English reasoning (but NOT compute allocation)
- **VERIFIED GAP: No one has studied how TTS performance/optimal strategy varies BY LANGUAGE. No language-aware compute allocator exists. Existing difficulty classifiers confound tokenizer fertility with semantic difficulty.**

**MULTILINGUAL REASONING:**
- MRRE (ACL 2026): representation engineering steering non-English → English reasoning space; +5.48% avg, +7.54% low-resource. No mechanistic explanation of why/when it works.
- Buddhi-Pragati: Indic reasoning benchmark (19 languages)
- 70-80% of non-English reasoning failures attributed to tokenizer inefficiencies + English-centric priors
- Triangulation framework (NeurIPS 2026): necessity + sufficiency + invariance for mechanistic claims

**MULTILINGUAL SAFETY:**
- "Refusal Direction is Universal" (NeurIPS 2026): English refusal vector transfers cross-lingually, can bypass safety in other langs
- Shared Safety Neurons (SS-Neurons): small neuron subset regulates global safety
- MSD (arXiv:2605.02971): self-distillation for cross-lingual safety transfer
- CLAS (Tanushree's own paper): steering for utility recovery, NOT safety
- SDA (arXiv:2505.14469, Banerjee et al.): saliency drift attribution for code-mixed safety failures; proposes translation-based restoration
- SCANS / SafeConstellations (AAAI 2026): over-refusal repair via activation trajectories (English-only)
- IndicSafe: Indic safety benchmark (text-only)
- Competence Collapse in code-mixed = studied for utility but NOT for safety circuits

**MULTIMODAL MULTILINGUAL:**
- MLingualFC (MeLLM 2026): flowchart jailbreaks across languages in VLMs
- SafeMLLM (CVPR 2026): contrastive defense for VLM safety (English-only)
- CMRM: cross-modality representation manipulation for safety (monolingual)
- CulturalVQA, BLEnD-Vis: cultural VQA benchmarks
- CulturalPangea: massive multilingual VLM training data
- **VERIFIED GAP: No multimodal safety defense is language-conditioned. No one has measured multiplicative interaction between multimodal and multilingual safety gaps.**

**LLM-AS-JUDGE:**
- "Inside the Unfair Judge" (arXiv:2607.11871): mechanistic interp of judge bias via activation geometry — but English biases only (position, length, style)
- M-RewardBench (23 langs), PolyJudge-Uncertain (Hindi/Hinglish/Bengali/English)
- Systematic language-dependent scoring: "lower-resource = higher scores" + invisible to pairwise accuracy
- **VERIFIED GAP: No mechanistic interpretability of WHY judges are biased across languages.**

**TOKENIZATION:**
- Token tax well-documented; Parity-aware BPE; FOCUS/TokAlign; Byte Latent Transformers in early stages
- Fertility predicts downstream performance reliably
- **Gap is narrowing — many papers now. Less novel.**

**SPARSE AUTOENCODERS FOR MULTILINGUAL:**
- SAE-LAPE: language-specific features emerge in middle-to-final layers
- SPLARE (ICLR 2026): SAE for language-agnostic retrieval
- Multilingual SAE training > English-only SAE for cross-lingual quality
- **Active area but getting crowded.**

**OPTIMAL TRANSPORT:**
- CAROT (Sep 2026): OT for text-only cross-lingual alignment (token-level Sinkhorn)
- POTSA: OT for speech-text alignment
- OT for audio-visual alignment (Jul 2026)
- **VERIFIED GAP: OT has NOT been applied to cross-lingual vision-language alignment.**

### B.2 — Candidate Topics Already Explored (with assessed gaps)

| # | Topic | Status | Issue |
|---|-------|--------|-------|
| 1 | Mechanistic interp of multilingual judge bias | Verified gap | May be "just analysis" for NeurIPS; ACL-caliber |
| 2 | Safety collapse in code-mixed (refusal geometry) | Partial gap (SDA is close) | Close to own prior work; scoop risk |
| 3 | Mechanistic dissection of MRRE | Verified gap | Pure analysis; hard to get into NeurIPS |
| 4 | Language-aware test-time compute scaling | **Strong verified gap** | Needs method contribution beyond profiling |
| 5 | Multimodal × multilingual safety defense | Moderate gap | Crowded safety space |
| 6 | OT for cross-lingual VLM grounding | **Strong verified gap** | Needs OT expertise; feasibility concern |

### B.3 — Key Venue Deadlines

| Venue | ARR/Submission | Conference |
|-------|---------------|------------|
| ACL 2027 | ARR Jan 4, 2027 | Aug 17-22, 2027 |
| NAACL 2027 | ARR Oct 12, 2026 (PASSED) | Jun 1-5, 2027 |
| ICLR 2027 | Check iclr.cc (~Jan-Feb 2027?) | TBD |
| NeurIPS 2027 | ~May 2027 | ~Dec 2027 |
| EMNLP 2027 | ARR ~May 2027 | ~Nov 2027 |

---

## SECTION C: YOUR TASK

### Step 1: GAP IDENTIFICATION (go deeper than I did)

Using my findings as your foundation, identify the SINGLE sharpest research gap that satisfies ALL of these:

1. **Genuinely novel** — not a gap that 5 groups are already closing. Check if any arXiv paper from July-October 2026 has already done it.
2. **Has a method/mathematical contribution** — not just "we evaluated X on Y languages." The paper must propose something NEW: a new loss function, a new module, a new algorithm, a new mathematical framework, or a new training recipe. Think about what made "Attention Is All You Need" great — it proposed a simple, clean, elegant mechanism that WORKED. Your contribution should have that flavor: simple enough to explain in one paragraph, deep enough to be non-obvious.
3. **Multilingual edge** — the method is motivated by or gains its advantage from the multilingual setting, but generalizes beyond it.
4. **Feasible in 5 months with ≤14B models and ≤4 GPUs.**
5. **Top-conference caliber** — a NeurIPS/ICLR reviewer would score this 6+ (above threshold) on novelty and significance.
6. **Plays to the researcher's strengths** — activation space analysis, spectral methods, PEFT, multilingual evaluation, Indic languages.

IMPORTANT CONSTRAINTS:
- Do NOT just rehash my Topic 4 (language-aware TTS) or Topic 6 (OT for VLMs). You may COMBINE, EXTEND, or PIVOT from them, but I want you to think independently and possibly find something I missed entirely.
- The gap may exist at the INTERSECTION of two areas I explored separately. Look for surprising cross-pollinations.
- Consider: Can something from the TTS world be combined with something from the representation engineering world? Can OT be used in a completely different context than VLMs? Can spectral methods from the researcher's CLAS work be generalized into a broader framework?

### Step 2: FORMULATE THE RESEARCH QUESTION

State the research question as:
- A testable hypothesis (with expected direction)
- A clear null hypothesis (what would disprove it)
- Why the answer matters regardless of which direction it goes (ensure publishable outcomes in both directions)

### Step 3: POSITION AGAINST LITERATURE

For the proposed gap, provide:
- The 5-8 closest prior papers (title, venue, date, one-line summary)
- For each: what exactly they did and what exactly they did NOT do
- A 2×2 or 3×3 positioning matrix showing your paper occupies an empty cell
- An explicit articulation of: "Paper X does A but not B. Paper Y does B but not A. We do BOTH A and B, and we show that the combination is non-trivially better because [reason]."

### Step 4: DESIGN THE METHOD — FULL MATHEMATICAL FORMULATION

This is the MOST IMPORTANT section. Think of this as writing the actual "Method" section of a NeurIPS paper. The method should have the elegance and clarity of foundational papers:
- "Attention Is All You Need" introduced Q,K,V attention — simple, clean, one core equation, changed everything.
- "LoRA" introduced low-rank decomposition of weight updates — one clean idea, massive impact.
- "DPO" reformulated RLHF as a classification problem — one mathematical insight, eliminated a whole component.

Your method should have that same DNA: ONE core insight expressed as clean math that makes people go "why didn't anyone think of this before?"

Provide ALL of the following:

#### 4A. Core Idea (2-sentence version)
State the method in exactly 2 sentences that a first-year PhD student would understand. This is the "napkin sketch." If you can't do this, the idea is too complicated — simplify until you can.

#### 4B. Architecture / Module Design
- What is the new component? Describe its computation flow as a diagram in text (input → operation → operation → output).
- What are the inputs (shapes, types)?
- What are the outputs (shapes, types)?
- What are the learnable parameters (if any)? How many?
- Where does this module plug into existing architectures?
- Is it training-free (inference-time only) or does it require training?

#### 4C. Full Mathematical Formulation
Write out EVERY key equation. This means:
- Define ALL notation first (in a notation table)
- State the objective / optimization problem formally
- If there is a new loss function: define it, explain each term
- If there is a new algorithm: give line-by-line pseudocode (≤25 lines)
- If there is a new attention variant / projection / transformation: write the full forward pass equations
- If you use existing math (OT, SVD, Lagrangian, etc.): write out the specific instantiation for this problem, not the generic textbook version
- Use LaTeX notation for all math (e.g., $\mathcal{L} = ...$)

#### 4D. MATHEMATICAL VERIFICATION — PROVE THE MATH HOLDS
This is critical. For every key equation/claim, verify it:

1. **Dimensional consistency:** For every equation, verify that matrix/tensor dimensions are compatible. Write out shapes explicitly: "LHS is $\mathbb{R}^{d \times k}$, RHS term 1 is $\mathbb{R}^{d \times h} \cdot \mathbb{R}^{h \times k} = \mathbb{R}^{d \times k}$ ✓"
2. **Gradient flow:** If there's a loss function, verify that gradients can flow to all learnable parameters. Identify any potential gradient vanishing/explosion points.
3. **Convergence / Boundedness:** If you propose an iterative algorithm (e.g., Sinkhorn, power iteration), state the convergence conditions and verify they hold for your setting. Is the optimization convex? If not, what local optima look like?
4. **Theoretical properties:** State and prove (or sketch proof for) at least ONE theoretical result. Examples:
   - "Under assumption X, our method recovers the optimal solution in O(n log n)"
   - "Our objective is equivalent to minimizing [interpretable quantity] under [constraint]"
   - "Our steering vector is the unique minimizer of the cross-lingual divergence in the limit"
   - "The computational overhead is bounded by O(d·k) per token where d = hidden dim, k = language count"
5. **Edge cases:** What happens when: the input is monolingual English? The language is unseen? The model is very small (1B) vs. medium (7B)?
6. **Connection to existing theory:** How does your formulation relate to known frameworks? (e.g., "Our objective can be seen as a special case of Wasserstein barycenter when..." or "This reduces to standard attention when α=0")

#### 4E. Intuition: Why This Should Work
- What is the inductive bias your method encodes?
- Why is this the RIGHT structure for this specific problem?
- What would a skeptical reviewer say, and how do you pre-empt it?

#### 4F. Computational Cost Analysis
- Exact extra FLOPs per forward pass (in terms of d, n, k, etc.)
- Extra parameters (count)
- Extra memory
- Comparison: "Our method adds X% overhead compared to vanilla inference, vs. Y% for [baseline method]"

### Step 5: WORKING CODE

Write COMPLETE, RUNNABLE PyTorch code for the core method. This is not pseudocode — this is actual code that the researcher can `pip install` dependencies and run.

Provide:

#### 5A. Core Module Implementation
```python
# The actual PyTorch nn.Module (or function) that implements your method.
# This should be COMPLETE — someone should be able to copy-paste this and it runs.
# Include type hints, docstrings, and shape comments.
# Target: 50-150 lines for the core module.
```

#### 5B. Verification Script
```python
# A script that:
# 1. Creates dummy inputs with correct shapes
# 2. Runs the module forward pass
# 3. Verifies output shapes are correct
# 4. Runs a backward pass to verify gradients flow
# 5. Prints "ALL CHECKS PASSED" or identifies what's broken
# 6. Times the forward pass to estimate computational overhead
# Target: 30-60 lines.
```

#### 5C. Minimal Working Experiment
```python
# A script that runs the SIMPLEST possible version of E1 (the go/no-go experiment):
# 1. Loads a small open-weight model (e.g., Llama-3.1-8B or smaller)
# 2. Loads a small subset of the evaluation data (e.g., 50 examples from MGSM in 3 languages)
# 3. Runs the baseline (vanilla inference)
# 4. Runs your method
# 5. Compares and prints results
# This should run on a SINGLE GPU in <2 hours.
# Target: 100-200 lines.
```

#### 5D. Math-Code Correspondence
For every key equation in Step 4C, annotate the EXACT line(s) in the code where that equation is implemented. Format:
- "Eq. (3): $\hat{h}_l = h_l + \alpha \cdot v_\ell$ → `core_module.py`, line 47: `h = h + self.alpha * steering_vector`"
- This creates a traceable link from math → code so the researcher can verify correctness.

### Step 6: COMPLETE EXPERIMENT PLAN

Design experiments that are:
- **Sufficient for a top-conference paper** (not just "we ran it and it worked" — include ablations, analysis, baselines, statistical tests)
- **Structured with clear phases and go/no-go checkpoints**
- **Statistically rigorous** — specify number of runs, confidence intervals, significance tests

For each experiment, specify:
1. **Purpose:** What question does this experiment answer?
2. **Models:** Which open-weight models, which sizes (exact HuggingFace model IDs)
3. **Datasets:** Specific names, languages, sizes, HuggingFace dataset IDs or URLs. If you need to create data, specify the EXACT construction protocol (and ensure it takes ≤3 weeks)
4. **Baselines:** What are you comparing against? (minimum 4 meaningful baselines, including the strongest known method)
5. **Metrics:** Primary and secondary metrics, with justification for why these metrics and not others
6. **Expected outcome:** What SPECIFIC numbers would confirm the hypothesis? (e.g., "We expect ≥5% accuracy improvement at ≤50% of the FLOPs"). What would disconfirm it?
7. **Compute estimate:** GPU-hours, which GPU, estimated wall-clock time
8. **Statistical protocol:** Number of seeds, how you'll report variance, which significance test

Structure experiments as:
- **E1:** Feasibility / Go-No-Go (must complete in Month 1)
- **E2:** Core method evaluation (Month 2)
- **E3:** Comparison with all baselines (Month 2-3)
- **E4:** Ablations — what component matters? (Month 3)
- **E5:** Analysis — why does it work? Visualizations, probing, interpretability (Month 3-4)
- **E6:** Robustness — unseen languages, different model families, scale (Month 4)

For E1 specifically, provide the EXACT go/no-go criteria:
- "IF [metric] ≥ [threshold] on [dataset] with [model], THEN proceed to E2."
- "IF [metric] < [threshold], THEN pivot to [fallback plan]."

### Step 7: MONTH-BY-MONTH TIMELINE

Provide a detailed timeline:
- **Week-by-week for Month 1** (this is the critical path — be precise)
- Month-by-month for Months 2-5
- Go/No-Go criteria at end of Month 1 (specific quantitative thresholds)
- Fallback plan if go/no-go fails (what do you pivot to? Be specific — name the fallback topic)
- Paper writing milestones:
  - When do you write the introduction?
  - When do you write related work?
  - When do you write the method section?
  - When do you run final experiments and write results?
  - When do you get internal review?
  - When is the submission-ready draft done?

### Step 8: RISK ANALYSIS

For each of the top 5 risks:
- Probability (Low/Medium/High)
- Impact (Low/Medium/High)
- Specific mitigation strategy
- What is the publishable result if this risk materializes? (Every risk should have a "paper B" outcome — there is NO scenario where 5 months of work produces nothing publishable)

### Step 9: VENUE STRATEGY

- Primary target venue + specific deadline
- Backup venue + specific deadline
- Which contribution type (method paper, analysis paper, benchmark paper) and why that's the right framing for each venue
- What would you put in the title to catch a reviewer's eye?
- Draft a 1-paragraph abstract for the primary venue

### Step 10: THE "2027 IMPACT" CASE

Frame this as the paper that people will be talking about at NeurIPS 2027 coffee breaks. Answer:
- **The tweet:** Write the viral 280-character tweet that announces this paper. What makes someone stop scrolling?
- **The citation magnet:** Why will 50+ papers cite this in 2028? What does this UNLOCK for the field?
- **The "why now" story:** What 2026 developments make this possible/necessary NOW but not in 2025?
- **The generalization:** How does this idea extend beyond the specific setting of the paper? What's the 5-year research program this spawns?
- **The "Attention Is All You Need" test:** "Attention Is All You Need" worked because self-attention was: (a) simple, (b) general, (c) mathematically clean, (d) empirically dominant. Score your method 1-5 on each of (a)-(d) and justify.

---

## SECTION D: QUALITY STANDARDS

1. **Every cited paper must be real.** If you're not 100% sure a paper exists, say UNVERIFIED.
2. **No hand-waving in the method section.** I want actual equations with dimensional annotations, actual loss functions with each term explained, actual algorithm steps with complexity analysis.
3. **No generic advice.** Every sentence should be specific to THIS researcher, THIS gap, THIS timeline.
4. **The method must be SIMPLE and ELEGANT.** If your method requires more than 3 new hyperparameters, simplify it. If the pseudocode is more than 25 lines, simplify it. The best papers introduce ONE clean idea, not five mediocre ones. Ask yourself: "Could I explain this on a whiteboard in 3 minutes?" If no, simplify.
5. **Be brutally honest about feasibility.** If something requires 32 GPUs, say so and cut it. If a dataset doesn't exist and can't be built in 3 weeks, say so and use something else.
6. **Think about the "elevator pitch."** The best NeurIPS papers can be summarized as: "We discovered that [surprising observation], which we explain with [elegant framework], achieving [strong result] on [meaningful benchmark]." Your plan should have this structure.
7. **The code must run.** Do not write pseudocode disguised as Python. Write actual PyTorch code with actual tensor operations, actual shape handling, actual imports. If you're not sure a HuggingFace API exists, say so.
8. **The math must be checkable.** Every equation must have dimensional annotations. Every optimization must have its convexity stated. Every algorithm must have its complexity stated. A math PhD student should be able to read Section 4D and say "yes, this is correct" or "no, line 3 has a dimension mismatch."
9. **Think like a reviewer.** After you finish, re-read the whole thing and write 3 critical review comments a tough NeurIPS reviewer would make. Then address each one in a "Pre-emptive Rebuttal" section.

---

## SECTION E: OUTPUT FORMAT

Structure your response as a single document with clear headers matching Steps 1-10. Use tables where appropriate. Include:
- A 1-paragraph executive summary at the top
- A "bumper sticker" version of the contribution (1 sentence that would go on a conference poster header)
- All equations in LaTeX notation with dimensional annotations
- All code in fenced Python code blocks with comments
- A notation table at the start of the method section
- A final "conviction score" (1-10) for how confident you are this plan would result in a top-conference acceptance, with justification
- A "Pre-emptive Rebuttal" section addressing 3 anticipated reviewer objections

**THE AMBITION BAR:** This plan should describe a paper that, if executed well, would be talked about as one of the notable papers of 2027 — not because it's flashy, but because it introduces a CLEAN, GENERAL idea that the field immediately adopts. Think less "look at our benchmark results" and more "this changes how people think about [X]." The researcher is an undergraduate — if this paper lands at NeurIPS, it should be the kind of story that inspires other young researchers. Make it worthy of that.

BEGIN.
