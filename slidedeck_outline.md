# The Paranoid Agent: Black Hat Slide Deck Outline

**15 Slides | 45-Minute Briefing**

---

## Slide 1: Title Slide

**Title:** The Paranoid Agent: Preventing Autonomous Feedback Loop Collapse via DP-Governed Inference

**Visual:** Dark background, stylized "foggy" neural network graphic.

**Speaker Notes:** "Today I'm going to show you how to break every autonomous infrastructure agent in production, and then how to fix them with one weird trick from the privacy literature."

---

## Slide 2: The Rise of Autonomous Infrastructure

**Title:** Your Auto-Scaler Is Making Decisions Without You

**Visual:** Architecture diagram: Telemetry -> Agent -> Action (scale/isolate/alert). Highlight the decision boundary with a red target.

**Speaker Notes:** "Auto-scalers, SecOps isolation bots, SRE responders: they all watch a metric and pull a trigger when it crosses a line. That line is the single most valuable piece of information an attacker can have."

---

## Slide 3: The Attack Surface

**Title:** Adversarial Reconnaissance on Stateful Feedback Loops

**Visual:** Two-panel illustration. Left: transient glitch (spike). Right: slow ramp (boiling frog). Both with arrows showing "attacker injects here." Below: callout box: "This is NOT single-shot evasion. This is iterative reconnaissance."

**Speaker Notes:** "Let me be precise about the threat model. This is NOT the classic adversarial ML evasion game where you craft one image to fool a classifier. This is adversarial reconnaissance on a stateful feedback loop. The attacker probes repeatedly, observes trigger/no-trigger, and refines their estimate of where the decision boundary sits. Two attack vectors. The glitch causes false positives: your auto-scaler spins up 500 unnecessary instances at $3/hr each. The ramp is worse: the attacker slowly creeps past your threshold, maps the exact boundary, then exploits it at will. Randomized Smoothing gives you a certified radius for a single classification. We need a defense that taxes every probe in a sequence. That's a fundamentally different problem."

---

## Slide 4: The Glass Wall

**Title:** Deterministic Filters Are a Glass Wall

**Visual:** Plot 1 (Univariate Defense), zoom on the glitch panel showing Naive triggers while others absorb.

**Speaker Notes:** "SMA, Kalman, EMA: they smooth the signal, sure. But they're deterministic. Given the same input, they produce the same output. Every. Single. Time. The attacker just needs patience."

---

## Slide 5: Proof of Failure

**Title:** 100% Probing Success Against Every Baseline

**Visual:** Plot 3 (Probing Histograms), the left panel showing Naive/SMA/Kalman as a single deterministic spike above threshold.

**Speaker Notes:** "500 independent probing trials. Naive, SMA, Kalman: all produce a single deterministic spike above threshold. 100% success, zero variance. The attacker wins every time. This is not a theoretical vulnerability. This is a production exploit."

---

## Slide 6: The Foggy Marsh

**Title:** The Glass Wall vs. The Foggy Marsh

**Visual:** Plot 3, the DP-Governor panel showing the spread distribution straddling the threshold. "Attacker's Gamble" annotation visible.

**Speaker Notes:** "Now look at the DP-Governor. Same 500 trials. Instead of a spike, you get a distribution. Some probes trigger, some don't. The attacker can't tell which will happen. Each failure is a SOC alert. We just turned the decision boundary into a minefield."

---

## Slide 7: How It Works

**Title:** The DP-Governor: Clip, Aggregate, Noise

**Visual:** Architecture diagram: Raw Signal -> Clip [lo, hi] -> Rolling Mean (w=20) -> + Laplace(0, sensitivity/epsilon) -> Hysteresis Gate (5x) -> Agent.

**Speaker Notes:** "The mechanism is deliberately simple. Clip to bound sensitivity. Average to smooth. Inject Laplace noise calibrated to the sensitivity. Then a hysteresis gate requires 5 consecutive breaches before the agent acts. Total latency: 0.2 milliseconds."

---

## Slide 8: The CISO Objection

**Title:** "Why Not Just Use a Wider Kalman Margin?"

**Visual:** Side-by-side. Left: Kalman with wider margin (moved threshold line, attacker still finds it). Right: DP-Governor (fuzzy threshold zone).

**Speaker Notes:** "I get this question every time. Here's the answer: a wider margin just moves the goalpost. The attacker probes at the new threshold and still gets 100% success. You can push the threshold to infinity and the attacker will find it. Differential Privacy doesn't move the goalpost; it makes the goalpost a quantum superposition. The attacker doesn't know where it is because IT doesn't know where it is until the noise is sampled."

---

## Slide 9: The Attacker Burn Rate

**Title:** 17.4% Per-Probe Burn Rate

**Visual:** Table 1 metrics, highlighted: DP probing = 0.826 +/- 0.108. Below: "P(survive 5 sequential probes) = 38.5%"

**Speaker Notes:** "Let's do the math. 82.6% probe success means 17.4% of probes fail to trigger the agent. Each probe independently risks SOC detection, so over 5 probes, the attacker survives undetected with probability (0.826)^5 = 38.5%. This isn't a wall; it's a tax. Every probe costs the attacker."

---

## Slide 10: The Epsilon Tuning Knob

**Title:** You Control the Tradeoff

**Visual:** Plot 4 (Epsilon Sweep) with Goldilocks Zone highlighted. Dual Y-axis showing burn rate climbing as epsilon decreases.

**Speaker Notes:** "Epsilon is your dial. Lower epsilon means more noise, harder to probe, but more jitter in the signal. Higher epsilon approaches deterministic behavior. But here's the critical finding: below epsilon 0.5, the agent goes completely blind. FNR hits 100% because the noise floor pushes the threshold above the maximum possible filtered value. The agent can't detect ANYTHING. So the Goldilocks zone, epsilon 0.5 to 2.0, isn't arbitrary. It's the only region where FNR is zero AND probing resistance is meaningful. Your SOC team picks within that range."

---

## Slide 11: Not Just One Magic Number

**Title:** The Defense Generalizes Across Probe Margins

**Visual:** Plot 6 (Margin Sweep). DP line curves upward; deterministic filters flat at 100%.

**Speaker Notes:** "A reviewer asked: what if the attacker probes at 0.1% above threshold instead of 0.5%? Answer: the defense gets STRONGER. At 0.1% margin, the attacker only succeeds about 53% of the time. The tighter they probe, the more the noise dominates. Deterministic filters? Still 100%. Every margin, every time."

---

## Slide 12: Even Smart Attackers Lose

**Title:** Adaptive Attacker vs. SOC: A Race the Attacker Loses

**Visual:** Plot 5 (Adaptive Attacker). Blue line (uncertainty) falling, orange line (SOC alert) rising. Crossing point annotated.

**Speaker Notes:** "The ultimate stress test. An adaptive attacker using binary search to narrow their estimate. They get smarter with each probe. But each probe also risks detection. The orange line (cumulative SOC alert probability) crosses the blue line (attacker uncertainty) within the first 5 probes. The SOC wins the race. By the time the attacker knows enough to exploit, they've already been caught."

---

## Slide 13: Production Viability

**Title:** 0.2ms. <0.001% False Alarms. Three Diverse Traces.

**Visual:** Table 2 (Multi-Trace) showing consistent results across ec2_cpu, rds_cpu, elb_req. Latency comparison bar chart.

**Speaker Notes:** "This isn't an academic exercise. 0.2ms for 4032 datapoints. Faster than SMA. Faster than Kalman. <0.001% spurious triggers via hysteresis. Tested on CPU utilization, database CPU, and load balancer request counts. The numbers hold."

---

## Slide 14: DP as a Control Theory Regularizer

**Title:** Why We Don't Care About Global Epsilon (And Neither Should You)

**Visual:** Two-column comparison. Left: "Traditional DP": global epsilon budget, protect user identity, infinite time horizon, database queries. Right: "DP-Governor": event-level noise, blind the attacker per-probe, burn rate compounds geometrically, control theory governor. Red X over "Traditional DP" column for SecOps context.

**Content:**
- We intentionally abandon global epsilon bounds. Traditional DP obsesses over protecting user identities over infinite time horizons. That paradigm is useless for SecOps.
- We repurpose DP as a localized, event-level stochastic regularizer. We only care that the noise floor at t_0 forces the attacker's probe into the SOC's visibility threshold.
- "But what about Randomized Smoothing?" RS certifies a radius for a single static classification. We need sequential burn rate over iterative probes, a fundamentally different guarantee.
- An attacker with FULL knowledge of epsilon, clip bounds, and window size can compute the noise distribution. They still can't predict the realization.
- Production deployment requires EWMA Z-score for diurnal drift and periodic re-calibration.
- Not intended for sub-millisecond systems (HFT, inline packet inspection).

**Speaker Notes:** "This is where I preempt the academic objection. Someone will say: 'Your epsilon budget blows up over 4000 timesteps.' Yes. We know. We don't care. Traditional DP is designed to protect a database from a query adversary over infinite time. We are not protecting a database. We are blinding a reconnaissance attacker at each individual probe. The noise is fresh every timestep. The burn rate is 17.4% per probe. That compounds geometrically regardless of what the global epsilon looks like. And if someone says 'this is just Randomized Smoothing': no. RS gives you a certified ball around a single prediction. We give you a sequential detection guarantee over an iterative probing campaign. Different threat model, different math, different guarantee."

---

## Slide 15: Call to Action

**Title:** The Paranoid Agent Is Watching

**Visual:** QR code to GitHub repo (https://github.com/asdfghjkltygh/paranoid-agent). Terminal screenshot of --demo mode output.

**Content:**
- Full PoC: `pip install -r requirements.txt && python dp_governor_poc.py`
- Terminal demo: `python dp_governor_poc.py --demo`
- White paper, slide deck, and all evaluation artifacts included
- MIT licensed

**Speaker Notes:** "The code is open. The data is real. Clone the repo, change the parameters, try to break it. If you can get deterministic probing above 65% without epsilon > 3, I owe you a drink. Thank you."
