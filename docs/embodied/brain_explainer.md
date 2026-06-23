# How the connectome LIF brain works — a plain-English explainer

*EB-0A report material. Written for a smart non-specialist (and Vishal). The goal
is intuition, not jargon: by the end you should be able to say what the brain is,
what one neuron does, what "the connectome" buys us, and how we make it think.*

---

## 1. Where the brain sits

We're building an **embodied fly**: a real fly brain, wired the way a real fly's
brain is wired, driving a physically realistic fly body in a simulator. Eon's
four-part loop:

1. **Sense** — a world event (a looming predator, a drop of sugar) activates
   *specific, named* sensory neurons.
2. **Brain** — those spikes propagate through the real wiring; we read out the
   firing rates of any neurons we care about. **← this layer is what EB-0A built.**
3. **Descending neurons** — a handful of "command" neurons carry the brain's
   decision down toward the body.
4. **Body** — NeuroMechFly turns those commands into leg/joint motion in MuJoCo;
   the motion changes what the fly senses → back to step 1.

EB-0A is **only the brain** (step 2). It's a wrapper around a *published* model —
we did not invent the brain, we made it easy to address and step.

---

## 2. One neuron = a leaky bucket that occasionally overflows ("LIF")

Every one of the brain's **138,639 neurons** is the same simple unit: a
**Leaky Integrate-and-Fire** (LIF) neuron. Picture a bucket of water:

- **Integrate** — every incoming spike pours a little water in (raises the
  neuron's *membrane voltage*).
- **Leak** — the bucket has a hole; left alone, the water drains back down to a
  resting level. (In the model: voltage decays toward the resting potential with
  a ~20 ms time constant.)
- **Fire** — if enough spikes arrive close enough together that the water crosses
  a **threshold**, the neuron **fires a spike** of its own — and that spike is
  what pours into all the neurons it connects to.
- **Reset** — right after firing, the bucket is emptied back to its reset level
  and ignores input for a brief **refractory** period (~2 ms), then starts over.

That's the whole neuron. The published model's actual constants (from the
Kakaria–de Bivort / Jürgensen / Lazar measurements baked into `model.py`):

| quantity | value | plain meaning |
|---|---|---|
| resting / reset voltage | −52 mV | the "empty bucket" level |
| threshold | −45 mV | cross this → fire |
| membrane time constant | 20 ms | how fast the bucket leaks |
| synaptic time constant | 5 ms | how fast one input's kick fades |
| refractory period | 2.2 ms | dead time after a spike |
| synaptic delay | 1.8 ms | travel time before a spike lands downstream |

No bucket is special. The intelligence is **not** in the neuron — it's in **how
the 138,639 buckets are plumbed together.**

---

## 3. The connectome = the plumbing (and it's *real*)

A **connectome** is a map of who connects to whom. The FlyWire project traced an
entire adult fly brain from electron-microscope images and published, for
release **v783**, every neuron and every synapse. Our model loads two tables:

- `Completeness_783.csv` — the **list of neurons** (138,639 of them), each with a
  unique **FlyWire ID**. The row order is the neuron's index in the model.
- `Connectivity_783.parquet` — the **list of connections**: ~**15.1 million**
  synaptic edges, each a (pre-neuron → post-neuron) pair with a strength.

Two facts make this map into a working brain:

**(a) Synapse count = weight.** If neuron A makes *many* synapses onto neuron B,
A's spike gives B a *big* voltage kick; one synapse gives a tiny kick. The model
literally multiplies a per-synapse weight (`w_syn = 0.275 mV`) by the number of
synapses. Strong connection = loud voice.

**(b) Neurotransmitter prediction = sign.** A connection can be **excitatory**
(pushes B *toward* firing, +) or **inhibitory** (pushes B *away*, −). Which one
depends on the neurotransmitter the presynaptic neuron releases. Nobody measured
this for every neuron, so Eckstein et al. (2024) *predicted* it from the imagery;
those predictions set the **sign** of each weight (the parquet's
`Excitatory x Connectivity` column — a signed synapse count). So the weight on
every edge is `sign × count × w_syn`.

That's the entire brain: 138k identical leaky buckets, wired by the real
connectome, with strengths from synapse counts and signs from predicted
neurotransmitters. **Structure is the program.**

---

## 4. How we make it think: activate → propagate → read rates

You don't "run" this brain by pressing play — at rest it is **silent** (we
verified: with no input, exactly 0 neurons fire). You make it think by
**injecting activity at named neurons** and watching what lights up.

- **Address by FlyWire ID.** Every neuron has a real, stable-ish ID. We say
  "neurons 720575940624963786, …" — these *are* specific identified cells (e.g.
  particular sugar-sensing neurons), not anonymous units.
- **Activate.** We attach a **Poisson input** to a chosen set and give it a rate
  in Hz. Poisson just means "random spikes arriving at, on average, N times per
  second" — a stand-in for a sensory drive or an optogenetic light pulse. Higher
  Hz = harder push. (Biologically: this is the knob Eon calls "at what rate
  should a stimulus activate sensory neurons" — a hand-chosen mapping.)
- **Silence** (the opposite manipulation). To test "is neuron X necessary?", we
  **zero all of its synapses** so it can fire but influences no one — the model's
  way of knocking a neuron out.
- **Propagate.** Run the simulation forward. Spikes from the activated neurons
  flow along the connectome, raising and lowering downstream buckets; some cross
  threshold and fire, and the activity ripples outward through the real wiring.
- **Read rates.** We count each neuron's spikes over the time window and divide by
  the window length → a **firing rate in Hz** for any neuron we ask about. That
  rate is the brain's "answer."

That four-verb loop — **activate, silence, propagate, read** — is the entire
interface. EB-0A wraps it as `BrainModel`:

```python
brain = BrainModel.load("783")            # load the v783 connectome
brain.activate(sugar_grn_ids, hz=150)     # drive the sugar neurons
rates = brain.run(1.0)                     # -> {flywire_id: firing_rate_hz}
```

---

## 5. The worked example — the proof it's wired right (sugar → feeding)

The cleanest test of "did we wire a *real* brain or just a random graph?" is to
reproduce a **known biological pathway**. Sugar is appetitive: taste it and the
fly extends its proboscis to feed. So activating **sugar gustatory receptor
neurons (GRNs)** should drive the **feeding circuit**, which we read out at the
feeding/proboscis motor neuron **MN9**.

What we got (v783, 150 Hz drive, 1 s windows):

| condition | MN9 firing rate | neurons active |
|---|---|---|
| **control** (no activation) | **0.0 Hz** | 0 |
| **activate 20 sugar GRNs** | **79.7 ± 3.3 Hz** (n=3) | ~370–380 |

MN9 goes from dead silent to firing ~80 Hz **purely because** the sugar neurons'
spikes found their way through the real connectome to the feeding motor neuron.
~370 neurons light up (matching the upstream paper's "~400 active"). That causal
jump — silent → feeding-motor-neuron firing — is the proof the brain is wired
correctly. (Script: `validate_feeding.py`; numbers: `validation_result.json`.)

One honest caveat surfaced here: the sugar IDs come from the older **630**
release of the example; **20 of 21 still resolve in v783** (FlyWire renumbers
neurons between releases), and the one dropped neuron doesn't change the result.

---

## 6. What this buys the next phase (EB-1), and how stepping works

EB-1 needs to run the brain **in lockstep with the body** — advance the brain a
little, send a motor command, let the body move, feed the new sensation back,
repeat (Eon syncs every ~15 ms). Two things make that possible here:

- **The brain remembers between calls.** The model keeps one live network alive,
  and Brian2 carries every neuron's voltage from one `run()` to the next. So
  `brain.run(0.015)` called repeatedly **advances the same brain** in 15 ms
  windows instead of restarting it — each call returns the rates *measured in
  that window*. (`brain.step(0.015)` is the same thing, named for the loop.) We
  verified state carries: stepping 0.3 s then 0.1 s lands the clock at 0.4 s with
  no rebuild.
- **The caveat to design around.** Today the activation *rate* is fixed when the
  network is built (Poisson inputs are baked in). For EB-1, where the looming
  stimulus's intensity changes every window, the rate needs to change on the fly.
  Two clean ways to do that without touching the LIF: (a) **short-window
  re-runs** — rebuild the inputs at the new rate each window and keep advancing
  (cheap: the heavy compile is cached, so a rebuilt 15 ms step is fast); or (b)
  swap the fixed `PoissonInput` for a runtime-settable rate source
  (`PoissonGroup`/`TimedArray`) — a small input-layer change, still no change to
  the neuron model. EB-1 should pick one early; the API is built so either fits.

---

## 7. Honest limitations (carry these, like Eon does)

- **LIF is a cartoon neuron.** It omits dendritic computation, diverse ion
  channels, plasticity (no learning — the wiring is frozen), internal state, and
  neuromodulation. Real neurons do more.
- **The wiring is a snapshot.** One connectome, predicted (not measured)
  neurotransmitter signs, synapse-count weights — all fixed.
- **Rates, not meaning.** We read firing rates; turning a rate into a "decision"
  or a motor command is the *coupling* we build later, and it's hand-tuned.
- **Activation is a stand-in.** Poisson drive at a chosen Hz approximates a real
  sensory input; the magnitude→Hz mapping is a design choice, not a measurement.
- This is a **research/demonstration platform**, not a biologically complete fly.
  "Structure → behavior" is a direction we explore, not a proven claim.

---

### One-paragraph version

The brain is 138,639 identical *leaky integrate-and-fire* neurons — each a bucket
that fills with incoming spikes, leaks back toward rest, and fires when it
overflows — wired together by the **real FlyWire v783 connectome**: ~15 million
synapses where the *number* of synapses sets a connection's strength and the
*predicted neurotransmitter* sets its sign (excite or inhibit). It's silent until
you **activate** specific, ID-addressed neurons with a Poisson drive at some Hz;
the spikes **propagate** through the real wiring and you **read** the resulting
firing rates of any neurons you like. We proved it's wired correctly by
reproducing a textbook pathway — activating sugar-taste neurons makes the feeding
motor neuron MN9 jump from 0 to ~80 Hz — and we kept the network *persistent* so
the next phase can step the brain in 15 ms windows alongside the body.
