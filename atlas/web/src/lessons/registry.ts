import { ComponentType, lazy } from "react";

export type Lesson = {
  slug: string;
  part: string;
  title: string;
  lede: string;
  /** The question from the map this lesson helps answer. */
  question: string;
  component: ComponentType;
};

const L = (f: () => Promise<{ default: ComponentType }>) => lazy(f);

export const PARTS = [
  "Causality by hand",
  "A network is a causal model",
  "Circuits",
  "Features and dictionaries",
];

export const LESSONS: Lesson[] = [
  {
    slug: "seeing-vs-doing", part: PARTS[0], title: "Seeing is not doing",
    lede: "Simpson's paradox, the back-door adjustment, and the one-line difference between P(y | x) and P(y | do(x)) — computed by hand, on numbers you can change.",
    question: "What happens if we intervene, rather than watch?",
    component: L(() => import("./SeeingVsDoing")),
  },
  {
    slug: "d-separation", part: PARTS[0], title: "Reading a graph: d-separation",
    lede: "Chains, forks and colliders. Click nodes to condition on them and watch paths open and close — the rule behind every adjustment you will ever make.",
    question: "When does information flow between two variables?",
    component: L(() => import("./DSeparation")),
  },
  {
    slug: "front-door", part: PARTS[0], title: "The front door and the rules of do-calculus",
    lede: "When the confounder is hidden and no back-door set exists, a mediator can still identify the effect. Derived step by step, checked by the graph at every step.",
    question: "Can we identify an effect when the confounder is never observed?",
    component: L(() => import("./FrontDoor")),
  },
  {
    slug: "counterfactuals", part: PARTS[0], title: "Counterfactuals: abduction, action, prediction",
    lede: "The third rung of the ladder: not what happens if we act, but what would have happened to this one unit. Three steps on a linear model.",
    question: "What would have happened, for this particular case?",
    component: L(() => import("./Counterfactuals")),
  },
  {
    slug: "network-as-scm", part: PARTS[1], title: "A neural network is a causal model",
    lede: "Seven neurons computing (a AND b) OR c. Every neuron is a variable, every weight an arrow, and do() is just overwriting an activation.",
    question: "What does intervening on a neuron mean?",
    component: L(() => import("./NetworkAsScm")),
  },
  {
    slug: "patching", part: PARTS[1], title: "Activation patching, attribution, mediation",
    lede: "Restore one neuron into a broken run, or break one in a working run: they disagree, and both are right. Then why gradients lie on a saturated output, and what integrated gradients fixes.",
    question: "Which parts of the network explain this output?",
    component: L(() => import("./Patching")),
  },
  {
    slug: "probing", part: PARTS[1], title: "Probing is not using",
    lede: "A linear probe finds a concept with 95% accuracy. Steering along the probe's direction barely moves the model. Decodable and causal are different questions.",
    question: "What information is encoded, and does the model use it?",
    component: L(() => import("./Probing")),
  },
  {
    slug: "ioi", part: PARTS[2], title: "The IOI circuit, head by head",
    lede: "“When Mary and John went to the store, John gave a drink to …” A six-head transformer that solves it the way GPT-2 does. Read the attention, ablate heads, decompose the logit.",
    question: "What algorithm does the model implement?",
    component: L(() => import("./Ioi")),
  },
  {
    slug: "circuit-discovery", part: PARTS[2], title: "Finding the circuit: patching, EAP, ACDC",
    lede: "Patch every head and every edge from a corrupted run. Compare the true effects to their gradient estimates, see where attention saturation fools them, and let ACDC prune the graph.",
    question: "How do we scale circuit discovery?",
    component: L(() => import("./CircuitDiscovery")),
  },
  {
    slug: "lenses", part: PARTS[2], title: "Logit lens, J-lens, tuned lens",
    lede: "A two-hop question — the capital of the country where Curie was born. The model thinks “Poland” and never says it. Which lens can see the thought?",
    question: "What is the model thinking without saying it?",
    component: L(() => import("./Lenses")),
  },
  {
    slug: "superposition", part: PARTS[3], title: "Superposition",
    lede: "Five features, two dimensions. Dense data gets two features; sparse data gets all five, at angles. Drag the sparsity and watch the pentagon appear.",
    question: "Why are neurons not the right variables?",
    component: L(() => import("./Superposition")),
  },
  {
    slug: "sae", part: PARTS[3], title: "Sparse autoencoders",
    lede: "Give an SAE the star-shaped cloud and ask for the arrows back. L1 versus TopK, dead latents, and the sparsity–fidelity frontier, with the ground truth on screen.",
    question: "What are the right variables?",
    component: L(() => import("./Sae")),
  },
  {
    slug: "transcoders", part: PARTS[3], title: "Transcoders and attribution graphs",
    lede: "An MLP with four polysemantic neurons, replaced by a sparse transcoder whose latents each do one job — and wire into a graph from input features to output features.",
    question: "How are features computed from features?",
    component: L(() => import("./Transcoders")),
  },
  {
    slug: "crosscoders", part: PARTS[3], title: "Crosscoders and model diffing",
    lede: "One dictionary for a base and a chat model. Latents that only one model uses show up by their decoder norms — and the training objective can invent some that are not real.",
    question: "What did fine-tuning change?",
    component: L(() => import("./Crosscoders")),
  },
];

export const bySlug = (slug: string) => LESSONS.findIndex((l) => l.slug === slug);
