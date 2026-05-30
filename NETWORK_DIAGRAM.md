# Multilayer Perceptron (MLP) Network Diagrams

## 1. NETWORK ARCHITECTURE OVERVIEW

```
                     HIDDEN LAYER              OUTPUT LAYER
INPUT LAYER         (3 neurons)                (1 neuron)
                   
                    ╔═════════════╗
          ╔────────→║  Neuron 1   ║──────═
          │         ╚═════════════╝      │
          │                              │      ╔═════════════╗
    x₁ ──┤         ╔═════════════╗      ├─────→║  Output     ║──→ ŷ
    │    │        │  Neuron 2   ║      │      ║  Neuron     ║
    │    │        ║             ║──────┤      ╚═════════════╝
    │    │        ╚═════════════╝      │
    └────┤                              │
         │         ╔═════════════╗      │
         └────────→║  Neuron 3   ║──────┘
    x₂ ─────→     ╚═════════════╝


Shape transformations:
x: (1, 2)  →  [Linear layer]  →  z1: (1, 3)  →  [ReLU]  →  h: (1, 3)  →  [Linear]  →  ŷ: (1, 1)
                  W1(2×3)                                        W2(3×1)
```

---

## 2. DETAILED FORWARD PASS COMPUTATION

```
╔════════════════════════════════════════════════════════════════════════╗
║                       FORWARD PASS FLOW                               ║
╚════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: INPUT                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│  Input vector:  x = [1.5, -0.5]                                    │
│  Shape:         (1, 2)  ← 1 sample, 2 features                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: LINEAR TRANSFORMATION (Hidden Layer)                       │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Formula:  z1 = x @ W1 + b1                                         │
│                                                                     │
│  Weights W1 (2×3):                                                  │
│  ┌──────────────┬──────────────┬──────────────┐                     │
│  │   0.76       │  -0.47       │   1.52       │  ← W1 row 1         │
│  ├──────────────┼──────────────┼──────────────┤                     │
│  │   0.54       │   1.67       │  -0.37       │  ← W1 row 2         │
│  └──────────────┴──────────────┴──────────────┘                     │
│                                                                     │
│  Bias b1 (1×3):  [-0.04, 0.88, -0.29]                              │
│                                                                     │
│  Computation:                                                       │
│  z1[0] = 1.5×0.76 + (-0.5)×0.54 + (-0.04) = 0.83                  │
│  z1[1] = 1.5×(-0.47) + (-0.5)×1.67 + 0.88 = -0.21                 │
│  z1[2] = 1.5×1.52 + (-0.5)×(-0.37) + (-0.29) = 2.08                │
│                                                                     │
│  Output:  z1 = [0.83, -0.21, 2.08]                                 │
│  Shape:   (1, 3)                                                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: ACTIVATION FUNCTION (ReLU)                                 │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Formula:  h = relu(z1) = max(0, z1)                                │
│                                                                     │
│  Input:   z1 = [0.83, -0.21, 2.08]                                 │
│                                                                     │
│  ReLU operation:                                                    │
│  ┌─────────────────────────────────────────────┐                   │
│  │ relu(0.83)  = max(0, 0.83) = 0.83   ✓       │  ← Positive kept  │
│  │ relu(-0.21) = max(0, -0.21) = 0    ✗        │  ← Negative → 0   │
│  │ relu(2.08)  = max(0, 2.08) = 2.08  ✓        │  ← Positive kept  │
│  └─────────────────────────────────────────────┘                   │
│                                                                     │
│  Output:  h = [0.83, 0, 2.08]                                      │
│  Shape:   (1, 3)                                                    │
│                                                                     │
│  Why ReLU?                                                          │
│  • Introduces NON-LINEARITY                                         │
│  • Kills negative signals (makes network sparse)                    │
│  • Computationally efficient                                        │
│  • Works better than sigmoid/tanh for deep networks                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: OUTPUT LAYER (Linear Transformation)                       │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Formula:  ŷ = h @ W2 + b2                                          │
│                                                                     │
│  Hidden output:  h = [0.83, 0, 2.08]                               │
│                                                                     │
│  Weights W2 (3×1):           Bias b2:                               │
│  ┌──────────┐                [0.41]                                 │
│  │  -0.51   │  ← w2[0]                                              │
│  ├──────────┤                                                       │
│  │   1.52   │  ← w2[1]                                              │
│  ├──────────┤                                                       │
│  │   0.95   │  ← w2[2]                                              │
│  └──────────┘                                                       │
│                                                                     │
│  Computation:                                                       │
│  ŷ = 0.83×(-0.51) + 0×1.52 + 2.08×0.95 + 0.41                      │
│  ŷ = -0.42 + 0 + 1.98 + 0.41                                        │
│  ŷ = 1.97                                                           │
│                                                                     │
│  Output:  ŷ = [1.97]                                               │
│  Shape:   (1, 1)  ← Final prediction                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. DIMENSION TRACKING

```
┌────────────────────────────────────────────────────────────────────────┐
│                    SHAPE TRANSFORMATIONS                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  INPUT                                                                 │
│  ├─ Shape: (1, 2)                                                     │
│  └─ Meaning: 1 sample, 2 features                                     │
│                                                                        │
│        ↓ [Matrix multiply with W1(2×3)]                               │
│        ↓ [Add bias b1(1×3)]                                           │
│                                                                        │
│  HIDDEN LINEAR (z1)                                                    │
│  ├─ Shape: (1, 3)                                                     │
│  └─ Meaning: 1 sample, 3 neurons' raw outputs                         │
│                                                                        │
│        ↓ [Apply ReLU activation]                                      │
│                                                                        │
│  HIDDEN ACTIVATION (h)                                                 │
│  ├─ Shape: (1, 3)                                                     │
│  └─ Meaning: 1 sample, 3 neurons' activated outputs                   │
│                                                                        │
│        ↓ [Matrix multiply with W2(3×1)]                               │
│        ↓ [Add bias b2(1×1)]                                           │
│                                                                        │
│  OUTPUT (ŷ)                                                            │
│  ├─ Shape: (1, 1)                                                     │
│  └─ Meaning: 1 sample, 1 prediction                                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. WEIGHT & BIAS MATRIX SHAPES

```
┌──────────────────────────────────────┐
│     LAYER 1: INPUT → HIDDEN          │
└──────────────────────────────────────┘

Weight Matrix W1: (2 × 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                 Hidden Neuron 1   Hidden Neuron 2   Hidden Neuron 3
                 ────────────────   ────────────────   ────────────────
Input Feature 1  │     w1[0,0]      │     w1[0,1]      │     w1[0,2]
Input Feature 2  │     w1[1,0]      │     w1[1,1]      │     w1[1,2]

                           ↓

z1 = x @ W1 + b1

[x₁  x₂] @ ┌─────────────┬─────────────┬─────────────┐   ┌───────────────┐
           │   w[0,0]    │   w[0,1]    │   w[0,2]    │   │   b1[0]       │
           ├─────────────┼─────────────┼─────────────┤ + │   b1[1]       │
           │   w[1,0]    │   w[1,1]    │   w[1,2]    │   │   b1[2]       │
           └─────────────┴─────────────┴─────────────┘   └───────────────┘
               (2 × 3)                                         (1 × 3)

        = [z1₀  z1₁  z1₂] shape (1, 3)


┌──────────────────────────────────────┐
│     LAYER 2: HIDDEN → OUTPUT         │
└──────────────────────────────────────┘

Weight Matrix W2: (3 × 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                 Output Neuron
                 ─────────────
Hidden Neuron 1  │   w2[0,0]
Hidden Neuron 2  │   w2[1,0]
Hidden Neuron 3  │   w2[2,0]

                           ↓

ŷ = h @ W2 + b2

[h₀  h₁  h₂] @ ┌─────────────┐   ┌───────────┐
               │   w[0,0]    │   │   b2[0]   │
               ├─────────────┤ + │           │
               │   w[1,0]    │   └───────────┘
               ├─────────────┤
               │   w[2,0]    │
               └─────────────┘
                  (3 × 1)          (1 × 1)

        = [ŷ₀] shape (1, 1)
```

---

## 5. HIDDEN LAYER ACTIVATION VISUALIZATION

```
┌─────────────────────────────────────────────────────────────────┐
│        HOW NEURONS PARTITION THE INPUT SPACE                    │
└─────────────────────────────────────────────────────────────────┘

Input Space: x1 from -5 to 5, x2 from -5 to 5

NEURON 1 ACTIVATION MAP:          NEURON 2 ACTIVATION MAP:
───��─────────────────────          ─────────────────────────
    x2                                  x2
    5  ╔════════════════╗              5  ╭───────────────╮
       ║░░░░░░░░░░░░░░░░║                 │░░░░░░░░░░░░░│
       ║░ High ░░░░░░░░║                 │░ ░High░░░░░░│
    0  ╠════════════════╣              0  ├───────────────┤
       ║░░░░░░░░░░░░░░░░║                 │░ ░░░░░░░░░░░│
   -5  ╚════════════════╝             -5  ╰───────────────╯
       -5        0        5               -5        0        5
                 x1                                  x1

NEURON 3 ACTIVATION MAP:
─────────────────────────
    x2
    5  ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱
       │░ High░░░░░░░░│
    0  ├──────��──────────┤
       │░ ░░░░░░░░░░░│
   -5  ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲
       -5        0        5
                 x1

═══════════════════════════════════════════════════════════════

KEY INSIGHT: Each neuron learns to detect DIFFERENT FEATURES
• Neuron 1: Fires when x1 > 0 AND x2 > 0 (top-right)
• Neuron 2: Fires when x2 > 0 (upper half)
• Neuron 3: Fires when x1 > x2 (diagonal region)

Together, they create COMPLEX DECISION BOUNDARIES!
```

---

## 6. DATA FLOW THROUGH THE NETWORK

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                            │
└──────────────────────────────────────────────────────────────────┘

                        Forward Pass

        Input                Hidden                 Output
        Layer                Layer                  Layer
        ─────                ─────                  ──────

x = [1.5]  ─────W1─────→  z1 = [0.83]  ─relu─→  h = [0.83]
    [-0.5]      +b1          [-0.21]             [0]
                                [2.08]             [2.08]
                                                    │
                                                    ├─W2─→ ŷ = [1.97]
                                                    └─+b2─┘

Shapes:       (1,2) ──(2,3)──→ (1,3) ──(3,1)──→ (1,1)
Parameters:        W1(2×3)          W2(3×1)
                   b1(1×3)          b2(1×1)
```

---

## 7. MATHEMATICAL EQUATIONS

```
┌────────────────────────────────────────────────────────────────────┐
│                  NETWORK MATHEMATICS                               │
└────────────────────────────────────────────────────────────────────┘

LAYER 1 (Input → Hidden):
─────────────────────────
  Linear transformation:
    h_lin = x · W₁ + b₁

  Activation function:
    h = φ(h_lin) = ReLU(h_lin) = max(0, h_lin)

  In code:
    z1 = np.dot(x, W1) + b1      # shape (1, 3)
    h = relu(z1)                 # shape (1, 3)


LAYER 2 (Hidden → Output):
──────────────────────────
  Linear transformation:
    ŷ = h · W₂ + b₂

  No activation (regression) or Sigmoid (classification):
    ŷ = h · W₂ + b₂             # No activation for regression
    ŷ = σ(h · W₂ + b₂)          # Sigmoid for binary classification

  In code:
    y_pred = np.dot(h, W2) + b2  # shape (1, 1)


DIMENSIONS:
───────────
  x      ∈ ℝ^(1×2)     Input samples
  W₁     ∈ ℝ^(2×3)     Input-to-hidden weights
  b₁     ∈ ℝ^(1×3)     Hidden biases
  h_lin  ∈ ℝ^(1×3)     Hidden pre-activation
  h      ∈ ℝ^(1×3)     Hidden activations
  W₂     ∈ ℝ^(3×1)     Hidden-to-output weights
  b₂     ∈ ℝ^(1×1)     Output bias
  ŷ      ∈ ℝ^(1×1)     Final prediction
```

---

## 8. ACTIVATION FUNCTION COMPARISON

```
┌────────────────────────────────────────────────────────────────────┐
│              WHY RELU IS BETTER THAN SIGMOID                       │
└────────────────────────────────────────────────────────────────────┘

SIGMOID: σ(z) = 1 / (1 + e^(-z))
─────────────────────────────────

    σ(z)
    1.0  ╔════════════╗
         ║  ╱╱╱ High ║
    0.5  ╠════╱╱╱════╣
         ║╱╱╱ ║ Low  ║
    0.0  ╚════════════╝
        -5   0   5    z

    Problems:
    ✗ Saturates at extremes (gradient → 0)
    ✗ Slow gradient flow in deep networks
    ✗ Computationally expensive (exponential)


RELU: f(z) = max(0, z)
──────────────────────

    f(z)
    5    ╱╱╱╱╱╱╱
         ╱╱ pos ╱╱
    0   ╠════════ z
        ╱╱ neg ╱╱
   -5   ╱╱╱╱╱╱╱

    Advantages:
    ✓ No saturation (linear for z > 0)
    ✓ Fast gradient flow
    ✓ Computationally cheap (just max)
    ✓ Works well in deep networks


COMPARISON TABLE:
─────────────────────────────────────────────────────────────────
            │  SIGMOID        │  RELU
────────────┼─────────────────┼────────────────────
Output      │  (0, 1)         │  [0, ∞)
Formula     │  1/(1+e^-z)     │  max(0, z)
Derivative  │  σ(z)(1-σ(z))   │  1 if z > 0, else 0
Vanishing   │  YES (extreme)  │  NO
Gradient    │                 │
Computation │  EXPENSIVE      │  CHEAP
Speed       │  SLOW           │  FAST
────────────┴─────────────────┴────────────────────
```

---

## 9. WHY WE NEED MULTIPLE LAYERS

```
┌────────────────────────────────────────────────────────────────────┐
│              LINEAR vs NON-LINEAR NETWORKS                         │
└────────────────────────────────────────────────────────────────────┘

WITHOUT HIDDEN LAYER (Linear Only):
────────────────────────────────────

x ──(W)──→ y

Can only learn LINEAR decision boundaries
     ┌─────────────────┐
     │     CLASS A      │╱
     │  ┌─────────────╱│     Only straight lines
     │  │             │ ╲    can separate classes
     │  │  CLASS B    │  ╲
     └──┴─────────────┘───

Problem: XOR problem UNSOLVABLE!
     ┌──────┬──────┐
     │ A    │ B    │
     ├──────┼──────┤
     │ B    │ A    │
     └──────┴──────┘


WITH HIDDEN LAYER (Non-Linear):
───────────────────────────────

x ──(W1, ReLU)──→ h ──(W2)──→ y

Can learn COMPLEX non-linear decision boundaries
     ╭──────╮╭──────╮
     │  A   │ B  │  │
     │  ╭───┴────╮  │
     │  │   B    │  │
     │  │╭──A──╮ │  │
     ╰──┴┴──────┴─┴──╯

Can solve XOR, circles, any complex pattern!

WHY? The hidden layer TRANSFORMS the space:
Step 1: Hidden neurons partition input space
Step 2: ReLU activation creates non-linearity
Step 3: Output layer combines these features
Result: Complex decision boundaries possible!
```

---

## 10. COMPLETE EXAMPLE WITH NUMBERS

```
┌────────────────────────────────────────────────────────────────────┐
│         REAL COMPUTATION WITH ACTUAL NUMBERS                       │
└────────────────────────────────────────────────────────────────────┘

GIVEN:
──────
x = [1.5, -0.5]

W1 = [0.76   -0.47   1.52]
     [0.54    1.67  -0.37]

b1 = [-0.04, 0.88, -0.29]

W2 = [-0.51]
     [ 1.52]
     [ 0.95]

b2 = [0.41]


STEP 1: z1 = x @ W1 + b1
─────────────────────────
z1[0] = 1.5×0.76 + (-0.5)×0.54 + (-0.04)
      = 1.14 - 0.27 - 0.04
      = 0.83

z1[1] = 1.5×(-0.47) + (-0.5)×1.67 + 0.88
      = -0.705 - 0.835 + 0.88
      = -0.66

z1[2] = 1.5×1.52 + (-0.5)×(-0.37) + (-0.29)
      = 2.28 + 0.185 - 0.29
      = 2.175

Result: z1 = [0.83, -0.66, 2.175]


STEP 2: h = relu(z1)
──────────────────────
h[0] = max(0, 0.83) = 0.83
h[1] = max(0, -0.66) = 0        ← Negative killed!
h[2] = max(0, 2.175) = 2.175

Result: h = [0.83, 0, 2.175]


STEP 3: ŷ = h @ W2 + b2
───────────────────────
ŷ = 0.83×(-0.51) + 0×1.52 + 2.175×0.95 + 0.41
  = -0.423 + 0 + 2.066 + 0.41
  = 2.053

Result: ŷ = [2.053]

═════════════════════════════════════════════════════════════════
FINAL PREDICTION: 2.053
═════════════════════════════════════════════════════════════════
```

---

## Summary

This notebook teaches you how a **Multilayer Perceptron (MLP)** works:

1. **Input Layer**: Raw features (2 values)
2. **Hidden Layer**: 3 neurons that learn intermediate features with ReLU
3. **Output Layer**: Combines hidden features into final prediction

The key insight: **Hidden layers with non-linear activations transform data into new spaces where complex patterns become separable.**

