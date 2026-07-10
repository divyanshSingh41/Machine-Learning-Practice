# Decision Trees

## What is Decision Tree?
A Decision Tree is a supervised machine learning algorithm used for classification and regression. It learns a series of if-else rules by splitting the dataset based on feature values.
- A Decision Tree is simply a series of questions. 
- It keeps asking questions until it reaches a final decision.

## Why is it Called a Decision Tree?
Imagine making decisions like this:
Should I carry an umbrella?

          Rain?

         /     \
      Yes       No
       /
    Temperature?
      /       \
    cold     warm

- Each question creates branches.
- Eventually you reach one decision.
- The structure resembles a tree.

## Components of a Decision Tree
- **Node** --> A node is any point in the tree where information is stored. Example:  Age >30
- **Root Node** --> The first node from which the tree begins.
- **Leaf Node** --> A leaf node represents the final outcome, decision, or prediction of the machine learning model.
- **Internal Node** --> Any node that is not the root or a leaf and still performs a split.
- **Branch/Edge** --> Connection between nodes.
- **Splitting Criterion** --> The mathematical rule used to decide the best split. Example: Gini Index, Entropy
- **Pruning** --> Pruning removes redundant decision tree branches to stop overfitting and improve generalization.
- **Impurity** --> A measure of how mixed the classes are in a node.

## How does it work?
1. Start with all data.
2. Find the best feature to split.
3. Divide data.
4. Repeat recursively.
5. Stop when stopping criteria are met.

## Splitting Criteria
`For Classification:`
### Gini Impurity
---
Gini Impurity measures the frequency with which a randomly chosen element from a set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.

$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
Where:
* **$p_i$**: The probability of an element belonging to class $i$.
* **$C$**: The total number of classes.
### 🧮 Gini Impurity Step-by-Step Example

Let's calculate the Gini Impurity for a dataset node containing **10 fruits** representing two classes:
* 🍏 **6 Apples**
* 🍌 **4 Bananas**

#### 1. Calculate Probabilities ($p_i$)
* Probability of Apple ($p_1$) = $\frac{6}{10} = 0.6$
* Probability of Banana ($p_2$) = $\frac{4}{10} = 0.4$

#### 2. Square the Probabilities ($p_i^2$)
* $p_1^2 = 0.6^2 = 0.36$
* $p_2^2 = 0.4^2 = 0.16$

#### 3. Sum the Squared Probabilities ($\sum p_i^2$)
$$\sum p_i^2 = 0.36 + 0.16 = 0.52$$

#### 4. Subtract from 1 ($1 - \sum p_i^2$)
$$Gini = 1 - 0.52 = \mathbf{0.48}$$

The **Gini Impurity** for this node is **0.48**.

---

### 💡 Quick Rule of Thumb
* **Gini = 0.0 (Pure Node):** All items belong to a single class (e.g., 10 Apples, 0 Bananas).
* **Gini = 0.5 (Max Impurity):** Items are split perfectly 50/50 between two classes.

```Python
def calculate_gini(labels: list) -> float:
    """
    Calculates the Gini Impurity for a list of target labels.
    
    Example:
        >>> calculate_gini(['apple']*6 + ['banana']*4)
        0.48
    """
    if not labels:
        return 0.0

    total_samples = len(labels)
    
    # 1. Count frequencies of each class
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        
    # 2. Calculate probabilities, square them, and sum them up
    squared_probabilities_sum = 0.0
    for count in counts.values():
        probability = count / total_samples
        squared_probabilities_sum += probability ** 2
        
    # 3. Subtract the sum from 1
    return round(1.0 - squared_probabilities_sum, 4)
    
# 6 Apples, 4 Bananas
node_samples = ['apple'] * 6 + ['banana'] * 4

gini_impurity = calculate_gini(node_samples)
print(f"Gini Impurity: {gini_impurity}")  # Output: 0.48
```

`For Regression`
## Regression Trees: How Splits Work

A quick, intuitive summary of how regression trees decide where to split — no heavy math required.

---

## 1. The Big Idea

A regression tree keeps splitting the data into smaller groups so that, inside each group, the target values $y$ are as **similar to each other** as possible.

- "Similar" = low variance = low spread.
- Each leaf predicts the **average** of the $y$ values that land in it.

---

## 2. What Makes a "Good" Split?

A good split takes one messy group and turns it into two cleaner groups.

**Example:**

| x | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| y | 2 | 3 | 3 | 8 | 9 | 10 |

If we split at $x \le 3$:

- Left group: `{2, 3, 3}` → all small numbers, tightly packed ✅
- Right group: `{8, 9, 10}` → all large numbers, tightly packed ✅

This is a great split because each side is now much more uniform than the original mixed group.

If we split at $x \le 2$ instead:

- Left: `{2, 3}`
- Right: `{3, 8, 9, 10}` → still messy, values are spread far apart ❌

Worse split, because the right side still mixes small and large values.

---

## 3. The Actual Rule (Simplified)

For each possible split, the tree measures **how spread out** the y-values are on each side using **squared error** (basically: how far each value is from its group's average, squared, then summed).

$$
\text{Spread}(\text{group}) = \sum (\,y_i - \text{average of group}\,)^2
$$

The tree tries every feature and every possible threshold, and picks whichever split gives the **smallest total spread** across both children combined.

> **In one sentence:** the tree picks the split that makes both resulting groups as tightly clustered around their own average as possible.

---

## 4. Why the Average?

Because the average is the single number that minimizes total squared distance to all points in a group. Predicting anything else (a random guess, the max, etc.) would only increase the error. That's why every leaf in a regression tree predicts the mean of its samples.

---

## 5. Quick Comparison of Criteria

| Criterion | What it does | Leaf predicts |
|---|---|---|
| **MSE / SSE** (default) | Minimizes squared differences | Mean |
| **MAE** | Minimizes absolute differences (more robust to outliers) | Median |
| **Friedman MSE** | Same idea as MSE, tuned for boosting models | Mean |

---

## Steps

1. Try splitting on every feature and threshold.
2. For each split, measure how spread out $y$ is in the left and right groups.
3. Pick the split that minimizes total spread.
4. Repeat inside each new group until you hit a stopping rule (max depth, min samples, etc.).
5. Each final leaf predicts the average $y$ of the samples inside it.

That's the entire idea — everything else (Friedman's formula, incremental computation tricks, MAE vs MSE) is just refinement on top of this core rule.

