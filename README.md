# 📌 Forex Hedging Optimization Using Gurobi

## 🎯 Aim

The objective of this project is to **hedge against foreign exchange risk** for a company that must purchase foreign currency at a future time \(t\). By utilizing **forward contracts** and **call options**, the company can strategically mitigate potential losses.

---

## 📝 Problem Description

Facing uncertainty in exchange rates, the company can reduce its risk exposure by employing the following financial instruments:

- **Forward Contracts**: Agreements to purchase a fixed amount of foreign currency at a predetermined future date.
- **Call Options**: Derivatives that provide the right—but not the obligation—to buy foreign currency at a specified strike price.

### Economic Scenarios

We consider **three scenarios** based on varying interest rate conditions:

| Scenario          | Domestic Interest Rate \(r_d\) | Foreign Interest Rate \(r_f\) | USD Strength |
|-------------------|-------------------------------:|------------------------------:|-------------:|
| **1. Strong USD** | 0.08                           | 0.02                          | High         |
| **2. Neutral**    | 0.05                           | 0.05                          | Balanced     |
| **3. Weak USD**   | 0.02                           | 0.08                          | Low          |

---

## 📊 Models

We introduce **three models** to optimize the hedging strategy:

### 1️⃣ Basic Model

This model minimizes the **Value-at-Risk (VaR)** and the excess loss, ensuring that the company's future currency needs are met.

#### **Mathematical Formulation**

$$
\min \ \eta + \frac{1}{1-\alpha} \sum_{s \in \mathcal{S}} \pi^s \, \xi^s
$$

**Subject to:**

$$
\begin{aligned}
\xi^s &\geq L^s - \eta, && \forall s \in \mathcal{S} \quad (1) \\
\xi^s &\geq 0, && \forall s \in \mathcal{S} \quad (2) \\
\gamma_i^s \, (S_T^s - K_i) &\geq 0, && \forall i \in \mathcal{I} \quad (3) \\
h_i^s &\leq y_i, && \forall i \in \mathcal{I} \quad (4) \\
x + \sum_{i \in \mathcal{I}} h_i^s + z^s - w^s &= V^s, && \forall s \in \mathcal{S} \quad (5)
\end{aligned}
$$

**Key Variables:**

- \( x \): Number of forward contracts.
- \( y_i \): Number of call options purchased.
- \( h_i^s \): Number of call options exercised in scenario \(s\).
- \( \gamma_i^s \): Binary variable indicating whether option \(i\) is exercised in scenario \(s\) (1 if exercised, 0 otherwise).
- \( z^s, w^s \): Amount of currency bought/sold at time \(t\).
- \( \xi^s \): Excess loss in scenario \(s\).
- \( \eta \): Value-at-Risk.

---

### 2️⃣ Mixed Model

This model extends the basic formulation by introducing a **penalty** for imbalances between the number of forward contracts and call options.

#### **Mathematical Formulation**

$$
\min \ \eta + \frac{1}{1-\alpha} \sum_{s \in \mathcal{S}} \pi^s \, \xi^s + \lambda \, (\theta^+ + \theta^-)
$$

**Subject to:**

$$
\begin{aligned}
\xi^s &\geq L^s - \eta, && \forall s \in \mathcal{S} \quad (1) \\
\theta^+ - \theta^- &= (1 - \rho)x - \rho \sum_{i \in \mathcal{I}} y_i, && \quad (2) \\
\theta^+, \theta^- &\geq 0, && \quad (3) \\
x + \sum_{i \in \mathcal{I}} y_i &\leq \text{MaxCoverage}, && \quad (4)
\end{aligned}
$$

**Additional Variables:**

- \( \lambda \): Penalty coefficient for the mismatch between forward contracts and call options.
- \( \theta^+, \theta^- \): Positive and negative deviations, respectively, between the number of forward contracts and call options.

---

### 3️⃣ DRO Mixed Model

The **Distributionally Robust Optimization (DRO) model** incorporates additional constraints on the **expected mean and variance** of the exchange rates, making the hedging strategy robust against distributional uncertainty.

#### **Mathematical Formulation**
```python
    Problem Definition
    min {v@r + 1/(1-alpha) * sum[p_i * excess_loss_i] + lambda * (theta_plus + theta_minus)}
    s.t. excess_loss >= loss - v@r
         excess_loss >= 0
         gamma_i * (s_t - k) >= 0
         h_i <= y_i
         x + sum[h_i * gamma_i] + z - w == foreign_currency
         theta_plus - theta_minus == (1 - rho) * x - rho * sum[y_i]
         x + sum[y_i] <= max_coverage
         theta_plus >= 0
         theta_minus >= 0
         sum[p_i * s_i] <= mean(s_t) * (1 + tolerance)
         sum[p_i * s_i] >= mean(s_t) * (1 - tolerance)
         sum[p_i * (s_i - mean(s_t))^2] <= var(s_t) * (1 + tolerance)
         sum[p_i * (s_i - mean(s_t))^2] >= var(s_t) * (1 - tolerance)
         sum[p_i * delta_i] == 1
         sum[delta_i] == 1

         p_i = {fixed_probabilities}
         gamma_i = {0, 1}
         delta_i = {0, 1}

    where:
    loss = x * f + z * s_t - w * s_t + sum[y_i * c_i] + sum[gamma_i * k_i * y_i] - benchmark_cost

    Variables description:
    v@r: Value at Risk
    alpha: Confidence level
    excess_loss_i: Loss of scenario i
    loss: Loss of the portfolio
    h_i: Number of exercised call options of scenario i
    y_i: owned call option of scenario i
    rho: Percentage of forward contracts to hedge the risk
    lambda: Multiplier for the constraint x - sum(y_i)
    theta: Difference between the number of forward contracts and the sum of call options
    theta_plus: Positive part of the difference between x and sum(y_i)
    theta_minus: Negative part of the difference between x and sum(y_i)
    p_i: Probability assigned from the distribution of scenarios
    max_coverage: Maximum amount of cash that can be used to cover the position
    mean(s_t): Expected value of the spot price
    var(s_t): Variance of the spot price
    delta: Probability selection variable
    fixed_probabilities: Discrete set of allowed probability values
```

---

## 🛠️ Implementation (Python + Gurobi)

This project is implemented in **Python using Gurobi** for optimization.

### 📌 Basic Model Implementation

Below is a sample implementation of the basic model:

```python
def basic_model(k, foreign_currency, pi, alpha, coeff, s_t, f, c,
                value_at_risk, benchmark_cost, n_scenarios, env):
    """
    Implements the basic model for forex hedging.
    """
    m = gp.Model("forex_currency_hedging", env=env)
    
    # Decision variables
    x = m.addVar(name="Number of forward contracts", lb=0, vtype=GRB.CONTINUOUS)
    y = m.addVars(len(k), name="Number of call options", lb=0, vtype=GRB.CONTINUOUS)
    z = m.addVars(n_scenarios, name="Currency to buy at time T", lb=0, vtype=GRB.CONTINUOUS)
    w = m.addVars(n_scenarios, name="Currency to sell at time T", lb=0, vtype=GRB.CONTINUOUS)

    # Constraint: Ensure that the future currency requirement is met for each scenario
    for i in range(n_scenarios):
        m.addConstr(x + gp.quicksum(y[j] for j in range(len(k))) + z[i] - w[i] == foreign_currency)
    
    # Objective Function: Minimize the VaR and the weighted excess loss
    objective = value_at_risk + (1 / (1 - alpha)) * gp.quicksum(pi[i] * (z[i] - w[i]) for i in range(n_scenarios))
    m.setObjective(objective, GRB.MINIMIZE)
    
    # Optimize the model
    m.optimize()
    
    return x.X, [y[j].X for j in range(len(k))], m.objVal
```

---

## 📈 Experimental Results

The model has been tested under various economic conditions. **Key metrics include:**

- **Objective Value:** The minimized loss achieved.
- **Average Loss:** The mean loss across scenarios.
- **Standard Deviation of Loss:** Variability in loss.
- **Optimization Time:** Time taken to solve the model.

Detailed results can be found in the appendix of the project repository.

---

### 📩 Contact

For any questions or to collaborate on this project, please open an **issue** or submit a **pull request**. Contributions and feedback are highly welcome! 😊
