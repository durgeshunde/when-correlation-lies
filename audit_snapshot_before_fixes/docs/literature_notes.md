# Literature Notes

## Primary Paper

**Title:** Information-Theoretic Approach to Detect Collusion in Multi-Agent Games  
**Authors:** Trevor Bonjour, Vaneet Aggarwal, Bharat Bhargava  
**Venue:** Proceedings of Machine Learning Research, Volume 180 (UAI 2022)  
**Pages:** 223–232  
**Date:** August 17, 2022  
**URL:** https://proceedings.mlr.press/v180/bonjour22a.html  
**PDF:** https://proceedings.mlr.press/v180/bonjour22a/bonjour22a.pdf

### Abstract (verbatim)
> Collusion in a competitive multi-agent game occurs when two or more agents 
> co-operate covertly to the disadvantage of others. Most competitive multi-agent 
> games do not allow players to share information, yet players may still collude 
> through their observed actions. This paper provides an information-theoretic 
> approach to detecting colluding agents in a multi-agent game.

### Key Method: Pairwise Net Influence

**Setting:** Fully observable simultaneous game with N agents.

**State S:** Jointly observable game state (e.g., previous round's action profile).

**Individual influence (Section 3.1):**

$$\gamma(i, j) = I(A_i; A_j \mid S) = H(A_j \mid S) - H(A_j \mid A_i, S)$$

Because this is a simultaneous game, I(A_i; A_j | S) = I(A_j; A_i | S), so 
gamma is symmetric.

**Net directed influence (Section 3.2):**

$$\Gamma(i \to j) = \gamma(i,j) - \max_{k \neq i, k \neq j} \gamma(k, j)$$

$$\Gamma(j \to i) = \gamma(j,i) - \max_{k \neq i, k \neq j} \gamma(k, i)$$

**Collusion detection criterion:** Pair (i, j) is flagged if:

$$\Gamma(i \to j) > \tau \quad \text{AND} \quad \Gamma(j \to i) > \tau$$

where τ is a threshold to be set by the analyst.

**Net influence is DIRECTED despite symmetric gamma** because the maximizing 
competitor k differs: the strongest alternative influence on agent j (from 
any k ≠ i, k ≠ j) may differ from the strongest alternative on agent i.

### Experimental Setup in Paper
- Agents: modified Rock-Paper-Scissors game.
- Four agents; one designated colluding pair.
- Collusion policy: pair coordinates to exploit a third agent.
- Ground-truth labels known; detector evaluated on trajectory data.

### Estimator
The paper uses empirical conditional mutual information (plug-in estimator).
The precise treatment of smoothing is not fully detailed in the paper.
This project uses Laplace/Jeffreys smoothing (pseudocount = 0.5) as documented 
in `detectors.py`.

### Key Assumptions
1. Fully observable game: all agents observe the same public state S.
2. Simultaneous action selection: actions are not sequential within each round.
3. Markov assumption: current actions depend only on current state, not history.

### Limitations Acknowledged by Authors
- Method assumes a fully observable game; partially observable settings are harder.
- Finite-sample effects can inflate MI estimates.
- Does not identify the mechanism of collusion, only evidence of coordination.

---

## Related Work

### LLM Agent Collusion (Liu et al.)
**Repository:** https://github.com/andyjliu/llm-agent-collusion  
**Paper:** "Evaluating LLM Agent Collusion in Double Auctions" (Andy Liu et al.)  
**Relevance:** Environment for studying LLM collusion in economic (double auction) 
settings. Uses actual LLM agents (GPT-4, Claude). Does NOT implement pairwise net 
influence. Inspected but not used directly in core experiments due to dependency 
complexity and focus on economic rather than RPS games.

### Apart Research Sprint Resources
- Sprint track: 1.4 — Markets and Collusion.
- Recommended starter game: iterated Rock-Paper-Scissors with 4 agents.
- Key question: Can information-theoretic methods detect collusion while 
  avoiding false positives from shared context?

---

## Information Theory References

**Conditional Mutual Information:**
Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- CMI defined: I(X; Y | Z) = H(X | Z) - H(X | Y, Z) = Σ_z P(z) I(X; Y | Z=z)
- Non-negativity: I(X; Y | Z) ≥ 0 by convexity of log.

**Laplace Smoothing for Entropy Estimation:**
Holste, D., Grosse, I. & Herzel, H. (1998). Bayes' estimators of generalized 
entropies. *Journal of Physics A*.
- Pseudocount of 0.5 corresponds to a Jeffreys prior analog for entropy estimation.
- Reduces bias in sparse-data regimes at the cost of slight systematic 
  underestimation.

**Finite-Sample Bias in MI:**
Paninski, L. (2003). Estimation of Entropy and Mutual Information. *Neural 
Computation*, 15(6), 1191–1253.
- Plug-in MI estimator is upward-biased (overestimates MI in finite samples).
- Bias direction relevant: naive MI may over-flag pairs as correlated.
- Conditional MI bias is more complex; each state slice has its own sample count.
