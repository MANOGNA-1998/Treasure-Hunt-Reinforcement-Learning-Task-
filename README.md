Defining RL Environment
Theme: The Adventures of Tintin
Scenario Description:
The Adventures of Tintin is a treasure hunting grid world environment in which Tintin, the
agent, navigates through the grid that is filled with treasures, traps and a goal. Tintin should
collect all the treasures while avoiding the traps and then reach the goal to complete his
adventure.


I. Components of the Environment:
1. States:
The grid world consists of 5 x 5 grid with the following positions:
• {S1=(0,0), S2=(0,1)…S25=(4,4)}
• There are totally 25 states in the grid.
• These states represent Tintin’s position in the grid and treasures he has collected
so far.
2. Actions:
Tintin can perform the following actions:
• Up (1): Move one step up
• Down (2): Move one step down
• Right (2): Move one step to the right
• Left (3): Move one step to the left.
Hence, Tintin can take 4 different actions.
3. Rewards:
• +30 for collecting the treasure for the first time.
• -10 for stepping onto a trap.
• -5 for revisiting previously visited tiles – this is to discourage aimless wandering.
• +2 for moving closer to the uncollected treasure∗
• -2 for moving away from the uncollected treasure.
• +150 for reaching the goal after collecting all treasures.
• -50 for reaching the goal without collecting all treasures.
There are a total of 3 treasure and trap tiles each and one goal tile.
To encourage Tintin to collect all rewards before reaching the goal, a negative reward of -50
is awarded for reaching the goal prematurely whereas, +150 is given when the goal is
reached after collecting all treasures.
* for each of the uncollected treasure, we’re computing the Manhattan distance between the
agent’s current position and treasure’s position. We track the minimum among the three.
We’re using a variable self.previous_distance to store the Manhattan distance from the
previous step. We compare the new minimum distance with self.previous_distance. If Tinin
is moving closer to the nearest uncollected treasure, then he gets a +2 reward, else -2. This is
to encourage Tintin to collect all treasures and reach the goal.
The Manhattan Distance is calculated by:
Distance = |x1 - x2| + |y1 – y2|
Reference: I took the inspiration to use this while I was going though resources to help me understand how to
set up rewards in MDP. This is the source that helped me:
Reward Shaping: https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html
4. Objective:
Tintin’s objective is to collect all treasures scattered across the grid, while avoiding traps
in order to minimize penalties and reach the goal state with all the treasures collected in
order to maximize the rewards.
5. Termination:
The episode ends if either Tintin reaches the goal or maximum number of timestamps is
reached.



II. Visualization of the environment:
I used a custom renderer class – TintinAdventureEnvRenderer for rendering the environment
using images for different elements like- treasure, trap, agent goal, agent and goal, agent and
trap, neutral tile.
Reference: I downloaded the opensource images online for implementing this part. Reference the source of each
image is given in the references section.
Then, the images were loaded and resized. I found the appropriate size and zoom for the image
through some trial and error. For these tasks I used matplotlib and Python Imaging Library
(PIL).
Next, I used a random agent to explore the environment as a test run, to see if the state and
actions are behaving as expected, running through a single episode of max 100 steps.
A sample of the execution for visualization:
Episode 1 starts
Action: 1, Reward: 2, Total Reward: 2



III. Safety in AI
To ensure safety in Adventures of Tintin grid environment, all actions of agent are
bounded within by environment’s defined state-space using clipping mechanism (using
np.clip), to prevent the agent from moving outside of grid boundaries. The action space
is explicitly defined using spaces.Discrete(4), ensuring that the agent chooses only from
vaild actions – Up, Down, Left, Right. Rewards and Penalties are logically designed to
encourage meaningful exploration by Tintin and also discouraging unsafe and redundant
behaviour. The is also a check on possibility of infinite loop by limiting max_timesteps
to 100 which terminates the episode in case Tintin fails to reach the goal within these
steps. Finally, safeguards like state-reset mechanisms ensure that every episode starts
from a well-defined initial state, maintaining the integrity of the environment and the
agent's exploration.

***************************************************************************************************************************************


Implement SARSA
1. SARSA Method:
SARSA (State-Action-Reward-State-Action) is an on-policy reinforcement ML algorithm
which updates Q-values for state action pairs based on the action the agent selects using the
current policy. Here the greedy policy is epsilon-greedy. In my environment, SARSA was
used to train Tintin where he aims to maximize the total reward by collecting treasures,
avoiding traps and finally reaching the goal.
Update Function:
The update rule in SARSA is:
𝑄(𝑆, 𝐴) ← 𝑄(𝑆, 𝐴) + ∝ [ 𝑅 + 𝛾𝑄(𝑆" , 𝐴") − 𝑄(𝑆, 𝐴) ]
Where,
𝑸(𝑺, 𝑨): Current Q-value for the state 𝑆 𝑎𝑛𝑑 action 𝐴.
∝ : Learning rate. This determines how much the agent learns from the new information.
𝑹: Reward received after executing 𝐴 𝑎𝑡 𝑆.
𝜸 : Discount Factor. This determines the importance of future rewards compare to
immediate rewards.
𝑸(𝑺" , 𝑨") : Q-value for the next state action pair, which is chosen by the current policy.
The Q-value is updated to move closer to the target:
𝑅 + 𝛾𝑄(𝑆" , 𝐴") − 𝑄(𝑆, 𝐴)
This is done to ensure that the agent update its policy based on both current and the
expected future rewards.
Key Features of SARSA
• On Policy Learning: The agent selects the next action 𝐴" using the same policy that
it is trying to improve that is, epsilon greedy.
• Learning through interaction: The agent learns by interacting with the
environment, collecting rewards, and updating Q-values iteratively.
• Exploration and Exploitation: epsilon-greedy policy ensures the balance between
trying new actions to discover better paths while choosing to exploit the best
known actions based on the learned Q-values.
• Stochastic Nature: refers to environments or systems where outcomes are uncertain
or determined probabilistically, even when the same action is taken from the same
state multiple times. SARSA is well suited for environments that are Stochastic in
nature with transitions and rewards as it adapts based on the current policy.


Advantages:
• It handles Stochastic Environments well because it is advantageous when
transitions or rewards are probabilistic, it leans directly from the actual policy.
• It explicitly evaluates the current policy while updating the Q-values which leads
to more realistic behavior.
• It uses exploration into updates by using the same policy for selecting and
learning actions.


Disadvantages:
• Since it updates Q-values based on the current policy, the convergence is slow.
• If the exploration rate is not decayed appropriately, the agent might never fully
exploit the learned policy.
• The performance depends heavily on the hyperparameters, hence careful tuning is
necessary.
• It updates specific trajectory followed by the agent, which leads to slow
adaptation to the dynamic environment.
Reference: I referred to this source to understand SARSA:
https://www.datacamp.com/tutorial/sarsa-reinforcement-learning-algorithm-in-python

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

SARSA Implementation for my environment:

In Tintin's Adventure, the SARSA algorithm was implemented to train an agent
navigating a grid world environment, collecting treasures while avoiding traps, and
reaching the goal state with maximum rewards.
• The Q-values for all state-action pairs were initialized to 0.
• An epsilon-greedy policy was used to balance exploration and exploitation.
ε=0.001 was initially used.
• The update rule was given according to the formula.
• The agent was trained over 1000 episodes, with a maximum of 50 steps per
episode. Over time, the Q-values converged, leading the agent to consistently
collect treasures and reach the goal.
• During testing, the agent exploited the learned policy (maximizing Q-values) to
achieve a total reward of 232.
• Visualizations were drawn to better understand the results.

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
3. Hyperparameter Tuning:

How Hyperparameters influence performance:
• Learning Rate (α):
A higher learning rate allows the agent to quickly adapt its Q-values but may cause
instability in learning, leading to overshooting. A lower learning rate slows convergence
but ensures stability.
• Discount Factor (γ):
It determines the importance of future rewards. A high value (close to 1) encourages the
agent to prioritize long-term rewards, while a low value makes it focus on immediate
rewards.
• Exploration Rate (ε):
It controls the trade-off between exploration (choosing random actions) and exploitation
(using the learned policy). A high ε enables better exploration in early episodes but may
delay policy convergence. A lower ε favors exploitation.
• Epsilon Decay Rate:
It defines how quickly the exploration rate decreases over episodes. A slower decay
allows more exploration, while a faster decay favors exploitation sooner.
• Number of Episodes and Maximum Steps:
A higher number of episodes gives the agent more opportunities to refine its policy but
increases computational time.
The values that I experimented with in my implementation of SARSA are:
𝐆𝐚𝐦𝐦𝐚 𝐯𝐚𝐥𝐮𝐞(𝛄) : [0.8, 0.9, 0.95]
Epsilon Decay : [0.99, 0.995, 0.999]
Epsilon min(𝛆𝒎𝒊𝒏):[0.01, 0.05]
Epsilon max(𝛆𝒎𝒂𝒙): [0.1, 0.5]
After hyperparameter tuning, the following combination yielded best performance:
𝐆𝐚𝐦𝐦𝐚 𝐯𝐚𝐥𝐮𝐞(𝛄) : 0.95
Epsilon Decay : 0.095
Epsilon min(𝛆𝒎𝒊𝒏): 0.01
Epsilon max(𝛆𝒎𝒂𝒙): 0.5
And the Best Average Reward was : 237.65

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

4. Results and Evaluation:
Total Rewards per Episode:
The total rewards per episode showed significant improvement as training progressed.
Initially, the agent received negative rewards due to exploration and missteps like traps
or not optimal moves. Over time, rewards became consistently higher, reaching nearoptimal
performance after approximately 200 episodes.
Epsilon Decay over Episodes:
The epsilon decay plot highlights how exploration was reduced over episodes. Starting
with a higher exploration rate (ε=0.1), the agent gradually favored exploitation as ε
decayed to its minimum value of ε =0.01.
Final Performance:
• Best Average Reward: 237.65 (which was computed over the last 100 episodes).
• Test Episode Performance: The agent successfully completed the task with a final
reward of 240 and a total of 450 successful episodes out of 1000.


Understanding the Results:
The optimized hyperparameters allowed the agent to prioritize long-term rewards (γ=0.95) by
focusing on reaching the goal while collecting all treasures. A gradual epsilon decay ensured
sufficient exploration in early episodes, preventing local optima, while a small epsilon minimum
promoted policy exploitation in later episodes.
After nearly 200 episodes, rewards stabilized, that indicates the agent had learned an effective
strategy. The final reward of 240 beat the initial runs (232–236), with a success rate of 450/1000
episodes, which highlight the effectiveness of the tuned hyperparameters.

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


Implement N-step Double Q-learning

1. Derive the n-step Double Q-Learning update rule:
Reference: for the below derivation and understanding of concepts, I extensively used: Reinforcement
Learning: An Introduction, Second edition, Richard S. Sutton and Andrew G. Barto
I studied the ideas of n-step returns and bootstrapping to better understand how to do
this derivation:
n-step methods in reinforcement learning (RL) are a generalization of the one-step
Temporal Difference (TD) learning methods. They combine elements of Monte Carlo
methods and TD learning by using n steps of observed rewards to update value estimates.
This allows for a spectrum of methods ranging from plain TD (1-step) to full Monte
Carlo (when n equals the length of the episode).
Bootstrapping refers to the process of updating a value estimate for a state or stateaction
pair using the current estimates of other states or state-action pairs. This approach
relies on approximating future returns rather than waiting until the actual outcomes are
observed.
• let 𝑆( denote the state at time t
• let 𝐴( denote the action taken at time t
• let 𝑅()* denote the reward received after transitioning from 𝑆( to 𝑆()*
• Let 𝛾 be the discount factor
• 𝐿𝑒𝑡 𝑄*(S,A) and 𝑄+(S,A) be two Q-value estimates in double Q-learning
• 𝑇ℎ𝑒 𝑡𝑎𝑟𝑔𝑒𝑡 𝑟𝑒𝑡𝑢𝑟𝑛 𝐺(: (). is computed using n-step returns and bootstrapping
The n step return 𝑮𝒕: 𝒕)𝒏 𝒊𝒔 computed by:
𝐺(: (). = 𝑅()* + 𝛾𝑅()++. . . .+γ01*( R2)0) +
γ0Q32456 (𝑺𝒕)𝒏, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝒏, 𝒂))
Bootstrapping with other Q-table:
• if 𝑄* is the current table being updated:
𝑄;(<=> = 𝑄+ , 𝑄?@>>=.( = 𝑄*
• if 𝑄+ 𝑖𝑠 𝑡ℎ𝑒 𝑐𝑢𝑟𝑟𝑒𝑛𝑡 table being updated:
𝑄;(<=> = 𝑄* , 𝑄?@>>=.( = 𝑄+
Update Rule for Double Q-Learning:
Step 1: Computing the n-step return 𝐺(: ().
𝐺(: (). = Σ 𝛾A1*𝑅()A
.A
BA + γ0Q32456 (𝑺𝒕)𝒏, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝒏, 𝒂))
• The summation Σ 𝛾A1*𝑅()A
.A
BA accounts for the cumulative discounted rewards
over n steps.
• γ0Q32456 (𝑺𝒕)𝒏, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝒏, 𝒂) bootstraps from the n-steps future
state, using the other Q-table to prevent overestimation.
Step 2: Updating the Q-table that we choose (that is 𝑄*or
𝑄+) 𝑓𝑜𝑟 𝑠𝑡𝑎𝑡𝑒 𝑆( 𝑎𝑛𝑑 𝑎𝑐𝑡𝑖𝑜𝑛 𝐴(
𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕 (𝑆(, 𝐴() ß 𝑄?@>>=.((𝑆(, 𝐴() + α(𝐺(: ().-𝑄?@>>=.( (𝑆(, 𝐴())
• 𝛼: Learning rate
• (𝐺(: ().-𝑄?@>>=.( (𝑆(, 𝐴() )represents the temporal difference error
Step 3: Generalizing for n values 1 to 5:
For n=1 (1-step Q learning):
𝐺(: ()* = 𝑅()* + 𝛾Q32456 (𝑺𝒕)𝟏, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝟏, 𝒂))
For n=2 (2-step Q learning):
𝐺(: ()+ = 𝑅()* + 𝛾𝑅()+ + γ+Q32456 (𝑺𝒕)𝟐, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝟐, 𝒂))
Similarly, for n=3,4,5:
𝐺(: ()E = 𝑅()* + 𝛾𝑅()+ + γ+𝑅()E + γEQ32456 (𝑺𝒕)𝟑, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝟑, 𝒂))
Now, generalizing this for any n:
𝐺(: (). = q𝛾A 𝑅()A
.
ABA
+ 𝛾.Q32456 (𝑺𝒕)𝒏, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕)𝒏, 𝒂))
Which is the required equation for update rule for the n-step Double Q-learning
algorithm.
Practical Implementation of n-step Double Q-learning Algorithm for Tintin’s
Adventure grid:
1. n-step Double Q-learning Method:
Double Q-learning is an enhancement of standard Q-learning that is designed to
address the issue of overestimation bias in action-value estimation.
• The action to evaluate is selecting using one Q-table say 𝑄*.
• The value of that action is estimated using another Q-table 𝑄+.
• The updates alternate between the two Q-table with 50% of probability.
Update Rule:
The n- step return is given as:
𝐺!: !$% = 𝑅!$& + 𝛾𝑅!$'+. . . .+γ()&( R*$() + γ(Q+*,-. (𝑺𝒕$𝒏, 𝒂𝒓𝒈𝒎𝒂𝒙𝒂𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕(𝑺𝒕$𝒏, 𝒂))
The rewards 𝑅()* + 𝑅()+ + 𝑅()E+ . . 𝑅(). are observed. Then, bootstrapping is applied
using the alternate Q-table for the n-step table.
The chosen Q table is updated by:
𝑸𝒄𝒖𝒓𝒓𝒆𝒏𝒕 (𝑆(, 𝐴() ß 𝑄?@>>=.((𝑆(, 𝐴() + α(𝐺(: ().-𝑄?@>>=.( (𝑆(, 𝐴())
Key Features:
• There are two Q tables which reduces the risk of overestimation.
• Average of the two Q tables is used for the epsilon greedy policy.
• Bootstrapping method alternates between the two tables to estimate the action values.
• The n-step returns incorporates both Monte-carlo like cumulative rewards and TD
bootstrapping.
Advantages:
• Overestimation bias is reduced by minimizing optimistic value estimates which results in
better convergence.
• The epsilon-greedy policy makes sure that the agent explores efficiently while leveraging
learned policies.
• It supports n-step returns, balancing short and long term learning.
Disadvantages:
• Maintaining two Q-tables doubles the memory and computational requirements.
• Alternating updates leads to slower convergence as compared to single Q-learning.
• It is highly sensitive to hyperparameters 𝛾, 𝛼 𝑎𝑛𝑑 𝑛.
Results:
• The method achieved 831 successful episodes out of 1000, demonstrating effective
learning and policy optimization.
• The final reward in the test episode was 230, confirming stable policy performance.
• The n-step framework enhanced the balance between Monte Carlo and TD methods,
improving long-term decision-making in Tintin Adventure environment.
Influence of Hyperparameters:
• Gamma (𝛾): A lower gamma (0.7) focuses on the short term rewards and is better for
tasks where immediate goals are critical whereas a higher gamma (0.95) is better for long
term rewards.
• Epsilon Decay: Faster decay (0.99) allows the agent to transition to the exploration
quickly, whereas a slower decay (0.999) rate maintains the exploratory behavior for
longer time which prevents the agent from getting stuck in local optimum.
• Lower epsilon min value (0.05) encourages more deterministic behavior in later episodes
that leverages the learned policy whereas higher epsilon (0.3) allows more exploration in
early training which ensures state space coverage.
• A higher number of episodes (like 2000) provides agent with more opportunities to refine
the policy but can lead to vanishing returns in an already well leaned environment.
• Increasing Max Timesteps (say 100), allows the agent to explore deeper within the same
episode which is beneficial to learning.
Best Hyperparameter Combination:
I ran Tintin’s adventure in different set of hyperparameters like:
And observed the best performance in :
Gamma : 0.7
Epsilon Decay : 0.99
Epsilon min : 0.05
Epsilon max : 0.2
Number of episodes : 1000
Max Timesteps: 50
In this setup, Tintin achieved a best average of 240 across the test episodes. This demonstrates a
balanced approach to learning policy achieved by hyperparameters.
Running the model with these set of hyperparameters gave a reward of 240 which was the best
average observed earlier.
Epsilon Decay over Episodes:
This plot shows the epsilon decay over 1000 episodes starting at rate of 0.2 and decaying
exponentially till 0.99 where it reaches the minimum of 0.05 at around 175th episode and
stabilizes.
Results with values of n=[1,2,3,45]
After running the model with n value=[1,2,3,4,5]:
Training for different values of n:
train_with_best_hyperparameters function stored the Q1 and Q2 values for the model trained
with best hyperparameters. These values were used for every value of n between 1 to 5. For each
n, the agent was trained over multiple episodes and the reward points for each episode was
recorded.
Testing the Greedy Policy for each n
For each n after training, the agent’s performance was tested using greedy actions for atleast 10
episodes. The trained Q tables were used to choose these greedy actions. The total rewards from
these was averaged.
By comparing the following metrics, n=2 was determined as the optimal n step for Tintin’s
Adventure Grid:
1. Rewards per episode: Consistently high rewards after convergence. Over time, the
rewards are stabilizing and converging close to maximum reward of 240, showing
agent’s improved learning policy.
2. Epsilon Decay Over Episodes:
The epsilon decay graph shows how the exploration rate reduces over episodes.
It starts at the maximum value (0.2) and gradually decreases to the minimum value (0.05),
promoting more exploitation of the learned policy as training progresses. This controlled
exploration-exploitation tradeoff allows the agent to explore initially but rely on its learned
policy in later stages.
3. Greedy Test Policy:
The graph shows the total rewards obtained during 10 greedy test episodes where the
agent always selects the best action (greedy action) based on the trained Q-values.
All greedy episodes achieve a consistent reward of 240, indicating that the agent has
learned an optimal policy for n=2.This consistency suggests that n=2 allows the agent to
balance short-term and long-term rewards effectively.
These results suggest that n=2 is the optimal value for the environment.


-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

SARSA Vs n-step Q-learning:

I plotted the following graph to compare the performances of SARSA and n-step Qlearning,
considering total rewards per episode during their test performances as a factor.
• n-step double Q-learning shows a significant improvement in rewards hitting the peak as
early as 200 episodes and maintaining a high reward for a few episodes whereas, SARSA
shows a slower improvement rate. It stabilizes well. beyond 600th episode.
• the n-step double learning algorithm adapts quickly, getting high rewards within 200
episodes, but it shows variability and doesn’t maintain the performance consistently. This
could be because of overreliance on multi step look ahead which is sensitive to noise.
• Although SARSA takes time to improve, it is stabilizing well after 600th episode. This
could be because the single step updates make it more resilient against environment
variability that leads to more stable performance.
• Choice of the algorithm hence depends on the environment’s dynamics. For fast
adaptation, n-step double Q learning is optimal while SARSA is suited better for
environment requiring stability and noise tolerance.

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
References:

• https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html
• https://www.datacamp.com/tutorial/sarsa-reinforcement-learning-algorithm-in-python
• Reinforcement Learning: An Introduction, Second edition, Richard S. Sutton and Andrew G. Barto
• https://www.geeksforgeeks.org/sarsa-reinforcement-learning/#
• https://gibberblot.github.io/rl-notes/single-agent/n-step.html
• https://ubuffalomy.
sharepoint.com/personal/avereshc_buffalo_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fav
ereshc%5Fbuffalo%5Fedu%2FDocuments%2F2024%5FFall%5FRL%2F%5Fpublic%2FCourse%20M
aterials%2FRL%20Environment%20Visualization&ga=1

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
Image References:


I downloaded images of Tintin, goal, treasures, trap and neural tiles from the following open sources:
• https://i.pinimg.com/236x/ac/fe/25/acfe2528ff0818525e991e0c6db272b8.jpg
• https://www.shutterstock.com/image-vector/doodle-darger-caution-emblem-warning-260nw-
1111829186.jpg
• https://www.ledr.com/colours/white.jpg
• https://img.freepik.com/premium-vector/simple-doodle-illustration-ship-sailing-sea_327835-10332.jpg
For superimposing the pictures over one another for agent-goal, agent-treasure, agent-trap, I used:
CollageMaker App
