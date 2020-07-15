# Report. Continuous Control 


\begin{align}
&\mathbf{\text{Algorithm 1: DDPG algorithm }}\\
&\hline\\
&\text{Randomly initialize critic network $Q(s,a|\theta^Q)$ and actor $\mu(s|\theta^\mu)$ with weights $\theta^Q$ and $\theta^\mu$.}\\
& \text{Initialize target network $Q'$ and $\mu'$ with weights $\theta^{Q'} \leftarrow \theta^Q, \theta^{\mu'} \leftarrow \theta^\mu$ .}\\
&\text{Initialize replay buffer $R$}\\
&\mathbf{ \text{for}} \text{ episode = 1,M } \mathbf{\text{do}} \\
&\quad \text{Initialize a random process $\mathcal{N}$ for action exploration}\\
&\quad \text{Receive initial observation state $s_1$}\\
&\quad \mathbf{\text{for}} \text{ t=1,T } \mathbf{\text{do}}\\
&\quad \quad \text{Select action $a_t = \mu (s_t|\theta^\mu) + \mathcal{N}_t$ acording to the current policy and exploration noise}\\
&\quad \quad \text{Execute action $a_t$ and observe reward $r_t$ and observe new state $s_{t+1}$}\\
&\quad \quad \text{Store transition $(s_t,a_t,r_t,s_{t+1})$ in $R$} \\
&\quad \quad \text{Sample a random mnibatch of $N$ transitions $(s_t,a_t,r_t,s_{t+1})$ from $R$}\\
&\quad \quad \text{Set $y_i = r_i + \gamma Q'(s_{i+1},\mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$} \\
&\quad \quad \text{Update critic by minimizing the loss: $L = \frac{1}{N} \sum_{i} \big(y_i - Q(s_i,a_i|\theta^Q)\big)^2$} \\
&\quad \quad \text{Update the actor policy using the sampled policy gradient: } \\
&\quad \quad \quad \nabla_{\theta^\mu}J \approx \frac{1}{N} \sum_i \nabla_a Q(s,a\|\theta^Q) \big|_ {s=s_i, a = \mu(s_i)}\nabla_{\theta^\mu} \mu(s|\theta^\mu)|_ {s_i} \\
&\quad \quad \text{Update the target network: } \\
&\quad \quad \quad \theta^{Q'} \leftarrow \tau \theta^Q + (1-\tau) \theta^{Q'}\\
&\quad \quad \quad \theta^{\mu'} \leftarrow \tau \theta^\mu + (1-\tau) \theta^{\mu'} \\
&\quad \mathbf{ \text{end for}}\\
&\mathbf{ \text{end for}}\\
\end{align}

## Exemple d'algoritme


\begin{align} 
&\mathbf{\text{Algorithm 1: Depp Q-Learning with Experience Replay }}\\
&\text{Initialize Replay Memory } D \text{ to capacity } N\\
&\text{Initialize action-value function } Q \text{ with random weights } \theta\\
&\text{Initialize target action-value function } \hat{Q} \text{ with weights } \theta^- = \theta \\
&\mathbf{\text{For}} \text{ episode = 1,} M \mathbf{\text{ do}}\\
& \quad \text{Initialize sequence $s_1=\{x_1\}$ and preprocessed sequence $\phi_1 = \phi (s_1)$}\\
& \quad \mathbf{\text{For }} t=1,T \mathbf{\text{ do}} \\
& \quad \quad \text{With probability $\epsilon$ select a random action $a_t$}\\
& \quad \quad \text{otherwise select $a_t = {arg\,max}_a Q(\phi(s_t),a;\theta)$ }\\
& \quad \quad \text{Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$}\\
& \quad \quad \text{Set $s_{t+1}= s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1}=\phi(s_{t+1})$}\\
& \quad \quad \text{Store transition $(\phi_t,a_t,r_t,\phi_{t+1})$ in $D$}\\
& \quad \quad \text{Salple random minibatch of transitions $\big(\phi_j, a_j,r_j, \phi_{j+1}\big)$ from $D$}\\
& \quad \quad \text{Set } y_j = 
\begin{cases}
& r_j & \text{if episode terminates at step j+1} \\
& r_j + \gamma {max}_{a'} \hat{Q}\big(\phi_{j+1},a';\theta^-\big) & \text{otherwise}
\end{cases}\\
& \quad \quad \text{Perform a gradient descent step on $\big(y_j - Q\big(\phi_j,a_j;\theta\big)\big)^2$ with respect to the network parameters $\theta$}\\
& \quad \quad \text{Every $C$ steps reset $\hat{Q}=Q$}\\
& \quad \mathbf{\text{End For}}\\
& \mathbf{\text{End For}}\\
\end{align}


