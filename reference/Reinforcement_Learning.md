# 强化学习
> 强化学习主要问题：具有感知和决策能力的对象，即智能体(Agent)，通过与外界的环境进行交互从而完成某个任务,侧重在线学习并试图在探索-利用（exploration-exploitation）间保持平衡。。

> 智能体通过感知环境的状态(State)产生决策动作(Action),环境从某个初始初始状态s1开始，通过接受智能体的动作来动态改变自身状态，并给出相应的奖励(Reward)信号。

> 主要包含五个对象：状态s(反映环境特征)、动作a(智能体的行为)、策略π(a|s) (代表了智能体的决策模型，在某状态执行某动作的概论)、奖励r(s,a)(表达环境在状态𝑠时接受动作𝑎后给出的反馈信号)、状态转移概率p(s′|s𝑠,a) 表达了环境模型状态的变化规律

> 马尔科夫决策过程(MDP):下一个时间戳的状态s_t+1只与当前的状态s_t和当前状态上执行的动作a_t相关。 如果能够获得环境的状态转移概率P和激励函数R可以直接迭代计算值函数，这种已知环境模型的方法统称为基于模型的强化学习。然而现实世界中的环境模型大都是复杂且未知的，这类模型未知的方法统称为模型无关的强化学习。

> 为了权衡近期奖励与长期奖励的重要性，需要使用随着时间衰减(γ^t)的折扣回报

## 策略梯度思想
> 强化学习的目标是找到某个最优策略π使得期望回报J最大，可以采用梯度上升算法更新网络参数

### REINFORCE 算法
> 根据策略生成多条轨迹,再通过梯度上升算法训练更新参数
+ 缺点：优化轨迹方差很大，为避免之前学到的经验覆盖了，梯度增量都需要非常小，收敛速度较慢，训练过程并不足够平滑。
+ 优化：从因果性（只考虑从时间戳为t开始的累积回报R），基准线（真实环境中的奖励r并不是分布在 0 周围，添加一个基准偏置b使R能够分布在0周围）等方法实现方差缩减
+ 基准线b可以通过蒙特卡罗方法进行估计，也可以采用另一个神经网络来估计。

### 重要性采样
+ 策略梯度方法在更新网络参数后策略网络π发生变化，必须使用新的策略网络进行采样。为提高采样效率，通过重要性采样技术（从另一个分布q乘上比例来估计原分布p的期望） 重用过去的策略轨迹，来估计待优化目标策略的期望。实现时需要保证p、q分布尽可能，可以添加KL 散度约束。
+ 采样策略和待优化的目标策略不同的方法叫做离线(Off Policy)方法；采样策略和当前待优化的策略是同一策略的方法叫做在线(On-Policy)方法。
+ 较流行的离线策略梯度算法有 TRPO 算法和 PPO 算法，PPO可看成TRPO的近似简化版本，其将 KL 散度约束条件作为梯度惩罚项添加进损失函数，减少了计算代价。
+ 自适应 KL 惩罚项中通过设置 KL 散度的阈值来动态调整超参数β。

## 值函数方法
> 通过建模值函数从而间接获得策略，主要围绕 状态值函数和状态-动作值函数
+ 状态值函数V定义为从状态s_t开始，在策略π的控制下能获得的期望回报值，状态值函数的数值反映了当前策略下状态的好坏，最优策略是能取得V最大值的策略。
+ 状态-动作值函数Q 定义为从状态s_t并执行动作a_t的双重设定下，在策略π的控制下能获得的期望回报值。
+ 把Q(s,a)与V(s)的差定义为优势值函数A，表明在状态s下采取动作a比平均水平的差异。A>0说明采取动作a要优于平均水平；反之则劣于平均水平。

### 值函数估计
> 值函数的估计主要有蒙特卡罗方法(通过采样轨迹的总回报来估算期望回报)和时序差分方法(利用了值函数的贝尔曼方程性质,只需要交互一步或者多步即可获得值函数的误差,计算效率更高)。
+ ϵ-贪心法： 1-ϵ概率选择最值，少量几率ϵ采取随机策略，从而去探索未知的动作与状态
+ 在 DQN 算法中使用了经验回放池来减轻数据之间的强相关性。训练时存入轨迹数据，通过随机采样部分数据用于训练。
+ `epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))# epsilon 概率也会 8%到 1%衰减，越到后面越使用 Q 值最大的动作`


