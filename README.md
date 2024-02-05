# -SIR-SIR-model-with-stochasticity-SEIR-SEIR-model-with-stochasticity-SI-SIRS-SEIRS-V-SIRD-
# 项目介绍
这个项目旨在通过利用传染病模型，结合实际观测数据，实现对传染病传播过程的更准确预测。我们采用了多种经典传染病模型，包括SIR、SIR模型带有随机性、SEIR、SEIR模型带有随机性、SI、SIRS、SEIRS-V以及SIRD，并通过优化算法对模型参数进行调整，以最好地拟合现实世界的数据。
# 关键词
SIR、SIR model with stochasticity、SEIR、SEIR model with stochasticity、SI、SIRS、SEIRS-V、SIRD
# 项目思路
整个项目的实现逻辑可以分为以下几个关键步骤：
1. 数据加载与处理：
  - 从CSV文件中加载实际观测数据，包括累积确诊、治愈和死亡数据。
  - 进行数据预处理，如索引设置和数据类型转换。
2. 传染病模型定义：
  - 实现多种传染病模型，包括SIR、SIR模型带有随机性、SEIR、SEIR模型带有随机性、SI、SIRS、SEIRS-V以及SIRD等。
  - 每个模型都有相应的微分方程描述传播过程。
3. 时间序列扩展：
  - 通过扩展时间序列，使其覆盖预测范围，以匹配模型预测的时间点。
4. 优化参数：
  - 定义损失函数，用于衡量模型预测与实际数据的差异。
  - 利用数学优化算法（例如L-BFGS-B）调整模型参数，以最小化损失函数。
5. 模型预测：
  - 使用优化后的参数，结合传染病模型和ODE求解器，对未来一定时间范围内的传播过程进行预测。
  - 得到预测结果，包括感染者、康复者和死亡者的数量。
6. 结果可视化：
  - 将实际数据和模型预测结果可视化，以直观展示模型拟合效果。
  - 使用Matplotlib等工具创建图表，包括感染者、康复者和死亡者随时间的变化趋势。
# 具体代码
## SIR
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/023fa72e6df54dc285830eb68e6d700f.png)

```python
#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

predict_range = 200
s_0=99998
i_0=2
r_0=0
class Learner(object):
    def __init__(self, loss, predict_range, s_0, i_0, r_0):
        self.loss = loss
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
    def load_confirmed(self):
      df = pd.read_csv('02_SZ_DailyCases.csv')
      df.set_index(["Date"], inplace=True)
      dff=df["cummulative confirmed cases"]
      print(dff.T)
      return dff.T

    def load_recovered(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative cured cases"]
        print(dff.T)
        return dff.T

    def load_dead(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative dead cases"]
        print(dff.T)
        return dff.T

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, s_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIR, [0, size],
                                                                                         [s_0, i_0, r_0],
                                                                                         t_eval=np.arange(0, size, 1))

    def train(self):
        recovered = self.load_recovered()
        death = self.load_dead()
        data = (self.load_confirmed() - recovered - death)

        optimal = minimize(loss, [0.001, 0.001], args=(data, recovered, self.s_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, data,
                                                                                                  recovered, death,

                                                                                                  self.s_0, self.i_0,
                                                                                                  self.r_0)
        df = pd.DataFrame(
            {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death,
             'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        # df.to_csv("result_SIR.csv")
        #ax.set_title(self.country)
        df.plot(ax=ax)
        print(f" beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        # fig.savefig("result_SIR.png")


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

    # Here
    # solution = odeint(SIR, [0, size], [s_0, i_0, r_0])
    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)

    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered) ** 2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2


def main():
    learner = Learner(loss, predict_range, s_0, i_0, r_0)
    learner.train()


if __name__ == '__main__':
    main()
```

## SIR model with stochasticity
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1fe35e7c153c47c9979440f7aa9c66ac.png)

```python
#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
np.random.seed(42)  # for reproducibility
predict_range = 200
s_0=99998
i_0=2
r_0=0
class Learner(object):
    def __init__(self, loss, predict_range, s_0, i_0, r_0):
        self.loss = loss
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
    def load_confirmed(self):
      df = pd.read_csv('02_SZ_DailyCases.csv')
      df.set_index(["Date"], inplace=True)
      dff=df["cummulative confirmed cases"]
      print(dff.T)
      return dff.T

    def load_recovered(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative cured cases"]
        print(dff.T)
        return dff.T

    def load_dead(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative dead cases"]
        print(dff.T)
        return dff.T

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, s_0, i_0, r_0, sigma=0.1):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SIR_stochastic(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            dS = -beta * S * I * dt + sigma * S * np.sqrt(dt) * np.random.normal()
            dI = (beta * S * I - gamma * I) * dt + sigma * I * np.sqrt(dt) * np.random.normal()
            dR = gamma * I * dt + sigma * R * np.sqrt(dt) * np.random.normal()
            return [dS, dI, dR]

        dt = 1.0  # time step for Euler-Maruyama method

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        result = solve_ivp(SIR_stochastic, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1))

        return new_index, extended_actual, extended_recovered, extended_death, result

    def train(self):
        recovered = self.load_recovered()
        death = self.load_dead()
        data = (self.load_confirmed() - recovered - death)

        optimal = minimize(loss, [0.001, 0.001], args=(data, recovered, self.s_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, data,
                                                                                                  recovered, death,

                                                                                                  self.s_0, self.i_0,
                                                                                                  self.r_0)
        df = pd.DataFrame(
            {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death,
             'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        # df.to_csv("result_SIR.csv")
        df.plot(ax=ax)
        print(f" beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        fig.savefig("result_SIR.png")


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

    # Here
    # solution = odeint(SIR, [0, size], [s_0, i_0, r_0])
    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)

    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered) ** 2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2


def main():
    learner = Learner(loss, predict_range, s_0, i_0, r_0)
    learner.train()


if __name__ == '__main__':
    main()
```

## SEIR
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2ea21ebb585745858325fa02b1928deb.png)

```python
#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

predict_range = 250
s_0=99990
e_0=8
i_0=2
r_0=0
ratio=0.5

class Learner(object):
    def __init__(self, loss, predict_range, s_0, e_0,i_0, r_0):
        self.loss = loss
        self.predict_range = predict_range
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0

    def load_confirmed(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff=df["cummulative confirmed cases"]
        return dff.T

    def load_exposed(self,ratio):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff=df["cummulative confirmed cases"]
        dfff=dff*ratio
        return dfff.T


    def load_recovered(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative cured cases"]
        return dff.T

    def load_dead(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative dead cases"]
        return dff.T

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, alpha, gamma, data, exposed, recovered, death, s_0, e_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SEIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return [-beta * S * I, beta * S * I - alpha* E, alpha* E- gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_exposed = np.concatenate((exposed.values, [None] * (size - len(exposed.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_exposed, extended_recovered, extended_death, solve_ivp(SEIR,[0, size],
                                                                                         [s_0, e_0,i_0, r_0],
                                                                                         t_eval=np.arange(0, size, 1))

    def train(self):
        recovered = self.load_recovered()
        exposed = self.load_exposed(ratio)
        death = self.load_dead()
        data = (self.load_confirmed() - exposed - recovered - death)#易感人数

        optimal = minimize(loss, [0.001, 0.001, 0.001], args=(data, exposed, recovered, self.s_0, self.e_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, alpha, gamma = optimal.x
        new_index, extended_actual, extended_exposed, extended_recovered, extended_death, prediction = self.predict(beta, alpha, gamma, data, exposed, recovered, death,

                                                                                                  self.s_0, self.e_0,self.i_0,
                                                                                                  self.r_0)
        df = pd.DataFrame(
            {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death,
             'Susceptible': prediction.y[0], 'Exposed': prediction.y[1], 'Infected': prediction.y[2], 'Recovered': prediction.y[3]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        #ax.set_title(self.country)
        df.plot(ax=ax)
        print(f" beta={beta:.8f}, alpha={alpha:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        fig.show()
        print(df)


def loss(point, data, exposed, recovered, s_0, e_0, i_0, r_0):
    size = len(data)
    beta, alpha, gamma = point

    def SEIR(t, y):
        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]
        return [-beta * S * I, beta * S * I - alpha * E, alpha * E - gamma * I, gamma * I]

    solution = solve_ivp(SEIR, [0, size], [s_0, e_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - exposed) ** 2))
    l3 = np.sqrt(np.mean((solution.y[3] - recovered) ** 2))
    a1 = 0.1
    a2 = 0.1
    return a1 * l1 + a2 * l2 + (1 - a1 - a2) * l3


def main():
    learner = Learner(loss, predict_range, s_0, e_0, i_0, r_0)
    learner.train()


if __name__ == '__main__':
    main()
```

## SEIR model with stochasticity
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/92447cf65dcc471f9b77e8c0378072be.png)

```python
#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

np.random.seed(42)  # for reproducibility

predict_range = 250
s_0 = 99990
e_0 = 8
i_0 = 2
r_0 = 0
ratio = 0.5

class Learner(object):
    def __init__(self, loss, predict_range, s_0, e_0, i_0, r_0):
        self.loss = loss
        self.predict_range = predict_range
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0

    def load_confirmed(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative confirmed cases"]
        return dff.T

    def load_exposed(self, ratio):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative confirmed cases"]
        dfff = dff * ratio
        return dfff.T

    def load_recovered(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative cured cases"]
        return dff.T

    def load_dead(self):
        df = pd.read_csv('02_SZ_DailyCases.csv')
        df.set_index(["Date"], inplace=True)
        dff = df["cummulative dead cases"]
        return dff.T

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, alpha, gamma, data, exposed, recovered, death, s_0, e_0, i_0, r_0, sigma=0.1):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        dt = 1.0  # time step for Euler-Maruyama method

        def SEIR_stochastic(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            dS = -beta * S * I * dt + sigma * S * np.sqrt(dt) * np.random.normal()
            dE = (beta * S * I - alpha * E) * dt + sigma * E * np.sqrt(dt) * np.random.normal()
            dI = (alpha * E - gamma * I) * dt + sigma * I * np.sqrt(dt) * np.random.normal()
            dR = gamma * I * dt + sigma * R * np.sqrt(dt) * np.random.normal()
            return [dS, dE, dI, dR]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_exposed = np.concatenate((exposed.values, [None] * (size - len(exposed.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        result = solve_ivp(SEIR_stochastic, [0, size], [s_0, e_0, i_0, r_0], t_eval=np.arange(0, size, 1))

        return new_index, extended_actual, extended_exposed, extended_recovered, extended_death, result

    def train(self):
        recovered = self.load_recovered()
        exposed = self.load_exposed(ratio)
        death = self.load_dead()
        data = (self.load_confirmed() - exposed - recovered - death)

        optimal = minimize(loss, [0.001, 0.001, 0.001], args=(data, exposed, recovered, self.s_0, self.e_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, alpha, gamma = optimal.x
        new_index, extended_actual, extended_exposed, extended_recovered, extended_death, prediction = self.predict(
            beta, alpha, gamma, data, exposed, recovered, death,
            self.s_0, self.e_0, self.i_0, self.r_0)
        df = pd.DataFrame(
            {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death,
             'Susceptible': prediction.y[0], 'Exposed': prediction.y[1], 'Infected': prediction.y[2],
             'Recovered': prediction.y[3]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        df.plot(ax=ax)
        print(f" beta={beta:.8f}, alpha={alpha:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        fig.show()
        print(df)

def loss(point, data, exposed, recovered, s_0, e_0, i_0, r_0):
    size = len(data)
    beta, alpha, gamma = point

    def SEIR(t, y):
        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]
        return [-beta * S * I, beta * S * I - alpha * E, alpha * E - gamma * I, gamma * I]

    dt = 1.0
    solution = solve_ivp(SEIR, [0, size], [s_0, e_0, i_0, r_0], t_eval=np.arange(0, size, 1))
    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - exposed) ** 2))
    l3 = np.sqrt(np.mean((solution.y[3] - recovered) ** 2))
    a1 = 0.1
    a2 = 0.1
    return a1 * l1 + a2 * l2 + (1 - a1 - a2) * l3

def main():
    learner = Learner(loss, predict_range, s_0, e_0, i_0, r_0)
    learner.train()

if __name__ == '__main__':
    main()

```

## SI
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b85847803d304267b11c7ce87d2721ac.png)

```python
import numpy as np
import pandas as pd
import scipy.optimize as optimization
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract relevant columns
confirmed = data['Confirmed'].values
recovered = data['Recovered'].values
deaths = data['Deaths'].values

# Total population
N = 10000

# Initial infected individuals
I_0 = confirmed[0]
S_0 = N - I_0

# Contact rate (to be optimized)
P = 1

# Transmission rate (to be optimized)
beta = 0.25

# Recovery rate (to be optimized)
gamma = 0

# Time points
T = len(confirmed) - 1
T_range = np.arange(0, T + 1)

# Initial conditions
INI = (S_0, I_0)

# Define the SI model
def funcSI(inivalue, _):
    Y = np.zeros(2)
    X = inivalue
    Y[0] = - (P * beta * X[0] * X[1]) / N + gamma * X[1]
    Y[1] = (P * beta * X[0] * X[1]) / N - gamma * X[1]
    return Y

# Define the objective function to minimize
def objective(params):
    global beta, gamma
    beta, gamma = params
    RES = spi.odeint(funcSI, INI, T_range)
    predicted_infected = RES[:, 1]
    return np.sum((predicted_infected - confirmed) ** 2)

# Initial guess for parameters
initial_params = [beta, gamma]

# Parameter bounds (within 0 and 10)
parameter_bounds = [(0, 100), (0, 10)]

# Perform optimization with bounds
result = optimization.minimize(objective, initial_params, method='L-BFGS-B', bounds=parameter_bounds)

# Get optimized parameters
optimal_params = result.x
beta_optimal, gamma_optimal = optimal_params

# Simulate the SI model with optimized parameters
RES_optimal = spi.odeint(funcSI, INI, T_range)

# Plot the results
plt.plot(RES_optimal[:, 0], color='darkblue', label='Susceptible', marker='.')
plt.plot(RES_optimal[:, 1], color='red', label='Infection', marker='.')
plt.scatter(T_range, confirmed, color='black', label='Actual Confirmed', marker='x')
plt.title('SI Model with Parameter Optimization')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()

# Display optimized parameters
print(f"Optimized Transmission Rate (beta): {beta_optimal}")
print(f"Optimized Recovery Rate (gamma): {gamma_optimal}")

```

## SIRS
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/97c33511661844ec8d5ecca6d6d15533.png)

```python
import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.optimize as optimization
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract relevant columns
confirmed = data['Confirmed'].values
recovered = data['Recovered'].values
deaths = data['Deaths'].values

# Total population
N = 10000

# Initial infected individuals
I_0 = confirmed[0]
R_0 = recovered[0]
S_0 = N - I_0 - R_0

# Transmission rate (to be optimized)
beta = 0.25

# Recovery rate (to be optimized)
gamma = 0.05

# Antibody duration (to be optimized)
Ts = 7

# Time points
T = len(confirmed) - 1
T_range = np.arange(0, T + 1)

# Initial conditions
INI = (S_0, I_0, R_0)

# Define the SIRS model
def funcSIRS(inivalue, _):
    Y = np.zeros(3)
    X = inivalue
    Y[0] = - (beta * X[0] * X[1]) / N + X[2] / Ts
    Y[1] = (beta * X[0] * X[1]) / N - gamma * X[1]
    Y[2] = gamma * X[1] - X[2] / Ts
    return Y

# Define the objective function to minimize
def objective(params):
    global beta, gamma, Ts
    beta, gamma, Ts = params
    RES = spi.odeint(funcSIRS, INI, T_range)
    predicted_infected = RES[:, 1]
    predicted_recovered = RES[:, 2]
    return np.sum((predicted_infected - confirmed) ** 2) + np.sum((predicted_recovered - recovered) ** 2)

# Initial guess for parameters
initial_params = [beta, gamma, Ts]

# Parameter bounds (within 0 and 10)
parameter_bounds = [(0, 10), (0, 10), (1, 140)]  # Assuming Ts must be between 1 and 14 days

# Perform optimization with bounds
result = optimization.minimize(objective, initial_params, method='L-BFGS-B', bounds=parameter_bounds)

# Get optimized parameters
optimal_params = result.x
beta_optimal, gamma_optimal, Ts_optimal = optimal_params

# Simulate the SIRS model with optimized parameters
RES_optimal = spi.odeint(funcSIRS, INI, T_range)

# Plot the results
# plt.plot(RES_optimal[:, 0], color='darkblue', label='Susceptible', marker='.')
plt.plot(RES_optimal[:, 1], color='red', label='Infection', marker='.')
plt.plot(RES_optimal[:, 2], color='green', label='Recovery', marker='.')
plt.scatter(T_range, confirmed, color='black', label='Actual Confirmed', marker='x')
plt.scatter(T_range, recovered, color='blue', label='Actual Recovered', marker='o')
plt.title('SIRS Model with Parameter Optimization')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()

# Display optimized parameters
print(f"Optimized Transmission Rate (beta): {beta_optimal}")
print(f"Optimized Recovery Rate (gamma): {gamma_optimal}")
print(f"Optimized Antibody Duration (Ts): {Ts_optimal} days")

```

## SEIRS-V
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a6d4036874b0430b934203af86a49642.png)

```python
import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.optimize as optimization
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract relevant columns
confirmed = data['Confirmed'].values
recovered = data['Recovered'].values
deaths = data['Deaths'].values

# Total population
N = 10000

# Initial infected individuals
I_0 = confirmed[0]
E_0 = 0
R_0 = recovered[0]
V_0 = 0
S_0 = N - I_0 - E_0 - R_0 - V_0

# Transmission rate (to be optimized)
beta = 0.25

# Incubation rate (to be optimized)
sigma = 0.1

# Recovery rate (to be optimized)
gamma = 0.05

# Vaccination rate (to be optimized)
vaccination_rate = 0.01

# Time points
T = len(confirmed) - 1
T_range = np.arange(0, T + 1)

# Initial conditions
INI = (S_0, E_0, I_0, R_0, V_0)

# Define the SEIRS-V model
def funcSEIRSV(inivalue, _):
    Y = np.zeros(5)
    X = inivalue
    Y[0] = -beta * X[0] * X[2] / N - vaccination_rate * X[0]  # Susceptible
    Y[1] = beta * X[0] * X[2] / N - sigma * X[1]  # Exposed
    Y[2] = sigma * X[1] - gamma * X[2]  # Infectious
    Y[3] = gamma * X[2]  # Recovered
    Y[4] = vaccination_rate * X[0]  # Vaccinated
    return Y

# Define the objective function to minimize
def objective(params):
    global beta, sigma, gamma, vaccination_rate
    beta, sigma, gamma, vaccination_rate = params
    RES = spi.odeint(funcSEIRSV, INI, T_range)
    predicted_infected = RES[:, 2]
    predicted_recovered = RES[:, 3]
    return np.sum((predicted_infected - confirmed) ** 2) + np.sum((predicted_recovered - recovered) ** 2)

# Initial guess for parameters
initial_params = [beta, sigma, gamma, vaccination_rate]

# Parameter bounds (within 0 and 10)
parameter_bounds = [(0, 10), (0, 100), (0, 10), (0, 10)]

# Perform optimization with bounds
result = optimization.minimize(objective, initial_params, method='L-BFGS-B', bounds=parameter_bounds)

# Get optimized parameters
optimal_params = result.x
beta_optimal, sigma_optimal, gamma_optimal, vaccination_rate_optimal = optimal_params

# Simulate the SEIRS-V model with optimized parameters
RES_optimal = spi.odeint(funcSEIRSV, INI, T_range)

# Plot the results
plt.plot(RES_optimal[:, 0], color='darkblue', label='Susceptible', marker='.')
plt.plot(RES_optimal[:, 1], color='orange', label='Exposed', marker='.')
plt.plot(RES_optimal[:, 2], color='red', label='Infectious', marker='.')
plt.plot(RES_optimal[:, 3], color='green', label='Recovered', marker='.')
plt.plot(RES_optimal[:, 4], color='purple', label='Vaccinated', marker='.')
plt.scatter(T_range, confirmed, color='black', label='Actual Confirmed', marker='x')
plt.scatter(T_range, recovered, color='blue', label='Actual Recovered', marker='o')
plt.title('SEIRS-V Model with Parameter Optimization')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()

# Display optimized parameters
print(f"Optimized Transmission Rate (beta): {beta_optimal}")
print(f"Optimized Incubation Rate (sigma): {sigma_optimal}")
print(f"Optimized Recovery Rate (gamma): {gamma_optimal}")
print(f"Optimized Vaccination Rate: {vaccination_rate_optimal}")

```

## SIRD
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8e410f3462154d10824ec70e8e9d689e.png)

```python
import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.optimize as optimization
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract relevant columns
confirmed = data['Confirmed'].values
recovered = data['Recovered'].values
deaths = data['Deaths'].values

# Total population
N = 10000

# Initial infected individuals
I_0 = confirmed[0]
R_0 = recovered[0]
D_0 = deaths[0]
S_0 = N - I_0 - R_0 - D_0

# Transmission rate (to be optimized)
beta = 0.25

# Recovery rate (to be optimized)
gamma = 0.05

# Mortality rate (to be optimized)
mu = 0.01

# Time points
T = len(confirmed) - 1
T_range = np.arange(0, T + 1)

# Initial conditions
INI = (S_0, I_0, R_0, D_0)

# Define the SIRD model
def funcSIRD(inivalue, _):
    Y = np.zeros(4)
    X = inivalue
    Y[0] = -beta * X[0] * X[1] / N  # Susceptible
    Y[1] = beta * X[0] * X[1] / N - gamma * X[1] - mu * X[1]  # Infectious
    Y[2] = gamma * X[1]  # Recovered
    Y[3] = mu * X[1]  # Deceased
    return Y

# Define the objective function to minimize
def objective(params):
    global beta, gamma, mu
    beta, gamma, mu = params
    RES = spi.odeint(funcSIRD, INI, T_range)
    predicted_infected = RES[:, 1]
    predicted_recovered = RES[:, 2]
    predicted_deceased = RES[:, 3]
    return np.sum((predicted_infected - confirmed) ** 2) + np.sum((predicted_recovered - recovered) ** 2) + np.sum((predicted_deceased - deaths) ** 2)

# Initial guess for parameters
initial_params = [beta, gamma, mu]

# Parameter bounds (within 0 and 10)
parameter_bounds = [(0, 10), (0, 1), (0, 10)]

# Perform optimization with bounds
result = optimization.minimize(objective, initial_params, method='L-BFGS-B', bounds=parameter_bounds)

# Get optimized parameters
optimal_params = result.x
beta_optimal, gamma_optimal, mu_optimal = optimal_params

# Simulate the SIRD model with optimized parameters
RES_optimal = spi.odeint(funcSIRD, INI, T_range)

# Plot the results
# plt.plot(RES_optimal[:, 0], color='darkblue', label='Susceptible', marker='.')
plt.plot(RES_optimal[:, 1], color='red', label='Infectious', marker='.')
plt.plot(RES_optimal[:, 2], color='green', label='Recovered', marker='.')
plt.plot(RES_optimal[:, 3], color='purple', label='Deceased', marker='.')
plt.scatter(T_range, confirmed, color='black', label='Actual Confirmed', marker='x')
plt.scatter(T_range, recovered, color='blue', label='Actual Recovered', marker='o')
plt.scatter(T_range, deaths, color='purple', label='Actual Deceased', marker='v')
plt.title('SIRD Model with Parameter Optimization')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()

# Display optimized parameters
print(f"Optimized Transmission Rate (beta): {beta_optimal}")
print(f"Optimized Recovery Rate (gamma): {gamma_optimal}")
print(f"Optimized Mortality Rate (mu): {mu_optimal}")

```

# 项目链接

GitHub：https://github.com/Olivia-account/-SIR-SIR-model-with-stochasticity-SEIR-SEIR-model-with-stochasticity-SI-SIRS-SEIRS-V-SIRD-
# 后记
如果觉得有帮助的话，求 关注、收藏、点赞、星星 哦！
