#!/usr/bin/env python3
"""Pure-Python teaching examples, NOT a RankMixer/Flood implementation.

Run: python3 introduce/rankmixer_v6_e2_small_3_learning_demo.py
No network, third-party dependencies, files, or production state are used.
The BCE example intentionally omits the production epsilon/logit clipping.
The sparse update is ordinary SGD solely to teach the mechanics.
"""

import copy
import math


def mix(tokens, heads):
    width = len(tokens[0])
    assert width % heads == 0
    chunk = width // heads
    return [
        [value for row in tokens for value in row[h * chunk:(h + 1) * chunk]]
        for h in range(heads)
    ]


def revert(mixed, token_count):
    chunk = len(mixed[0]) // token_count
    return [
        [value for row in mixed for value in row[t * chunk:(t + 1) * chunk]]
        for t in range(token_count)
    ]


def show_mix():
    print('1. Mixing / Reverting：只改变坐标位置')
    tokens = [['a0', 'a1', 'a2', 'a3'], ['b0', 'b1', 'b2', 'b3']]
    mixed = mix(tokens, heads=2)
    restored = revert(mixed, token_count=2)
    print('输入：', tokens)
    print('Mix： ', mixed)
    print('还原：', restored)
    assert restored == tokens
    print('逆变换核对通过；这一步没有可训练参数。\n')


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def forward(state, keys, label):
    # Sum pooling preserves repeated keys in this explicitly defined toy.
    pooled = [sum(state['emb'][key][j] for key in keys) for j in range(2)]
    logit = sum(x * w for x, w in zip(pooled, state['weight'])) + state['bias']
    prediction = sigmoid(logit)
    # Numerically stable ideal sigmoid BCE: softplus(logit) - label * logit.
    loss = max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - label * logit
    return pooled, logit, prediction, loss


def backward(state, keys, label):
    pooled, _, prediction, _ = forward(state, keys, label)
    delta = prediction - label  # This toy has B=1.
    embedding_gradient = {key: [0.0, 0.0] for key in state['emb']}
    for key in keys:
        for j in range(2):
            embedding_gradient[key][j] += delta * state['weight'][j]
    return {
        'weight': [delta * x for x in pooled],
        'bias': delta,
        'emb': embedding_gradient,
    }


def access(state, path):
    obj = state
    for key in path[:-1]:
        obj = obj[key]
    return obj, path[-1]


def finite_difference(state, keys, label, path, epsilon=1e-5):
    plus, minus = copy.deepcopy(state), copy.deepcopy(state)
    obj_plus, key = access(plus, path)
    obj_minus, _ = access(minus, path)
    obj_plus[key] += epsilon
    obj_minus[key] -= epsilon
    return (forward(plus, keys, label)[-1] - forward(minus, keys, label)[-1]) / (2 * epsilon)


def verify_gradient(state, keys, label, gradient):
    paths = [('weight', 0), ('weight', 1), ('bias',)]
    paths += [('emb', key, j) for key in state['emb'] for j in range(2)]
    maximum_error = 0.0
    for path in paths:
        obj, key = access(gradient, path)
        analytical = obj[key]
        numerical = finite_difference(state, keys, label, path)
        maximum_error = max(maximum_error, abs(analytical - numerical))
    assert maximum_error < 1e-8, maximum_error
    print('手算梯度 vs 有限差分：{} 个标量，最大误差 {:.3e}'.format(len(paths), maximum_error))


def sgd_step(state, gradient, dense_lr, sparse_lr):
    updated = copy.deepcopy(state)
    # All updates use the same pre-update state and pre-update gradients.
    updated['weight'] = [w - dense_lr * g for w, g in zip(state['weight'], gradient['weight'])]
    updated['bias'] -= dense_lr * gradient['bias']
    for key in state['emb']:
        updated['emb'][key] = [
            value - sparse_lr * grad
            for value, grad in zip(state['emb'][key], gradient['emb'][key])
        ]
    return updated


def adam_scalar(theta, gradient, learning_rate, step, first_moment, second_moment):
    """The standard TF1-style Adam formula; no claim about Flood internals."""
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
    corrected_lr = learning_rate * math.sqrt(1.0 - beta2 ** step) / (1.0 - beta1 ** step)
    theta -= corrected_lr * first_moment / (math.sqrt(second_moment) + epsilon)
    return theta, first_moment, second_moment


def main():
    show_mix()
    print('2. 一个样本：Embedding sum pooling → Dense 线性层 → sigmoid → BCE')
    state = {
        'emb': {'a': [0.2, -0.1], 'b': [0.1, 0.3], 'unused': [0.7, 0.8]},
        'weight': [0.4, -0.2],
        'bias': -0.1,
    }
    keys, label = ['a', 'b', 'a'], 1.0
    pooled, logit, prediction, loss = forward(state, keys, label)
    gradient = backward(state, keys, label)
    print('keys =', keys, 'label =', label)
    print('pooled embedding =', pooled)
    print('logit = {:.6f}, p = {:.6f}, loss = {:.6f}'.format(logit, prediction, loss))
    print('dL/dlogit = p-y = {:.6f}'.format(prediction - label))
    print('Dense 梯度：W =', gradient['weight'], 'bias =', gradient['bias'])
    print('Sparse 梯度：', gradient['emb'])
    for j in range(2):
        assert gradient['emb']['a'][j] == 2 * gradient['emb']['b'][j]
    assert gradient['emb']['unused'] == [0.0, 0.0]
    print('a 被 sum pooling 使用两次，数据梯度是 b 的两倍；unused 数据梯度为零。')
    verify_gradient(state, keys, label, gradient)

    print('\n3. 教学 SGD 更新：两组参数可使用各自学习率')
    updated = sgd_step(state, gradient, dense_lr=2e-5, sparse_lr=0.05)
    print('Dense W：', state['weight'], '→', updated['weight'])
    print('Sparse a：', state['emb']['a'], '→', updated['emb']['a'])
    print('Sparse unused：', state['emb']['unused'], '→', updated['emb']['unused'])
    new_prediction, new_loss = forward(updated, keys, label)[2:]
    print('更新后 p = {:.6f}, loss = {:.6f}'.format(new_prediction, new_loss))
    assert new_loss < loss
    print('这个样本 label=1，更新后概率升高、loss 降低。')
    print('生产环境用 FloodAdam/PS；上面普通 SGD 只演示梯度怎样改变参数。')

    print('\n4. 同一个 Dense 梯度：标准 Adam 第一步与 SGD 比较')
    theta = state['weight'][0]
    grad = gradient['weight'][0]
    after_adam, m, v = adam_scalar(theta, grad, 2e-5, 1, 0.0, 0.0)
    print('原 W[0] = {:.9f}, 梯度 = {:.9f}'.format(theta, grad))
    print('SGD 后 = {:.9f}, 标准 Adam 后 = {:.9f}'.format(updated['weight'][0], after_adam))
    print('Adam 状态 m = {:.9f}, v = {:.9f}'.format(m, v))
    print('不同优化器对相同梯度产生不同更新；这些数值没有包括生产学习率调度。')


if __name__ == '__main__':
    main()
