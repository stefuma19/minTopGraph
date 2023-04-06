def topk(k, data, q, verbose=False):
    df = data.copy(deep=True)

    df['S(r)'] = q.dot(df.T)
    df = df.sort_values(by=['S(r)'], ascending=True)

    if verbose:
        print("Executed top{} query with params={}".format(k, q))

    return df[:k]


def evaluateDominance(data, p_i, dims=2):
    dominators = []
    dominees = []
    incomp = []

    p = data.loc[p_i]

    for idx, r in data.iterrows():
        if idx == p_i:
            continue

        if all([r[d] <= p[d] for d in range(dims)]) and any([r[d] < p[d] for d in range(dims)]):
            dominators.append(r)
        elif all([p[d] <= r[d] for d in range(dims)]) and any([p[d] < r[d] for d in range(dims)]):
            dominees.append(r)
        else:
            incomp.append(r)

    return dominators, dominees, incomp
