import pandas as pd
import math
from collections import Counter


def entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_of_list(ls):
    total_instances = len(ls)
    cnt = Counter(x for x in ls)
    probs = [x / total_instances for x in cnt.values()]
    return entropy(probs)


def information_gain(df, split_attribute, target_attribute):
    df_split = df.groupby(split_attribute)
    nobs = len(df.index) * 1.0
    df_agg1 = df_split.agg({target_attribute: lambda x: entropy_of_list(x)})
    df_agg2 = df_split.agg({target_attribute: lambda x: len(x) / nobs})
    df_agg1.columns = ["Entropy"]
    df_agg2.columns = ["Proportion"]
    new_entropy = sum(df_agg1["Entropy"] * df_agg2["Proportion"])
    old_entropy = entropy_of_list(df[target_attribute])
    return old_entropy - new_entropy


def id3(df, target_attribute, attribute_names):
    cnt = Counter(x for x in df[target_attribute])

    if len(cnt) == 1:
        return next(iter(cnt))

    elif df.empty or (not attribute_names):
        return max(cnt.keys())

    else:
        gainz = [
            information_gain(df, attr, target_attribute) for attr in attribute_names
        ]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute, remaining_attribute_names)
            tree[best_attr][attr_val] = subtree

        return tree


def print_decision_tree(tree, depth=0, parent_name="Root"):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print("  " * depth + f"{parent_name} -> {key}")
            print_decision_tree(value, depth + 1, key)
    else:
        print("  " * (depth + 1) + f"{parent_name} -> {tree}")


# Example usage:
df = pd.read_csv("D:\dataset\playgolf_data.csv")
t = df.keys()[-1]
attribute_names = list(df.keys())[:-1]
tree = id3(df, t, attribute_names)
print_decision_tree(tree)
