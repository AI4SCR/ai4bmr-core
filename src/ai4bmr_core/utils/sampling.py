import pandas as pd


def sample_min_per_group_then_uniform(
    grouped: pd.DataFrame.groupby, n=10000, min_per_group: int = None, random_state=None
) -> pd.DataFrame:
    """
    Tries to sample min_per_group samples from each group and then samples uniformly from the remaining samples.
    If the number of samples in a group is less than min_per_group, all samples are taken from that group.

    Args:
        grouped: grouped DataFrame
        n: number of samples
        min_per_group: minimum number of samples per group, default is n // grouped.ngroups
        random_state: random state

    Returns:
        sub-sampled dataframe with approx. n samples
    """
    if min_per_group is None:
        min_per_group = n // grouped.ngroups
    else:
        assert min_per_group <= grouped.size().min()

    # compute number of samples per group
    a = grouped.size()[min_per_group - grouped.size() >= 0]
    # groups with more samples than min_per_group
    b = grouped.size()[min_per_group - grouped.size() < 0]
    # from these groups we sample according to their proportion
    b = (n - a.sum()) * b / b.sum()
    b = b.astype(int)
    c = pd.concat((a, b))

    sampled = pd.DataFrame()
    for grp_name, num_samples in c.items():
        sampled = pd.concat(
            (
                sampled,
                grouped.get_group(grp_name).sample(
                    num_samples, random_state=random_state
                ),
            )
        )

    assert sampled.index.duplicated().any() == False
    return sampled
