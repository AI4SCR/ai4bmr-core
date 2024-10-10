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
    if n > grouped.size().sum():
        return grouped.obj

    if min_per_group is None:
        min_per_group = n // grouped.ngroups
    else:
        assert min_per_group <= grouped.size().min()

    # compute number of samples per group
    a = grouped.size()[min_per_group - grouped.size() >= 0]
    # from the groups with larger size we sample the minimum number
    b = pd.Series(min_per_group, index=grouped.size()[min_per_group - grouped.size() < 0].index)
    # compute the remaining number of samples we can sample
    remaining = n - (a.sum() + b.sum())
    # groups with more samples than min_per_group
    c = grouped.size()[min_per_group - grouped.size() < 0]
    c = c - b

    # from these groups we sample according to their proportion
    c = remaining * c / c.sum()
    c = c.astype(int)
    # combine the min_per_group and the uniform sample size for groups that have more the min_per_group samples
    c = c + b
    assert (c >= min_per_group).all()

    d = pd.concat((a, c))
    assert len(d) == grouped.ngroups
    assert d.index.duplicated().any() == False
    # note: check that the sample size per group is not larget than the group
    s, d = grouped.size().align(d)
    assert (s >= d).all()
    assert d.sum() <= n

    sampled = pd.DataFrame()
    for grp_name, num_samples in d.items():
        sampled = pd.concat(
            (
                sampled,
                grouped.get_group(grp_name).sample(num_samples, random_state=random_state),
            )
        )

    assert sampled.index.duplicated().any() == False
    return sampled
