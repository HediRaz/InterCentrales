"""Training utilities."""


def aggregate_loss_dict(agg_loss_dict):
    """Aggregate a dict of losses."""
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key, val in mean_vals.items():
        if len(val) > 0:
            val = sum(val) / len(val)
        else:
            print(f'{key} has no value')
            val = 0
    return mean_vals
