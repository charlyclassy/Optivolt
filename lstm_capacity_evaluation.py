"""Calculate nine-criteria breach classification scores from held-out truth/predictions.

Requires matched arrays, actual capacity in kW, and documented evaluation scope.
No training durations are inferred. All percentages are 0..100.
"""
import numpy as np


def capacity_metrics(actual, predicted, capacity_kw):
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if a.shape != p.shape or not np.isfinite(capacity_kw) or capacity_kw <= 0:
        raise ValueError('Unmatched predictions or invalid capacity')
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if a.size == 0:
        raise ValueError('No held-out observations')
    a_pos, p_pos = a >= capacity_kw, p >= capacity_kw
    tp = int(np.sum(a_pos & p_pos)); fp = int(np.sum(~a_pos & p_pos))
    fn = int(np.sum(a_pos & ~p_pos)); tn = int(np.sum(~a_pos & ~p_pos))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
    return {'test_f1_score': f1, 'test_balanced_accuracy': (recall+specificity)/2 if recall is not None and specificity is not None else None,
            'test_breach_recall': recall, 'test_false_alarm_ratio': fp/(tp+fp) if tp+fp else None,
            'risk_capacity_kw': float(capacity_kw), 'risk_tp': tp, 'risk_fp': fp, 'risk_fn': fn, 'risk_tn': tn,
            'risk_evaluated_points': int(a.size)}
