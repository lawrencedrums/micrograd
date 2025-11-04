from micrograd.engine import Value


def mean_squared_loss(gt: list[Value], pred: list[Value]) -> list[Value]:
    return [(pred_i - gt_i)**2 for pred_i, gt_i in zip(pred, gt)]


def hinge_loss(gt: list[Value], pred: list[Value]) -> list[Value]:
    return [(1 + -gt_i*pred_i).relu() for pred_i, gt_i in zip(pred, gt)]
