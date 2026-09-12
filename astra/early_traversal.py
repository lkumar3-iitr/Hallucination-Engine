"""Rejected slow experiment; not a runtime backend. Reference sample lattice."""
import torch
import torch.nn.functional as functional


@torch.inference_mode()
def intersect(hull, origin, directions):
    shape = directions.shape[:2]
    ray = torch.as_tensor(directions.reshape(-1, 3), device=hull.device, dtype=torch.float32)
    start = torch.as_tensor(origin, device=hull.device, dtype=torch.float32)
    safe = torch.where(ray.abs() > 1e-9, ray, torch.full_like(ray, 1e-9))
    t0, t1 = (hull.bounds_low-start)/safe, (hull.bounds_high-start)/safe
    near = torch.maximum(torch.minimum(t0, t1).amax(dim=1), torch.zeros(len(ray), device=hull.device))
    far = torch.maximum(t0, t1).amin(dim=1)
    active = torch.nonzero(far > near, as_tuple=False).flatten()
    points = torch.zeros_like(ray)
    mask = torch.zeros(len(ray), dtype=torch.bool, device=hull.device)
    for ids in active.split(2048):
        if not len(ids):
            continue
        step = hull.spacing*.5/torch.linalg.vector_norm(ray[ids], dim=1)
        count = int(torch.ceil(((far[ids]-near[ids])/step).max()).item())+1
        for offset in range(0, count, 64):
            if not len(ids):
                break
            # Repeat the preceding sample so a boundary hit has its interpolation pair.
            begin = max(0, offset-1)
            times = near[ids, None]+torch.arange(begin, min(offset+64, count), device=hull.device)[None, :]*step[:, None]
            d = ray[ids]
            samples = start+d[:, None, :]*times[:, :, None]
            grid = 2*(samples-hull.bounds_low)/(hull.bounds_high-hull.bounds_low)-1
            values = functional.grid_sample(hull.volume, grid[None, None], mode='bilinear',
                padding_mode='zeros', align_corners=True)[0, 0, 0]
            inside = (values >= .5) & (times <= far[ids, None])
            hit = inside.any(dim=1)
            first = inside.to(torch.int32).argmax(dim=1)
            previous = torch.clamp(first-1, min=0)
            batch = torch.arange(len(ids), device=hull.device)
            before, after = values[batch, previous], values[batch, first]
            fraction = torch.clamp((.5-before)/torch.clamp(after-before, min=1e-6), 0, 1)
            depth = times[batch, previous]+fraction*step
            candidate = start+d*depth[:, None]
            if offset == 0:
                points[ids] = candidate
            points[ids[hit]] = candidate[hit]
            mask[ids[hit]] = True
            keep = ~hit
            ids, step = ids[keep], step[keep]
    return points.cpu().numpy().reshape(*shape, 3), mask.cpu().numpy().reshape(shape)
