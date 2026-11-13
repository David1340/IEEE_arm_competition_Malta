"""Planning a pick and place for one arm: every waypoint's pose (M-FRPSO IK) and the collision-free path to it
(RRT-Connect), in the planning world of planning/sim.py (the table, the bins, the other objects and the other arm as
obstacles).

    plans, grasp_z = plan_task(task, start_q, obstacles, args)          # [(label, joint path)], grasp height
    plans = plan_sequence(world, start_q, [("above", xyz), ("grasp", xyz)], seed)

`args` holds the tuning values (the games' command line options: --approach, --lift, --clearance, ...).
PyBullet is used by one thread at a time: hold PLAN_LOCK around plan_task when arms plan in parallel.
"""
import math
import threading

import numpy as np

from core import arms as arm_config
from core.arms import arm_from_table
from hardware.gripper import GRIPPER_OPEN
from planning.sim import JOINTS, START_Q, URDF_FILE, World
from state_machine.coordinator import TO_HANDOFF, drop_point

HOME_Q = list(START_Q)                    # the home pose, radians (config/arms.yaml home)
SHAPE_RADIUS = 0.03      # other marked shapes are obstacles: a column of this radius up to their top [m]
PLAN_LOCK = threading.Lock()   # PyBullet is used by one thread at a time

PRECISE = 0.012        # m: how close the grasp must be at most
GRASP_TIGHT = 0.005    # m: the grasp is first tried this close (one start), PRECISE only when that fails: a top-down
                       # pose far out often settles short along the reach, which a 12 mm allowance hid
ROUGH = 0.03           # m: the in-between points (above, lift, over the box) only need to be about right
SEED_TRIES = 3         # a failed point is tried again from other random starts


def plan_sequence(world, start_q, waypoints, seed, postures=None, seed_tries=None):
    """Plan start -> each waypoint in turn. waypoints: [(label, xyz in the arm frame)] or
    [(label, xyz, tolerance m)]; without a tolerance the grasp is PRECISE and every other point ROUGH.
    postures: {label: [gripper postures to try, in order]} ("horizontal", "down", "free"; planning.sim.World.find_pose),
    seed_tries: random starts per point (default SEED_TRIES); a failed start costs ~1.5 s, a pose that exists
    is found in ~10 ms.
    default horizontal. Returns [(label, path)] or raises RuntimeError naming the step that failed."""
    q = np.array(start_q, float)
    q[4] = world.start_q[4]          # wrist roll is fixed at its value when the world was built
    plans = []
    for waypoint in waypoints:
        label, xyz = waypoint[:2]
        if len(waypoint) > 2:
            tolerances = [waypoint[2]]
        else:   # aim precisely; only the in-between points fall back to ROUGH when that fails
            tolerances = [GRASP_TIGHT, PRECISE] if label == "grasp" else [PRECISE, ROUGH]
        world.start_q = q.tolist()
        result = None
        print(f"    planning '{label}' ...", flush=True)
        for posture in (postures or {}).get(label, ["horizontal"]):
            for tolerance in tolerances:
                tries = SEED_TRIES if seed_tries is None else seed_tries
                if label == "grasp" and tolerance == GRASP_TIGHT and len(tolerances) > 1:
                    tries = 1                   # one quick try at the tight one; a failed start costs ~1.5 s
                for attempt in range(tries):
                    result = world.find_pose(xyz, seed + attempt, tolerance=tolerance, posture=posture)
                    if result is not None and result.success:
                        break
                if result is not None and result.success:
                    if tolerance != tolerances[0]:
                        print(f"  '{label}': {1000 * result.position_error_m:.0f} mm off (allowed for this point)")
                    break
            if result is not None and result.success:
                if posture != "horizontal":
                    print(f"    '{label}': gripper {posture}")
                break
        if result is None or not result.success:
            err = "" if result is None else f" (best {1000 * result.position_error_m:.0f} mm off)"
            raise RuntimeError(f"no pose for '{label}' at {np.round(xyz, 3)}{err}")
        if label == "grasp":                    # where the planned grasp really is, from the model
            world.show(result.joint_positions)
            off = world.tcp() - np.asarray(xyz, float)
            out = np.asarray(xyz[:2], float) / max(np.linalg.norm(xyz[:2]), 1e-9)   # the arm frame: base at 0
            along = 1000 * float(off[:2] @ out)
            across = 1000 * float(off[:2] @ np.array([-out[1], out[0]]))
            print(f"    planned grasp: {1000 * np.linalg.norm(off):.0f} mm from the object's centre "
                  f"({abs(along):.0f} mm {'beyond' if along > 0 else 'short'} along the reach, {abs(across):.0f} mm "
                  f"{'left' if across > 0 else 'right'}, {1000 * off[2]:+.0f} mm in height)")
        route = world.find_path(result.joint_positions, seed)
        if route is None:
            raise RuntimeError(f"no collision-free path to '{label}'")
        plans.append((label, route.path))
        q = np.asarray(route.path[-1], float)
    return plans



# ---------------------------------------------------------------------------------------------- the planning world
def add_obstacle(world, x, y, half_x, half_y, height):
    """A box from the table up to `height` at (x, y) in the arm frame, as an obstacle for the planners."""
    import pybullet as p

    c = world.client
    half = [half_x, half_y, height / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.6, 0.4, 0.2, 0.5], physicsClientId=c)
    body = p.createMultiBody(0, col, vis, basePosition=[x, y, height / 2], physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    getattr(world, "extra_bodies", []).append(body)        # removed by the world's next reset()
    return body


def add_column(world, x, y, radius, height):
    """A cylinder from the table up to `height` at (x, y) in the arm frame (an object next to the one being
    picked: its size, any orientation), as an obstacle for the planners."""
    import pybullet as p

    c = world.client
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=c)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.6, 0.4, 0.2, 0.5],
                              physicsClientId=c)
    body = p.createMultiBody(0, col, vis, basePosition=[x, y, height / 2], physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    getattr(world, "extra_bodies", []).append(body)
    return body


def object_stl(mark):
    """The STL of a mark's object at its detected size (the large one when the size is unknown), or None."""
    from perception.detection import OBJECT_SIZES, OBJECT_STL_FILES

    name = mark.get("name")
    return OBJECT_SIZES.get(name, {}).get(mark.get("size", "")) or OBJECT_STL_FILES.get(name)


def neighbour_radius(mark, args):
    """How far around another object's centre the gripper must keep out [m]. --neighbour-radius < 0 (default): the
    object's own half width at its widest (any orientation: the camera does not give it) + --neighbour-margin;
    0: other objects are not obstacles; > 0: that radius for every object."""
    radius = getattr(args, "neighbour_radius", -1.0)
    if radius >= 0:
        return radius
    from hardware.gripper import object_width

    stl = object_stl(mark)
    half = object_width(stl) / 2 if stl is not None else 0.03
    return half + getattr(args, "neighbour_margin", 0.005)


# ---------------------------------------------------------------------------------------------- wrist roll (top-down)
# For a top-down grasp the wrist roll turns the jaws about the vertical: at roll r (deg) they close along a line
# about r deg from the reach (measured in the planning world: -90 = across the reach, the home pose's roll; 0 =
# along it, the fixed finger on the base's side). The planner tries the rolls whose fingers come down in free space.
DOWN_ROLLS_DEG = (0, -90, 90, -45, 45, -135, 135)   # preference when nothing is near: 0 first (an arm stopping short
                                                    # of the object still closes on it), then the old -90
FIXED_FACE = 0.030        # the fixed finger's inner face from the grasp point, along the jaws' closing line [m]
FINGER_THICK = 0.008      # finger plate thickness [m]
FINGER_HALF_WIDTH = 0.012  # half the finger plates' width, across the closing line [m]


def finger_points(xy, roll_deg, opening_cmd):
    """Points on the two fingers (arm frame xy) for a top-down grasp at xy with that wrist roll and jaw opening."""
    from hardware.gripper import opening_width

    reach = math.atan2(xy[1], xy[0])
    axis = np.array([math.cos(reach + math.radians(roll_deg)), math.sin(reach + math.radians(roll_deg))])
    side = np.array([-axis[1], axis[0]])
    fixed = np.asarray(xy) - axis * (FIXED_FACE + FINGER_THICK / 2)
    moving = np.asarray(xy) + axis * (opening_width(opening_cmd) / 2 + FINGER_THICK / 2)
    return [c + side * k * FINGER_HALF_WIDTH for c in (fixed, moving) for k in (-1.0, 0.0, 1.0)]


def finger_clearance(points, neighbours, boxes, box_half):
    """The least room [m] between the finger points and the other objects (circles: xy, radius) and the bin walls
    (squares: xy, half side); negative: a finger would land on something."""
    best = math.inf
    for point in points:
        for centre, radius in neighbours:
            best = min(best, float(np.linalg.norm(point - centre)) - radius - FINGER_THICK / 2)
        for centre in boxes:
            d = np.abs(point - centre)
            inside = d.max() < box_half
            to_wall = box_half - d.max() if inside else float(np.linalg.norm(np.maximum(d - box_half, 0.0)))
            best = min(best, to_wall - FINGER_THICK / 2)
    return best


def down_rolls(world, start_q, xy, opening_cmd, neighbours, boxes, args):
    """The wrist rolls [rad] to try for a top-down grasp at xy, best first: those whose fingers have
    --finger-clearance of room in --down-roll's preference order, then the rest by room. --down-roll across: the
    start pose's roll only (the old behaviour)."""
    if getattr(args, "down_roll", "auto") == "across":
        return [float(start_q[4])]
    low, high = world.roll_limits()
    need = getattr(args, "finger_clearance", 0.006)
    scored = []
    for rank, roll in enumerate(DOWN_ROLLS_DEG):
        if not low <= math.radians(roll) <= high:
            continue
        room = finger_clearance(finger_points(xy, roll, opening_cmd), neighbours, boxes, args.box_size / 2)
        scored.append((room < need, rank if room >= need else -room, roll, room))
    scored.sort()
    return [math.radians(roll) for _, _, roll, _ in scored[:max(1, getattr(args, "down_rolls", 3))]]


def add_other_arm(world, to_arm, other, q):
    """The other arm's robot model at its base and joint angles q, as an obstacle (it is not planned). A reused
    world keeps the model between plans (world.other_arms) and only moves it."""
    import pybullet as p

    c = world.client
    b = arm_config.base(other)
    xy = (to_arm @ np.array([b["x"], b["y"], 0.0, 1.0]))[:2]
    yaw = math.atan2(to_arm[1, 0], to_arm[0, 0]) + math.radians(b["yaw_deg"])
    kept = getattr(world, "other_arms", None)
    if kept is not None and other in kept:
        body = kept[other]
        p.resetBasePositionAndOrientation(body, [*xy, 0.0], p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=c)
    else:
        body = p.loadURDF(str(URDF_FILE), basePosition=[*xy, 0.0],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]), useFixedBase=True, physicsClientId=c)
        if kept is not None:
            kept[other] = body
    index = {p.getJointInfo(body, i, physicsClientId=c)[1].decode(): i
             for i in range(p.getNumJoints(body, physicsClientId=c))}
    for name, value in zip(JOINTS, q):
        p.resetJointState(body, index[name], float(value), physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    return body


def handoff_checker(handoff_points, args):
    """can_grasp(arm, point index, top z): can that arm grasp a shape of that height on that handoff point (an IK
    check in an empty world, from home)? Cached, so each combination is solved once. Returns (can_grasp, close)."""
    cache, worlds = {}, {}

    def can_grasp(arm, i, top):
        # keyed by the point's position, not its index: a caller may replace the list's points between rounds
        key = (arm, round(float(handoff_points[i][0]), 3), round(float(handoff_points[i][1]), 3), round(float(top), 2))
        if key not in cache:
            with PLAN_LOCK:
                if arm not in worlds:
                    worlds[arm] = World([], list(HOME_Q), gui=False, clearance=args.clearance, arm=arm,
                                        calibration_path=arm_config.follower_calibration(arm))
                    worlds[arm].level_tolerance_deg = getattr(args, "level_tolerance", 30.0)
                    worlds[arm].stall_iterations = getattr(args, "ik_stall", 30) or None
                world = worlds[arm]
                h = (arm_from_table(arm) @ np.array([*handoff_points[i][:2], 0.0, 1.0]))[:3]
                grasp_z = grasp_height(top, args)
                ok = False
                for posture in postures_for(h, args):
                    for z in grasp_heights(top, posture, args):
                        world.start_q = list(HOME_Q)
                        result = world.find_pose([h[0], h[1], z], args.seed, posture=posture)
                        if result is not None and result.success:
                            ok = True
                            break
                    if ok:
                        break
            cache[key] = ok
            if not ok:
                print(f"  handoff point H{i} {handoff_points[i]}: {arm} cannot grasp a {100 * top:.1f} cm shape there")
        return cache[key]

    def close():
        for world in worlds.values():
            world.close()
    return can_grasp, close


def grasp_height(top, args):
    """Height of the point between the jaws at the grasp: half the object's height (--grasp-at middle, the default)
    or --grasp-depth below its top (--grasp-at top); never below --min-grasp-z (the lowest the jaws reach). If
    there is no pose there, the planner tries 5 mm higher each time."""
    z = top / 2 if getattr(args, "grasp_at", "middle") == "middle" else top - args.grasp_depth
    return max(z, args.min_grasp_z)


# How far the fingers reach below the grasp point when the gripper is level (PyBullet, models/robot/so101: plates from
# 3.3 cm below to 2.9 cm above it). A level grasp point cannot go lower than about 3.5 cm (table clearance).
LEVEL_JAW_BELOW = 0.033


def grasp_heights(top, posture, args):
    """The heights to try for the grasp point, lowest first (5 mm steps); the planner's table clearance check
    decides which are possible.
      level (horizontal): the jaws are at that height, so from half the object's height (--grasp-at) up to
          --level-below-top under its top (higher, they close over it); not below --min-grasp-z. With
          --pick-posture level, up to where the fingers still cover --level-min-contact of the object.
      down: the fingertips hang tip_below_grasp() (from the URDF) below the grasp point, so it never goes lower than
          that + 1.2 cm (the tips 1.2 cm over the table), and at most so high that the tips still reach 1.2 cm
          below the object's top."""
    from hardware.gripper import tip_below_grasp

    if posture == "down":
        low = max(grasp_height(top, args), tip_below_grasp() + 0.012)
        high = top + tip_below_grasp() - 0.012
    else:
        low, high = grasp_height(top, args), top - args.level_below_top
        # Never below where the level fingers still clear the table (--clearance): a lower target is unreachable, and
        # the IK would then stop at the nearest reachable pose up to PRECISE (12 mm) away, partly sideways, so the
        # jaws ended off-centre on small objects. Starting at the real lowest height keeps the error near zero.
        low = max(low, getattr(args, "clearance", 0.01) + LEVEL_JAW_BELOW + 0.002)
        if getattr(args, "pick_posture", "auto") == "level":
            # The level fingers are tall plates reaching LEVEL_JAW_BELOW under the grasp point, so a short object is
            # still between them with the grasp point above its top: allow that while they cover
            # --level-min-contact of it (the table clearance check decides the lowest possible height).
            high = max(high, top + LEVEL_JAW_BELOW - getattr(args, "level_min_contact", 0.015))
    if high < low - 1e-6:
        return np.array([])
    return np.arange(low, max(high, low) + 1e-6, 0.005)


def postures_for(xy_arm, args):
    """The gripper postures to try for a pick at this point (arm frame).
    --pick-posture level (default): only level (fingers parallel to the table, within --level-tolerance); an object
        too close to the base for a level gripper (about 20 cm) then has no plan instead of a top-down grasp.
    --pick-posture auto: level (straight in) first, unless it is within --close-radius of the base (a level gripper
        cannot get that close): top-down first there. The other one when the first has no pose."""
    if getattr(args, "pick_posture", "auto") == "level":
        return ["horizontal"]
    close = np.hypot(xy_arm[0], xy_arm[1]) < args.close_radius
    return ["down", "horizontal"] if close else ["horizontal", "down"]


def miss_position(shape, args):
    """The gripper command (0-100) below which closing on this object without a current rise means a miss: where its
    jaws touch it (its narrow side, for its detected size) minus --miss-margin. None: close fully (--miss-margin < 0 or
    the shape is unknown)."""
    if getattr(args, "miss_margin", 10.0) < 0:
        return None
    from hardware import gripper as jaw
    from perception.detection import OBJECT_SIZES, OBJECT_STL_FILES

    name = shape.get("name")
    stl = OBJECT_SIZES.get(name, {}).get(shape.get("size", "")) or OBJECT_STL_FILES.get(name)
    if stl is None:
        return None
    contact = jaw.open_command(jaw.object_footprint(stl)[0], margin=0.0, min_open=0.0)
    return max(0.0, contact - args.miss_margin)


_WORLDS = {}       # (arm, clearance) -> the arm's planning world, reused for every plan (World.reset)


def planning_world(arm, start_q, args, gripper_open=None):
    """The arm's planning world, ready for a plan from start_q: built the first time, reset afterwards (the same
    as a new one, without the ~0.85 s it takes to build). With args.gui a new world is opened each time (shown)."""
    if args.gui:
        return World([], start_q, gui=True, clearance=args.clearance, arm=arm,
                     calibration_path=arm_config.follower_calibration(arm), gripper_open=gripper_open)
    key = (arm, round(float(args.clearance), 6))
    world = _WORLDS.get(key)
    if world is None:
        world = _WORLDS[key] = World([], start_q, gui=False, clearance=args.clearance, arm=arm,
                                     calibration_path=arm_config.follower_calibration(arm), gripper_open=gripper_open)
    else:
        world.reset(start_q, gripper_open)
    return world


def close_worlds():
    for world in _WORLDS.values():
        world.close()
    _WORLDS.clear()


def plan_task(task, start_q, obstacles, args, handoff_xyz=None):
    """Plan the whole task in the arm's frame. Returns ([(label, path)], grasp z) or raises RuntimeError."""
    arm = task.arm
    to_arm = arm_from_table(arm)
    o = (to_arm @ np.append(task.shape["xyz"], 1.0))[:3]
    base = arm_config.base(arm)
    drop = drop_point(task.box["xyz"], [base["x"], base["y"]])      # inside the box, on the side nearest the arm
    b = (to_arm @ np.array([*drop, 0.0, 1.0]))[:3]
    top = max(o[2], 0.0)
    opening = task.shape.get("open_cmd", GRIPPER_OPEN)        # the jaws open only as wide as this shape needs
    world = planning_world(arm, start_q, args, gripper_open=opening / 100)
    world.level_tolerance_deg = getattr(args, "level_tolerance", 30.0)
    world.stall_iterations = getattr(args, "ik_stall", 30) or None
    try:
        neighbours, boxes_xy = [], []
        for other in obstacles["shapes"]:
            if other is task.shape or other is task.shape.get("origin"):
                continue
            xyz = other.get("handoff_xyz") if other["state"] == "handoff" else other["xyz"]
            xy = (to_arm @ np.append(xyz, 1.0))[:2]
            radius = neighbour_radius(other, args)
            if radius <= 0:                    # 0: the other objects are not obstacles (the gripper may push them)
                continue
            if getattr(args, "neighbour_radius", -1.0) < 0:
                add_column(world, *xy, radius, max(xyz[2], 0.01))       # each object's own size
            else:
                add_obstacle(world, *xy, radius, radius, max(xyz[2], 0.01))   # the old square columns
            neighbours.append((xy, radius))
        for box in obstacles["boxes"]:
            xy = (to_arm @ np.append(box["xyz"], 1.0))[:2]
            boxes_xy.append(xy)
            add_obstacle(world, *xy, args.box_size / 2, args.box_size / 2, args.box_height + args.box_margin)
        for other, q in obstacles["arms"].items():
            if other != arm:
                add_other_arm(world, to_arm, other, q)

        if task.kind == TO_HANDOFF:
            h = (to_arm @ np.array([*handoff_xyz[:2], 0.0, 1.0]))[:3]
            rest = lambda z: [("over handoff", [h[0], h[1], top + args.lift]), ("place", [h[0], h[1], z]),
                              ("leave handoff", [h[0], h[1], top + args.approach])]
            place_postures = postures_for(h, args)
        else:
            h = None
            drop_z = max(args.box_height + args.drop_above, top + args.lift)
            rest = lambda z: [("over box", [b[0], b[1], drop_z])]
            place_postures = ["horizontal", "down"]
        error = None
        # one posture for the whole pick (above, down, lift: a straight descent); the other one if that fails.
        # A pose that exists is found in ~10 ms, but each failed search takes ~1.5 s, and a grasp height that is out
        # of reach fails for every random start. So scan every height with ONE start first and only then give the
        # same heights the other starts (before: 3 starts per height, ~20 s for an object whose lowest heights are
        # out of reach).
        # A top-down grasp is tried with each of the best wrist rolls (down_rolls); the wrist turns at the start
        # pose, before the arm moves (checked collision-free), and the whole pick and carry keep that roll.
        variants = []
        for posture in postures_for(o, args):
            if posture == "down":
                variants += [(posture, roll) for roll in down_rolls(world, start_q, o[:2], opening, neighbours,
                                                                    boxes_xy, args)]
            else:
                variants.append((posture, float(start_q[4])))
        turned = {}

        def turn(roll):
            """The start pose with the wrist at `roll`, or None when the turn is not collision-free."""
            q = list(start_q)
            q[4] = roll
            if abs(roll - float(start_q[4])) < 1e-6:
                return q
            if roll not in turned:
                turned[roll] = q if world.turn_is_free(start_q, q) else None
            return turned[roll]

        def attempt(posture, roll, z, seed, tries):
            start = turn(roll)
            if start is None:
                raise RuntimeError(f"no room to turn the wrist to {math.degrees(roll):.0f} deg at the start pose")
            world.set_start(start)
            # placing on a handoff point: the posture it was picked with first (top-down keeps the object upright)
            place = [posture] + [p for p in place_postures if p != posture] if h is not None else place_postures
            plans = plan_sequence(world, start, [
                ("above shape", [o[0], o[1], top + args.approach]),
                ("grasp", [o[0], o[1], z]),
                ("lift", [o[0], o[1], top + args.lift]),
                *rest(z),
            ], seed, postures={"above shape": [posture], "grasp": [posture], "lift": [posture],
                               "over handoff": place, "place": place,
                               "leave handoff": place, "over box": ["horizontal", "down"]},
                seed_tries=tries)
            if start is not start_q and abs(start[4] - float(start_q[4])) > 1e-6:
                label, path = plans[0]                   # turn the wrist first, where the arm is
                plans[0] = (label, [list(start_q), *path])
            return plans

        def done(posture, roll, plans, z):
            how = f"{posture} grasp" + (f", wrist roll {math.degrees(roll):+.0f} deg" if posture == "down" else "")
            print(f"  {arm}: {how} ({100 * np.hypot(*o[:2]):.0f} cm from its base)")
            if args.gui:
                for _, path in plans:
                    world.start_q = list(path[0])
                    world.play(path)
            return plans, z

        retry = (args.seed + 1, SEED_TRIES - 1)
        passes = [(args.seed, 1), retry] if SEED_TRIES > 1 else [(args.seed, 1)]
        for seed, tries in passes:
            for posture, roll in variants:
                heights = grasp_heights(top, posture, args)
                if not len(heights) and seed == args.seed:
                    print(f"  {arm}: a {100 * top:.1f} cm object is too short for a {posture} grasp")
                for i, z in enumerate(heights):
                    try:
                        plans = attempt(posture, roll, z, seed, tries)
                    except RuntimeError as exc:
                        error = exc
                        if "grasp" not in str(exc) and "place" not in str(exc):
                            break
                        continue
                    if seed == args.seed and i > 0 and SEED_TRIES > 1:
                        # Found, but not at the lowest height: a lower one may only have missed its single quick
                        # try (near the edge of what the arm reaches level). Give those the other starts first, so
                        # the grasp is as low as possible (costs ~1-3 s, only for these picks).
                        for lower in heights[:i]:
                            try:
                                return done(posture, roll, attempt(posture, roll, lower, *retry), lower)
                            except RuntimeError:
                                pass
                    return done(posture, roll, plans, z)
        raise RuntimeError(str(error))
    finally:
        if args.gui:
            world.close()
        else:
            world.reset(world.start_q)               # drop this plan's obstacles; the world is kept for the next

