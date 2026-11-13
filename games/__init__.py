"""The competition games: one script per game, built from the shared parts (perception/, planning/,
state_machine/, hardware/) and its rules in config/games/<game>.yaml. Run one with `python run_game.py <game>`.

Robot Arm AI Challenge (docs/Robot-Championship-2026-V0.5.pdf, section 5.8):
    pick_and_place          Game 1: colour sort, shape sort, dynamic sort
    two_arm_coordination    Game 2: pouring, folding, wrap and pack     (not implemented yet)
"""

GAMES = {
    "pick_and_place": "games.pick_and_place",
    "two_arm_coordination": "games.two_arm_coordination",
}
