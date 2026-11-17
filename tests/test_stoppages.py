"""Synthetic tests for stoppage (dead-ball) evidence. No match data is used."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_all
from synx import data, possession, stoppages
from synx.data import AWAY, HOME

FPS = 25
KICKOFF = 11185                     # first in-play frame (published), so play starts here
PITCH = (105.0, 68.0)


def _speed(ball):
    return data.ball_speed(np.asarray(ball, dtype=np.float32))


def throw_in_match():
    """Home keeps the ball, it goes out over the touchline, is held out, thrown in, and away plays it.

    Frames (relative to KICKOFF): 0-999 home dribbles up the pitch; 1000 the ball crosses y = 68;
    1000-1299 it is out (held for the throw); 1300 thrown in fast; 1300+ away has it.
    """
    n = KICKOFF + 2400
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    t = np.arange(n - KICKOFF, dtype=np.float32)
    live = np.column_stack((20 + 0.03 * t, 30 + 0.0 * t))
    ball[KICKOFF:] = live
    out_start, restart = KICKOFF + 1000, KICKOFF + 1300
    ball[out_start:restart] = (50.0, 69.5)                       # held beyond the touchline
    throw = np.arange(restart, n)
    ball[throw, 0] = 50.0
    ball[throw, 1] = np.maximum(67.0 - 0.3 * (throw - restart), 20.0)  # 7.5 m/s back into play
    ball[throw[60:], 1] = 20.0 + 0.04 * np.arange(len(throw) - 60)       # then a slow dribble
    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, 0] = ball + (0.5, 0.0)                              # home player
    players[:, 1] = ball + (8.0, 0.0)                              # away player far
    players[restart:, 0] = ball[restart:] + (8.0, 0.0)            # away takes the throw
    players[restart:, 1] = ball[restart:] + (0.5, 0.0)
    track = {"ball": ball, "players": players, "player_speed": np.ones((n, 2), np.float32),
             "team": np.array([HOME, AWAY]), "ball_speed": _speed(ball), "pitch": PITCH}
    events = [{"code": "SET PIECES", "start_frame": out_start - 25, "end_frame": out_start + 850,
               "tags": {"Type": "Throw-in", "Team": "Away"}}]
    return track, events, out_start, restart


def test_basic_mode_is_unchanged_default_mask():
    track, _, _, _ = throw_in_match()
    dead, report = stoppages.dead_mask(track, "basic")
    assert np.array_equal(dead, possession.default_dead_mask(track))
    assert report["mode"] == "basic"


def test_still_ball_needs_two_seconds():
    at_rest = np.zeros(200, dtype=bool)
    at_rest[10:10 + 49] = True          # 1.96 s: open-play pause, not a stoppage
    at_rest[100:100 + 50] = True        # 2 s: placed ball
    still = stoppages.still_ball(at_rest)
    assert not still[10:59].any()
    assert still[100:150].all() and still.sum() == 50


def test_out_bridge_only_follows_a_seen_out_ball_and_is_capped():
    n = 2000
    out = np.zeros(n, dtype=bool)
    seen = np.ones(n, dtype=bool)
    out[100:110] = True
    seen[110:140] = False               # unseen right after the ball went out: bridged
    seen[500:540] = False               # unseen during play: never dead
    out[900] = True
    seen[901:1901] = False              # longer than the 30 s cap
    bridge = stoppages.out_bridge(out, seen)
    assert bridge[110:140].all()
    assert not bridge[500:540].any()
    assert bridge[901:901 + stoppages.MAX_OUT_BRIDGE_FRAMES].all()
    assert not bridge[901 + stoppages.MAX_OUT_BRIDGE_FRAMES:].any()


def test_missing_ball_alone_never_becomes_dead():
    track, events, _, _ = throw_in_match()
    gap = slice(KICKOFF + 400, KICKOFF + 500)
    track["ball"][gap] = np.nan
    track["ball_speed"] = _speed(track["ball"])
    for mode in ("tracking", "tagged"):
        dead, _ = stoppages.dead_mask(track, mode, events)
        assert not dead[gap].any(), mode


def test_tagged_throw_in_dead_from_ball_out_to_restart():
    track, events, out_start, restart = throw_in_match()
    dead, report = stoppages.dead_mask(track, "tagged", events)
    assert report["used"] == 1 and report["restart_found"] == 1
    assert dead[out_start:restart].all()
    assert not dead[restart:restart + 400].any()        # live again once the ball is thrown in
    assert not dead[KICKOFF:out_start - 1].any()        # play before the stoppage stays live


def test_tag_window_is_never_copied_wholesale():
    track, events, out_start, restart = throw_in_match()
    dead, _ = stoppages.dead_mask(track, "tagged", events)
    window = slice(events[0]["start_frame"], events[0]["end_frame"])
    assert 0 < dead[window].sum() < dead[window].size


def test_tag_without_stoppage_evidence_is_skipped():
    track, _, _, _ = throw_in_match()
    events = [{"code": "SET PIECES", "start_frame": KICKOFF + 100, "end_frame": KICKOFF + 975,
               "tags": {"Type": "Free Kick"}}]
    tracking, _ = stoppages.dead_mask(track, "tracking")
    tagged, report = stoppages.dead_mask(track, "tagged", events)
    assert report["skipped"] == 1 and report["used"] == 0
    assert np.array_equal(tagged, tracking)


def test_kickoff_after_goal_dead_from_goal_to_kickoff():
    n = KICKOFF + 3000
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[KICKOFF:] = (60.0, 34.0)
    goal, kickoff = KICKOFF + 500, KICKOFF + 2000
    ball[KICKOFF:goal, 0] = np.linspace(60, 105, goal - KICKOFF)
    ball[goal:goal + 25] = (106.0, 34.0)      # in the net, beyond the goal line
    ball[goal + 25:kickoff] = np.nan           # celebrations off camera
    ball[kickoff - 100:kickoff] = (52.5, 34.0)  # back on the centre spot
    ball[kickoff:kickoff + 40, 0] = 52.5 - 0.25 * np.arange(40)  # kick-off pass, 6.25 m/s
    ball[kickoff + 40:, 0] = 42.5
    track = {"ball": ball, "ball_speed": _speed(ball), "pitch": PITCH}
    events = [{"code": "SET PIECES", "start_frame": kickoff - 60, "end_frame": kickoff + 65,
               "tags": {"Type": "Kick Off (After Goal)"}},
              {"code": "SET PIECES", "start_frame": KICKOFF - 60, "end_frame": KICKOFF + 65,
               "tags": {"Type": "Kick Off (Start)"}}]
    dead, report = stoppages.dead_mask(track, "tagged", events)
    assert report["after_goal_used"] == 1 and report["set_pieces"] == 1
    assert dead[goal:kickoff].all()
    # The ball first moves on the frame after the kick, and speed needs two positions.
    assert not dead[kickoff + 3:kickoff + 30].any()


def test_new_spell_starts_after_restart_and_chunks_agree():
    track, events, out_start, restart = throw_in_match()
    dead, _ = stoppages.dead_mask(track, "tagged", events)
    labels = run_all.predict(track, chunk_size=len(dead), dead=dead)
    assert (labels[KICKOFF + 100:out_start - 1] == "H").all()
    assert (labels[out_start:restart] == "D").all()
    after = labels[restart:]
    assert "H" not in after and (after[200:] == "A").all()
    for size in (1, 25, 512, 4096):
        assert np.array_equal(run_all.predict(track, chunk_size=size, dead=dead), labels), size


def test_restart_needs_sustained_fast_ball_inside_pitch():
    fast = np.zeros(100, dtype=bool)
    fast[20:29] = True                  # 9 frames: a deflection or noise, not a restart
    assert stoppages.restart_frame(fast, 0, 100) is None
    fast[50:60] = True                  # 10 frames (0.4 s): the ball is struck back into play
    assert stoppages.restart_frame(fast, 0, 100) == 50
    assert stoppages.restart_frame(fast, 55, 100) is None  # run cut short by the search start


def test_no_restart_keeps_only_evidence_frames():
    n = 2000
    out = np.zeros(n, dtype=bool)
    out[300:340] = True
    events = [{"code": "SET PIECES", "start_frame": 290, "end_frame": 1165,
               "tags": {"Type": "Corner Kick"}}]
    mask, counts = stoppages.tagged_stoppages(events, out, np.zeros(n, bool), np.zeros(n, bool), n)
    assert counts["used"] == 1 and counts["restart_found"] == 0
    assert np.array_equal(mask, out)


def test_malformed_and_irrelevant_events_do_not_crash_or_add_dead():
    n = 2000
    out = np.zeros(n, dtype=bool)
    out[300:340] = True
    events = [
        {"code": "BUILD UP", "start_frame": 290, "end_frame": 900, "tags": {"Team": "Home"}},
        {"code": "SET PIECES", "start_frame": "x", "end_frame": 900, "tags": {"Type": "Throw-in"}},
        {"code": "SET PIECES", "end_frame": 900, "tags": {"Type": "Throw-in"}},
        {"code": None, "start_frame": 290, "end_frame": 900},
        {"code": "SET PIECES", "start_frame": 900, "end_frame": 800, "tags": None},
        {"code": "SET PIECES", "start_frame": n + 10, "end_frame": n + 900, "tags": {}},
    ]
    mask, counts = stoppages.tagged_stoppages(events, out, np.zeros(n, bool), np.zeros(n, bool), n)
    assert not mask.any()
    assert counts["used"] == 0 and counts["skipped"] == 4


def test_basic_mode_matches_independent_rule_and_run_default():
    track, _, _, _ = throw_in_match()
    from synx import pitch
    n = len(track["ball"])
    expected = pitch.ball_out_of_pitch(track["ball"], PITCH) | ~pitch.in_play_window(n)
    dead, _ = stoppages.dead_mask(track, "basic")
    assert np.array_equal(dead, expected)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        implicit = run_all.run(track, os.path.join(tmp, "a.csv"))
        explicit = run_all.run(track, os.path.join(tmp, "b.csv"), dead=dead)
    assert np.array_equal(implicit, explicit)


def test_tracking_mode_adds_only_still_ball_and_out_bridge():
    track, _, out_start, restart = throw_in_match()
    still_at = slice(KICKOFF + 200, KICKOFF + 260)          # 2.4 s at rest in the middle of play
    track["ball"][still_at] = track["ball"][still_at.start]
    gap = slice(restart - 50, restart)                        # ball out, then lost off camera
    track["ball"][gap] = np.nan
    track["ball_speed"] = _speed(track["ball"])
    basic, _ = stoppages.dead_mask(track, "basic")
    tracking, report = stoppages.dead_mask(track, "tracking")
    added = tracking & ~basic
    assert added[still_at.start + 5:still_at.stop].all()     # speed needs a frame to settle
    assert added[gap].all()
    assert report["tracking_added"] == int(added.sum())
    assert added.sum() <= (still_at.stop - still_at.start) + (gap.stop - gap.start)


def test_tag_evidence_search_window_edges():
    n = 3000
    start = 1000
    events = [{"code": "SET PIECES", "start_frame": start, "end_frame": start + 875,
               "tags": {"Type": "Free Kick"}}]
    lookback, ahead = stoppages.TAG_LOOKBACK_FRAMES, stoppages.TAG_EVIDENCE_FRAMES
    for frame, used in ((start - lookback, 1), (start - lookback - 1, 0),
                        (start + ahead - 1, 1), (start + ahead, 0)):
        out = np.zeros(n, dtype=bool)
        out[frame] = True
        _, counts = stoppages.tagged_stoppages(events, out, np.zeros(n, bool),
                                               np.zeros(n, bool), n)
        assert counts["used"] == used, frame


def test_kickoff_after_goal_without_goal_or_restart():
    n = 6000
    out = np.zeros(n, dtype=bool)
    events = [{"code": "SET PIECES", "start_frame": 5000, "end_frame": 5125,
               "tags": {"Type": "Kick Off (After Goal)"}}]
    mask, counts = stoppages.tagged_stoppages(events, out, np.zeros(n, bool), np.zeros(n, bool), n)
    assert counts["skipped"] == 1 and not mask.any()          # no goal seen: nothing guessed
    out[4000] = True
    mask, counts = stoppages.tagged_stoppages(events, out, np.zeros(n, bool), np.zeros(n, bool), n)
    assert counts["after_goal_used"] == 1 and counts["restart_found"] == 0
    assert mask[4000:5000].all() and not mask[5000:].any()   # dead up to the tagged window


def test_overlapping_tags_give_one_union():
    track, events, out_start, restart = throw_in_match()
    twice = events + [dict(events[0], start_frame=events[0]["start_frame"] + 100)]
    once, _ = stoppages.dead_mask(track, "tagged", events)
    both, report = stoppages.dead_mask(track, "tagged", twice)
    assert report["used"] == 2
    assert np.array_equal(once, both)


def test_non_mapping_tags_are_skipped():
    n = 500
    events = [{"code": "SET PIECES", "start_frame": 10, "end_frame": 400, "tags": "Throw-in"},
              {"code": "SET PIECES", "start_frame": float("inf"), "end_frame": 400, "tags": {}},
              "not an event"]
    out = np.ones(n, dtype=bool)
    mask, counts = stoppages.tagged_stoppages(events, out, np.zeros(n, bool), np.zeros(n, bool), n)
    assert not mask.any() and counts["skipped"] == 3


def test_main_reads_smart_tagging_only_when_needed():
    track, events, _, _ = throw_in_match()
    calls = []
    original_load, original_events = run_all.data.load, run_all.events.load_smart_tagging_events
    run_all.data.load = lambda *a, **k: track
    run_all.events.load_smart_tagging_events = lambda *a, **k: calls.append(1) or events
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_all.main(["--dead-ball", "basic", "--output", os.path.join(tmp, "basic.csv")])
            run_all.main(["--dead-ball", "tracking", "--output", os.path.join(tmp, "tracking.csv")])
            assert calls == []
            run_all.main(["--output", os.path.join(tmp, "default.csv")])  # default is tagged
            assert calls == [1]
    finally:
        run_all.data.load, run_all.events.load_smart_tagging_events = original_load, original_events


def test_modes_validate_inputs():
    track, _, _, _ = throw_in_match()
    for bad in (lambda: stoppages.dead_mask(track, "tagged"),
                lambda: stoppages.dead_mask(track, "smart")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_runner_accepts_dead_ball_modes():
    parser = run_all.build_parser()
    assert parser.parse_args([]).dead_ball == "tagged"
    assert parser.parse_args(["--dead-ball", "basic"]).dead_ball == "basic"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
