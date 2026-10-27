"""The E4 prompt: ask a video model for the same numbers the pipeline measures.

Definitions match kickcoach/features/geometry.py (DEC-006), so answers are comparable with the
pipeline and the labels. "Cannot determine" is always allowed (eval plan, E4 protocol).
The kicking leg is given when it is labelled; before labelling (the Phase 1 gut-check) the
model is asked to name it.
"""

from __future__ import annotations

PROMPT_VERSION = "e4-v1"

_DEFINITIONS = """\
Definitions (2D, measured in the image as filmed):
- Ball contact: the first video frame in which the kicking foot touches the ball.
- Knee angle at contact: the interior angle at the kicking leg's knee between the thigh
  (knee to hip joint centre) and the shank (knee to ankle joint centre). 180 degrees is a
  fully straight leg; smaller means more bent.
- Trunk lean at contact: the angle between the line from the mid-point of the hips to the
  mid-point of the shoulders and the image vertical. Positive means the shoulders are ahead
  of the hips in the direction the ball is kicked (leaning forward); negative means leaning
  back.
"""


def build_prompt(duration_s: float, kicking_leg: str | None) -> str:
    leg = (
        f"The player kicks with the {kicking_leg} leg."
        if kicking_leg
        else "Identify which leg kicks the ball (left or right, from the player's perspective)."
    )
    return f"""\
This video ({duration_s:.2f} s long) shows one football (soccer) player taking an instep
kick, filmed roughly side-on. {leg}

{_DEFINITIONS}
Report:
1. contact_time_s: seconds from the start of this video to ball contact.
2. knee_angle_contact_deg: the kicking leg's knee angle at contact, in degrees.
3. trunk_lean_contact_deg: the trunk lean at contact, in degrees.

Give numbers only if you can determine them from the video. If you cannot, set the value to
null and say why in cannot_determine_reason. Do not guess.
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "kicking_leg": {"type": "string", "enum": ["left", "right", "cannot_determine"]},
        "contact_time_s": {"type": ["number", "null"]},
        "knee_angle_contact_deg": {"type": ["number", "null"]},
        "trunk_lean_contact_deg": {"type": ["number", "null"]},
        "cannot_determine_reason": {"type": ["string", "null"]},
        "notes": {"type": "string"},
    },
    "required": [
        "kicking_leg",
        "contact_time_s",
        "knee_angle_contact_deg",
        "trunk_lean_contact_deg",
        "cannot_determine_reason",
        "notes",
    ],
}
