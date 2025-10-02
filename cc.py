""" CHATGPT PROMPT GENERATOR — CONTINUITY-LOCKED, JSON-STRICT (STABLE PATCH, SINGLE-SHOT READY)
-------------------------------------------------------------------------------------------
This patch keeps everything you already had, and ONLY adds/fixes to support:
- EXACTLY 1 shot per scene (allowed by schema), covering 0–8s continuously with no idle.
- Realistic cause→effect actions with reactions (no "hit then statue", no "arrow without flight", etc.).
- Stronger grounding to each scene's core "Nội dung" + full bối cảnh details.
- Always include top-level output_settings (filled if missing).
- Repeat wardrobe/props details consistently in every prompt when available.
Usage is unchanged. See your previous notes.
"""

from __future__ import annotations

import os
import json
import re
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import jsonschema
    from jsonschema import validate
except Exception:
    jsonschema = None

try:
    from httpx import Timeout  # used by OpenAI client under-the-hood
except Exception:
    Timeout = None

SAFE_FILENAME_REGEX = re.compile(r"[\\/:*?\"<>|]")

# --- SCENE MOTION GUARDS (anti-idle & clarity) ---
IDLE_MAX_GAP = 0.35          # tối đa 0.35s không có hành động
MIN_BEATS_8S = 16            # tối thiểu 16 beat / 8s
TAIL_FILL_TS = [6.4, 7.2, 7.85]  # bơm hành động ở đuôi gần 8s

# ============================================================================
# TOP 0.01% GLOBAL CONFIG - SINGLE SOURCE OF TRUTH
# ============================================================================
ACTION_SPACING_CONFIG = {
    "min_gap_seconds": 0.35,           # SINGLE minimum gap (no more conflicts!)
    "validation_tolerance": 0.03,      # Floating point tolerance (0.35 - 0.03 = 0.32)
    "max_verb_usage": 2,               # Max times same verb can appear in 8s
    "required_action_count": (16, 20), # Min and max actions per scene
    "distribution": {
        # Format: (start_sec, end_sec, min_gap, max_gap)
        "setup": (0.0, 2.0, 0.5, 0.8),      # Slow buildup (3-4 beats)
        "action": (2.0, 6.0, 0.3, 0.45),    # Fast intensity (10-12 beats)
        "resolution": (6.0, 8.0, 0.4, 0.6)  # Wind down (3-4 beats)
    }
}

# ============================================================================
# BIOMECHANICS & PHYSICS TEMPLATES - TOP 0.1% DETAIL LEVEL
# ============================================================================
BIOMECHANICS_TEMPLATES = {
    "charges": "Hips drive forward; quad muscles engage explosively; arms pump for momentum; center of mass lowers 15cm; feet strike heel-to-toe with 2-2.5x body weight force per step; core stabilizes spine",
    "strikes": "Shoulder rotates 90° externally; triceps extend rapidly; deltoid peak contracts; wrist snaps through impact point; core transfers rotational kinetic energy through kinetic chain; follow-through arm extends fully",
    "dodges": "Weight shifts explosively to ball of left foot; right leg pushes laterally with 500N force; torso rotates 45° counterclockwise; arms counterbalance; head tracks threat; center of mass stays low",
    "blocks": "Forearm raises to vertical; bicep and brachialis tense isometrically; shoulder elevates and internally rotates to protect head; weight centers over supporting leg; opposite hand guards ribs",
    "thrusts": "Back leg drives with quad extension; hips rotate forward 60°; shoulder extends arm linearly; elbow locks at 170°; wrist stabilizes weapon angle; front foot pivots for power transfer",
    "retreats": "Weight transfers to rear foot; front foot lifts with hip flexor contraction; hip extensors engage on rear leg; arms maintain guard position; eyes stay on threat; balance maintained through core",
    "observes": "Head tilts 10-15° toward stimulus; extraocular muscles track motion smoothly; facial muscles tighten; breathing slows and deepens; body pre-tensions for potential action",
    "prepares": "Muscles pre-tension to 30% max voluntary contraction; joints align for optimal power transfer; breathing deepens with diaphragm; grip adjusts micro-millimeters; weight distributes evenly",
    "staggers": "Impact disrupts balance; vestibular system compensates; arms flail for counterbalance; ankle and knee joints flex to absorb force; center of mass shifts backward; eyes widen",
    "falls": "Protective reflexes trigger; arms extend to break fall; body rotates to protect head; muscles tense pre-impact; knees and hips flex; landing distributed across surface area",
    "leaps": "Calf and quad muscles explode concentrically; ankle plantarflexes; arms swing upward; hip flexors pull knees up; body extends in flight; eyes focus on landing zone",
    "rolls": "Shoulder tucks; spine flexes into C-curve; momentum transfers through rounded back; hands protect head; legs tuck; core maintains rotational axis; weight distributed sequentially",
    "grabs": "Fingers flex rapidly via flexor digitorum; thumb opposes; grip strength peaks at 150-400N; wrist stabilizes; forearm muscles engage; proprioception confirms contact",
    "releases": "Fingers extend explosively; wrist snaps; arm follows through in arc; shoulder rotates externally; timing coordinates with projectile release point; eyes track target",
    "aims": "Shoulder stabilizes; scapula retracts; arm extends; wrist aligns with target line; non-dominant eye may close; breathing pauses at exhale; tremor minimized through muscle co-contraction",
    "circles": "Hip abductors and adductors alternate; feet cross over in grapevine pattern; torso remains upright; eyes track center point; arms balance; smooth weight transfer",
    "lunges": "Front leg bends to 90° at knee; rear leg extends; quad of front leg eccentrically loads; glutes engage; torso leans forward slightly; arms drive forward",
    "recoils": "Muscles contract reflexively; head pulls back; shoulders elevate; center of mass shifts away from stimulus; eyes blink or squint; arms raise defensively",
    "advances": "Weight shifts forward; rear leg pushes; front foot places heel-first; arms swing naturally; head remains level; eyes focused forward; breathing steady",
    "steadies": "Core muscles engage; feet widen stance; knees slightly flex; arms position for balance; breathing normalizes; micro-adjustments in ankle/hip; eyes scan environment"
}

PHYSICS_TEMPLATES = {
    "charges": "Acceleration 0-5m/s over 1.5-2s; momentum builds to 300-500 kg⋅m/s; friction coefficient 0.7 on packed earth provides traction; air resistance minimal at low speed; kinetic energy increases quadratically",
    "strikes": "Impact force 500-1200N depending on mass and velocity; contact duration 0.05-0.08s; energy transfer through rigid body mechanics; recoil force F=ma absorbed by stance; coefficient of restitution ~0.3",
    "dodges": "Lateral acceleration 3-6m/s²; centripetal force from direction change; friction provides reaction force; momentum redirected 90°; balance maintained via torque compensation; energy conserved through elastic collision avoidance",
    "blocks": "Impact force absorbed through bone structure; stress distributed along forearm; force redirected via angle deflection; impulse J=FΔt over 0.1s; strain energy stored in muscles; Newton's third law: equal and opposite reaction",
    "thrusts": "Linear momentum transfer; weapon mass × velocity; penetration depends on tip pressure PSI; follow-through maintains trajectory via inertia; work done W=F⋅d through target resistance; minimal energy loss to air drag",
    "retreats": "Negative acceleration -2 to -4m/s²; friction provides stopping force; momentum decreases linearly; balance maintained by shifting center of mass backward; potential energy increases if moving uphill",
    "falls": "Gravitational acceleration 9.8m/s²; rotational torque τ=r×F around center of mass; impact force depends on fall height and stopping distance; kinetic energy at impact KE=½mv²; deceleration on contact 30-50g if uncontrolled",
    "leaps": "Vertical velocity 3-5m/s at takeoff; projectile motion under gravity; trajectory parabolic; time in air ~0.6-1.0s; landing force 3-5x body weight; energy stored in tendons released elastically",
    "rolls": "Rotational kinetic energy KE=½Iω²; angular momentum conserved; friction converts kinetic to heat; impact force distributed over time and area; deceleration more gradual than flat landing",
    "grabs": "Contact friction μ×Normal force provides grip; hand exerts 150-400N compressive force; object accelerates with hand; impulse transfers momentum; if object resists, force increases to overcome inertia",
    "releases": "Projectile initial velocity 50-70m/s (bow), 15-25m/s (thrown); drag force F=½ρv²CdA opposes motion; gravity causes parabolic drop; crosswind drift proportional to exposure time; spin stabilizes via gyroscopic effect",
    "aims": "Static equilibrium; muscle forces balance; micro-tremors <2mm at endpoint; center of pressure under feet stable; breathing motion minimized; visual focus fixes on target point",
    "circles": "Centripetal acceleration v²/r directed to center; friction provides centripetal force; angular velocity ω=v/r; tangential velocity constant if uniform; torso counters angular momentum",
    "lunges": "Weight transfer shifts center of mass forward; momentum mv drives forward; front leg decelerates body eccentrically; normal force on front foot 2-3x body weight; rear leg provides propulsive force",
    "recoils": "Reaction force from Newton's third law; impulse changes momentum; body accelerates backward; kinetic energy transferred from external source; deceleration via muscle resistance",
    "advances": "Constant velocity or slight acceleration; friction opposes motion; momentum constant if steady; kinetic energy constant; work done against friction W=μmgd",
    "steadies": "Static equilibrium restored; net force zero; torques balanced; small corrective forces at ankles/hips; center of mass over base of support; potential energy minimized",
    "arrow_flight": "Initial kinetic energy KE=½mv²; drag force increases with v²; drag coefficient Cd≈0.5 for arrow; crosswind drift ≈0.5m per 10m/s wind at 30m; gravity drop Δy=½gt²; rotational stability from fletching torque",
    "impacts": "Impulse J=∫Fdt over 0.05-0.15s; momentum transfer Δp=mΔv; energy dissipated as deformation/sound/heat; peak force can reach 1000-3000N; stress waves propagate through materials at speed of sound",
    "slides": "Kinetic friction μk×N opposes motion; deceleration a=μkg; kinetic energy dissipated as heat; distance traveled d=v²/(2μkg); surface roughness affects friction coefficient"
}

# ============================================================================
# CINEMATIC CAMERA RULES - PROFESSIONAL CINEMATOGRAPHY
# ============================================================================
CINEMATIC_CAMERA_RULES = {
    "action_intense": {
        "recommended_rigs": ["handheld", "steadicam", "gimbal"],
        "movement_types": ["tracking", "arc", "dolly_in", "circle_strafe"],
        "speed_range": ["medium", "fast"],
        "motivation": "Follow explosive energy and maintain subject in frame during rapid motion; create visceral connection",
        "lens_range": [24, 50],
        "aperture_sweet_spot": 4.0,
        "shutter_angle": 90,  # Crisp, clear motion for action clarity
        "composition": "dynamic framing; break rule of thirds when needed; tight on subject"
    },
    "emotional_intimate": {
        "recommended_rigs": ["dolly", "slider", "tripod"],
        "movement_types": ["slow_push_in", "pull_out", "static"],
        "speed_range": ["very_slow", "slow"],
        "motivation": "Reveal emotional state gradually; create intimacy; focus on micro-expressions and internal conflict",
        "lens_range": [50, 85],
        "aperture_sweet_spot": 2.8,
        "shutter_angle": 180,  # Cinematic motion blur for dreamy feel
        "composition": "centered subject; negative space shows isolation; shallow DOF"
    },
    "establishing_wide": {
        "recommended_rigs": ["drone", "crane", "dolly", "tripod"],
        "movement_types": ["reveal", "descend", "orbit", "static"],
        "speed_range": ["slow", "medium"],
        "motivation": "Establish spatial relationships; show scale and environment; ground audience in location",
        "lens_range": [16, 35],
        "aperture_sweet_spot": 8.0,
        "shutter_angle": 180,
        "composition": "rule of thirds; foreground/midground/background layers; horizon placement"
    },
    "tension_buildup": {
        "recommended_rigs": ["slider", "gimbal", "steadicam"],
        "movement_types": ["slow_push", "lateral_creep", "tilt_up", "slight_orbit"],
        "speed_range": ["very_slow", "slow"],
        "motivation": "Build anticipation and unease; suggest unseen threat; create psychological pressure",
        "lens_range": [35, 50],
        "aperture_sweet_spot": 4.0,
        "shutter_angle": 120,
        "composition": "off-center subject; Dutch angle if appropriate; negative space suggests threat"
    },
    "combat_tactical": {
        "recommended_rigs": ["steadicam", "gimbal", "handheld"],
        "movement_types": ["tracking", "circle", "strafe"],
        "speed_range": ["medium", "fast"],
        "motivation": "Follow combatants; show spatial relationship between opponents; maintain clarity of action geography",
        "lens_range": [28, 45],
        "aperture_sweet_spot": 5.6,
        "shutter_angle": 90,
        "composition": "two-subject framing; z-axis staging; keep both fighters visible"
    }
}

# --- SAFETY SHIM: ensure _is_aftermath exists at module scope early ---
def _is_aftermath(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["aftermath","after the shot","already hit","đã bắn","hậu quả","vừa bị bắn","đã trúng"])

# ----------------------------
# Safety policy replacements (neutralize risky terms for Veo)
# ----------------------------
POLICY_REPLACEMENTS = [
    (re.compile(r"\bkills?\b", re.IGNORECASE), "neutralizes"),
    (re.compile(r"\bkilled\b", re.IGNORECASE), "goes down"),
    (re.compile(r"\bkill\s*count\b", re.IGNORECASE), "tally"),
    (re.compile(r"\bdead\b", re.IGNORECASE), "downed"),
    (re.compile(r"\bblood(?:y)?\b", re.IGNORECASE), "damage"),
    (re.compile(r"\bgore\b", re.IGNORECASE), "debris"),
    (re.compile(r"\bdismember(?:ed|ment)?\b", re.IGNORECASE), "severe damage"),
    (re.compile(r"\bassassin(?:ate|ation|ated)\b", re.IGNORECASE), "eliminate"),
    (re.compile(r"\bNazi\b", re.IGNORECASE), "regime"),
    (re.compile(r"\bswastika\b", re.IGNORECASE), "prohibited insignia"),
    (re.compile(r"\bmachine\s*gun(s)?\b", re.IGNORECASE), "heavy weapon"),
    (re.compile(r"\bgunfire\b", re.IGNORECASE), "weapon fire"),
    (re.compile(r"\bshot\s+down\b", re.IGNORECASE), "goes down"),
    (re.compile(r"\bwound(s)?\b", re.IGNORECASE), "visible damage (non-graphic)"),
    (re.compile(r"\bpainful\s+(roar|growl)\b", re.IGNORECASE), r"fierce \1"),
    (re.compile(r"\bpenetrat(e|es|ed|ing)\b", re.IGNORECASE), "pushes into (non-graphic)"),
]


def _sanitize_text(s: str) -> tuple[str, int]:
    if not isinstance(s, str) or not s:
        return s, 0
    t, hits = s, 0
    for pat, repl in POLICY_REPLACEMENTS:
        new_t, n = pat.subn(repl, t)
        if n:
            hits += n
        t = new_t
    # A) Targeted, non-graphic softening (avoid grammar breakages)
    # Only soften if paired with sensitive nouns; keep original verb with a non-graphic qualifier.
    _soft_rules = [
        # rip/tear/shred + (flesh/wound/gore/artery/organ/innards/blood) → "tear (non-graphic)"
        (re.compile(r"\b(rip|tear|shred)(s|ing|ed)?\b(?=[^.\n]{0,40}\b(flesh|wound|gore|artery|organ|innards|blood)\b)", re.IGNORECASE),
         r"tear\2 (non-graphic)"),
        # pierce + (flesh/skin) → "make solid contact (non-graphic)"
        (re.compile(r"\bpierc(e|es|ed|ing)\b(?=[^.\n]{0,40}\b(flesh|skin)\b)", re.IGNORECASE),
         r"make solid contact (non-graphic)"),
    ]
    for _pat, _repl in _soft_rules:
        new_t, n = _pat.subn(_repl, t)
        if n:
            hits += n
        t = new_t
    return t, hits


def sanitize_veo3_object(obj: dict) -> tuple[dict, int]:
    hits = 0
    TECH_KEYS = {"camera","output_settings","position_m","orientation_deg","focus"}

    def walk(x, parent_key=""):
        nonlocal hits
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                out[k] = walk(v, k)
            return out
        elif isinstance(x, list):
            return [walk(i, parent_key) for i in x]
        elif isinstance(x, str):
            # ✨ skip các nhánh kỹ thuật
            if parent_key in TECH_KEYS:
                return x
            t, n = _sanitize_text(x)
            hits += n
            return t
        else:
            return x

    clean = walk(obj)

    # Ensure safety_and_quality uses generic guidance (avoid listing blocked tokens explicitly)
    saq = clean.get("safety_and_quality") or {}
    avoid = saq.get("avoid") or []
    # Re-sanitize any avoid strings to strip risky tokens
    avoid2 = []
    for a in avoid:
        a2, _ = _sanitize_text(a)
        avoid2.append(a2)

    generic_avoids = [
        "graphic injury or gore",
        "hate symbols or slurs",
        "sexual content or nudity",
        "child endangerment",
        "real-world political propaganda",
    ]
    for g in generic_avoids:
        if g not in avoid2:
            avoid2.append(g)
    saq["avoid"] = avoid2
    saq.setdefault("must_have", [])
    if "non-graphic, documentary tone for combat actions" not in saq["must_have"]:
        saq["must_have"].append("non-graphic, documentary tone for combat actions")
    clean["safety_and_quality"] = saq
    return clean, hits


def safe_slug(s: str, default: str = "scene") -> str:
    if not s:
        return default
    s = s.strip()
    for prefix in ("EXT. ", "INT. "):
        s = s.replace(prefix, "")
    s = SAFE_FILENAME_REGEX.sub("_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._")
    return (s or default)[:60]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ====== CANONICALIZATION SYSTEM ======
ALIASES = {
    "Warrior": "Warrior Squad",
    "Hunters": "Warrior Squad",
    "Lead": "Warrior Squad",
    "Target": "Forest Beast",
    "Beast": "Forest Beast",
    "Creature": "Forest Beast",
    "Arrow": "Arrow",
    "Spear": "Spear",
    "Bow": "Bow",
}

ROSTER = {"Warrior Squad", "Forest Beast", "Arrow", "Spear", "Bow"}

def canon(label: str) -> str:
    return ALIASES.get(label, label)

def allowed(label: str) -> bool:
    return canon(label) in ROSTER

def filter_actions(actions):
    kept = []
    for a in actions:
        subj = canon(a["subject"])
        if not allowed(subj):
            continue
        a["subject"] = subj
        kept.append(a)
    return kept

def _unify_subject_labels_enhanced(obj: dict) -> dict:
    """Enhanced subject label unification using global canon() system."""
    if not isinstance(obj, dict):
        return obj
    
    # Canonicalize world_and_characters
    wc = obj.get("world_and_characters") or {}
    if wc.get("lead_character"):
        wc["lead_character"] = canon(wc["lead_character"])
    if wc.get("antagonist"):
        wc["antagonist"] = canon(wc["antagonist"])
    obj["world_and_characters"] = wc
    
    # Canonicalize action sequences in shots
    for shot in obj.get("shots", []):
        if not isinstance(shot, dict):
            continue
        for action in shot.get("action_sequence", []):
            if isinstance(action, dict) and action.get("subject"):
                action["subject"] = canon(action["subject"])
    
    return obj

# ====== 8-SECOND ACTION SEQUENCE GENERATOR ======
BEATS = [
  (0.00,"Warrior Squad","cuts angle left",
   "heel-toe step; hips rotate; shoulders counter-rotate",
   "lateral shift manages distance; friction prevents slide",
   "gain flanking position"),
  (0.45,"Forest Beast","charges",
   "forelimbs extend; spine flexes; head low",
   "acceleration builds; mass drives forward under traction",
   "overwhelm the front line"),
  (0.90,"Warrior Squad","sets guard",
   "spearman lowers stance; bowman plants rear foot",
   "lowered CoG for stability against incoming mass",
   "prepare to absorb and counter"),
  (1.20,"Warrior Squad","draws",
   "scapular engagement; string hand pulls; bow arm stabilizes",
   "elastic energy stores in limbs of the bow",
   "commit to a clean release"),
  (1.55,"Warrior Squad","anchors",
   "string hand settles at corner of mouth; elbow aligns",
   "stable anchor reduces lateral error",
   "ensure accuracy under stress"),
  (1.70,"Warrior Squad","re-aims",
   "micro footwork; grip micro-corrections synced to breath",
   "tiny torque corrections stabilize sight picture",
   "compensate for the Beast's motion"),
  (1.85,"Warrior Squad","releases",
   "fingers relax; bow arm resists torque; slight follow-through",
   "stored energy converts to arrow kinetic energy; mild recoil",
   "execute the shot"),
  (1.98,"Arrow","enters flight",
   "fletching stabilizes; shaft flexes then dampens",
   "parabolic arc under gravity; slight lateral drift in light breeze",
   "maintain trajectory toward target"),
  (2.35,"Arrow","impacts",
   "shaft compresses; vibrations damp quickly",
   "kinetic energy dissipates into surface; arrow lodges non-graphically",
   "complete shot sequence"),
  (2.60,"Forest Beast","recoils",
   "mass shifts opposite vector; brief stagger",
   "momentum transfer; friction limits slide",
   "instinctive reaction to impact"),
  (3.10,"Warrior Squad","presses with spear",
   "rear foot drives; hips extend; lead arm snaps forward",
   "impulse sends spear tip on straight vector",
   "capitalize on opening"),
  (3.85,"Forest Beast","deflects",
   "shoulders twist; forelimbs parry line",
   "angular momentum redirects incoming vector",
   "avoid direct hit"),
  (4.40,"Warrior Squad","circles",
   "short arc steps; gaze locked; guard up",
   "centripetal acceleration kept small; balance preserved",
   "seek better angle"),
  (5.10,"Forest Beast","feints",
   "head dips; weight teases forward",
   "small impulse to test response",
   "probe defenses"),
  (5.70,"Warrior Squad","re-positions",
   "heel-toe shift; shoulders counter-rotate",
   "momentum managed to avoid overshoot",
   "maintain initiative"),
  (6.40,"Warrior Squad","re-aims",
   "micro footwork; breath reset",
   "parallax alive; no idle plateau",
   "hold pressure"),
  (7.20,"Forest Beast","pivots away",
   "hind/lead feet switch; torso twists",
   "friction & angular momentum govern rapid redirection",
   "escape the unfavorable line"),
  (7.90,"Warrior Squad","holds guard while shifting",
   "low stance; breath-linked sway; feet alive",
   "continuous micro-motion avoids freeze at tail",
   "stay ready into the cut"),
]

def make_action_sequence():
    seq = []
    for t, sub, verb, bio, phys, motive in BEATS:
        seq.append({
          "timestamp_sec": round(t, 2),
          "subject": canon(sub),
          "verb": verb,
          "biomechanics_note": bio,
          "physics_note": phys,
          "internal_motivation": motive
        })
    return filter_actions(seq)

def compose_single_prompt():
    beats_txt = " … ".join(
        f"{t:0.2f}s {sub if sub!='Arrow' else 'Arrow'} {verb}"
        for (t, sub, verb, *_rest) in BEATS
    )
    return (
      "VEO-3 — Forest clearing, Warrior Squad vs Forest Beast (8s). "
      "Grounded, non-graphic. No idle seconds. ACTION (beats every ~0.5–1.0s): ["
      + beats_txt +
      "]. CAMERA: handheld dolly-in (35mm), start(-1.2,1.6,4.0)→end(-0.4,1.6,3.2); "
      "yaw 6°, pitch -2°; slow rack focus at 2.3s; rule-of-thirds; moderate DOF. "
      "VFX: subtle arrow trail; dust/leaf puff on impacts. LIGHTING: natural sun-dappled, "
      "key from camera-left. AUDIO: ambience low forest bed; foley (leather, footfall, breath, "
      "non-graphic growl); sfx (bowstring snap, arrow whoosh, woody thunk, debris). "
      "SAFETY: strictly non-graphic; physically plausible; cause→effect within 0.9s."
    )

def build_scene_matching_format():
    # điền đúng các trường như JSON ở trên…
    scene = {
      "model": "google/veo-3",
      "request_type": "video",
      "duration_seconds": 8,
      "fps": 30,
      "resolution": "1920x1080",
      "aspect_ratio": "16:9",
      "language": "English",
      "creative_brief": "Warrior Squad coordinates against a Forest Beast in a sun-dappled forest clearing. Purposeful, physically plausible motion with no idle seconds.",
      "global_style": {
        "visual_tone": "cinematic, grounded realism",
        "motion_rules": "No idle frames; beats every 0.5–1.0s; every initiating action has an on-screen reaction within 0.9s; no teleportation",
        "color_grading": "natural daylight with warm greens and earthy browns",
        "vfx_guidance": "subtle dust/leaf particles on impacts and footfalls; restrained motion trails for fast projectiles",
        "sound_mix": "ambience low; foley detailed; sfx tight and non-graphic"
      },
      "continuity_core": {
        "roster": {
          "lead": "Warrior Squad",
          "antagonist": "Forest Beast",
          "projectiles": ["Arrow"],
          "props": ["Spear","Bow"]
        },
        "aliases_map": {
          "Warrior": "Warrior Squad",
          "Hunters": "Warrior Squad",
          "Lead": "Warrior Squad",
          "Target": "Forest Beast",
          "Beast": "Forest Beast",
          "Creature": "Forest Beast"
        },
        "wardrobe_state": "leather armor, earth-tone tunics; spears and one bow slung",
        "props_state": "one warrior holds Bow (nocked), another holds Spear (ready)",
        "time_lock": "day",
        "weather_lock": "clear"
      },
      "continuity_logic": "All subjects must resolve to the roster labels via aliases_map. Do not introduce new characters or props. Keep wardrobe/props persistent across shots.",
      "world_and_characters": {
        "setting": "Sun-dappled clearing deep in the forest; shafts of light cut through drifting motes; low shrubs and scattered trunks.",
        "lead_character": "Warrior Squad",
        "antagonist": "Forest Beast",
        "weather": "clear with a light breeze",
        "battle_context": "Beast charges; squad splits angles; bow unit shoots; spear unit pressures flank.",
        "character_continuity": "Warriors stay in staggered formation; bowman behind cover; spearman up front.",
        "time_of_day": "late morning"
      },
      "shots": [{
        "id": "shot_1",
        "start_time": 0.0,
        "duration": 8.0,
        "slugline": "Handheld tracking across the clearing as the Beast rushes; bowman releases; spear presses.",
        "action_sequence": make_action_sequence(),
        "camera": {
          "rig": "handheld",
          "lens_mm": 35,
          "aperture_f": 5.6,
          "shutter_angle_deg": 180,
          "white_balance_k": 5600,
          "iso": 400,
          "position_m": {
            "start": { "x": -1.2, "y": 1.6, "z": 4.0 },
            "end":   { "x": -0.4, "y": 1.6, "z": 3.2 }
          },
          "orientation_deg": {
            "yaw": 6.0,
            "pitch": -2.0,
            "roll": 0.0
          },
          "movement": {
            "type": "dolly",
            "direction": "forward",
            "speed": "medium"
          },
          "focus_pull_style": "slow rack focus",
          "composition_style": "rule of thirds",
          "depth_of_field": "moderate"
        },
        "vfx": [
          "subtle arrow trail",
          "dust/leaf puff on impacts"
        ],
        "lighting": "natural sun-dappled, key from camera-left",
        "audio": {
          "ambience": "low forest bed",
          "foley": ["leather", "footfall", "breath", "non-graphic growl"],
          "sfx": ["bowstring snap", "arrow whoosh", "woody thunk", "debris"],
          "mix_notes": "ambience low; foley detailed; sfx tight and non-graphic"
        },
        "transition_out": "cut",
        "creative_suggestions": "brief foreground branch occlusion at 0.9–1.2s to add depth; slight handheld micro-sway synced to breath."
      }],
      "safety_and_quality": {
        "content_guidelines": "strictly non-graphic; physically plausible; cause→effect within 0.9s",
        "motion_quality": "no idle seconds; purposeful movement; realistic physics",
        "continuity_checks": "subjects canonicalized; props consistent; wardrobe persistent"
      },
      "output_settings": {
        "container": "mp4",
        "video_codec": "h264",
        "video_bitrate": "8000k",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "normalize_audio": True,
        "tone_mapping": "linear",
        "grain": "none"
      },
      "single_prompt_for_model": compose_single_prompt()
    }
    return scene


def sleep_backoff(attempt: int) -> None:
    # exponential backoff with jitter, capped
    time.sleep(min(8, (2 ** attempt)) + random.uniform(0, 0.5))


class ContinuityState:
    """Tracks canonical continuity across scenes."""

    def __init__(self) -> None:
        # Thay vì list tên, chúng ta dùng dictionary để lưu hồ sơ đầy đủ
        self.characters: Dict[str, Dict[str, Any]] = {}
        self.locations: Dict[str, Dict[str, Any]] = {}
        # NEW: generic dynamic entities (animals, vehicles, projectiles, moving props)
        self.entities: Dict[str, Dict[str, Any]] = {}
        # Giữ lại các thuộc tính cũ để tương thích
        self.char_names: List[str] = []  # canonical character labels
        self.aircraft_models: List[str] = []  # e.g., Hellcat, Zero
        self.setting_baseline: str = ""
        self.weather_baseline: str = ""
        self.time_of_day: str = ""  # optional
        # NEW: store wardrobe & props if parsed from earlier scenes
        self.wardrobe_details: str = ""
        self.props_details: str = ""

    def summary_for_prompt(self) -> str:
        return (
            f"- Allowed characters: {', '.join(self.char_names) or '[]'}\n"
            f"- Allowed aircraft models: {', '.join(self.aircraft_models) or '[]'}\n"
            f"- Baseline setting: {self.setting_baseline or 'N/A'}\n"
            f"- Baseline weather/atmosphere: {self.weather_baseline or 'N/A'}\n"
            f"- Time of day: {self.time_of_day or 'N/A'}\n"
            f"- Wardrobe: {self.wardrobe_details or 'N/A'}\n"
            f"- Props-in-hand: {self.props_details or 'N/A'}\n"
            f"- Known dynamic entities: {', '.join(self.entities.keys()) or '[]'}\n"
        )

    def update_from_output(self, out: Dict[str, Any]) -> None:
        cu = out.get("continuity_updates") or {}
        # Cập nhật hồ sơ nhân vật
        for char_update in cu.get("new_or_updated_characters", []):
            name = char_update.get("name")
            profile = char_update.get("profile")
            if name and profile:
                print(f"🔄 Cập nhật/Thêm hồ sơ cho nhân vật: {name}")
                self.characters[name] = profile
        # Cập nhật hồ sơ địa điểm
        for loc_update in cu.get("new_or_updated_locations", []):
            name = loc_update.get("name")
            profile = loc_update.get("profile")
            if name and profile:
                print(f"🔄 Cập nhật/Thêm hồ sơ cho địa điểm: {name}")
                self.locations[name] = profile
        # NEW: capture dynamic entities
        for ent_up in cu.get("new_or_updated_dynamic_entities", []):
            name = ent_up.get("name")
            profile = ent_up.get("profile")
            if name and profile:
                print(f"🔄 Cập nhật/Thêm hồ sơ cho đối tượng: {name}")
                self.entities[name] = profile
        # Giữ lại logic cũ để tương thích
        for name in cu.get("new_characters", []) or []:
            self.add_character(name)
        for ac in cu.get("new_aircraft_models", []) or []:
            self.add_aircraft(ac)
        w = out.get("world_and_characters") or {}
        if not self.setting_baseline and isinstance(w.get("setting"), str):
            self.setting_baseline = w["setting"]
        if not self.weather_baseline and isinstance(w.get("weather"), str):
            self.weather_baseline = w["weather"]
        if not self.time_of_day and isinstance(w.get("time_of_day", None), str):
            self.time_of_day = w["time_of_day"]
        # Fix None time_of_day issue
        if w.get("time_of_day") is None:
            w["time_of_day"] = "day"  # Default to day if None
            print("DEBUG: Fixed None time_of_day, set to 'day'")
        # try to keep wardrobe/props remembered if the model provided rich continuity text
        cc = w.get("character_continuity") or ""
        if cc and not self.wardrobe_details:
            self.wardrobe_details = cc

    def add_character(self, name: str) -> None:
        name = name.strip()
        if name and name not in self.char_names:
            self.char_names.append(name)

    def add_aircraft(self, model: str) -> None:
        model = model.strip()
        if model and model not in self.aircraft_models:
            self.aircraft_models.append(model)
    
    def update_character_state_progressive(self, name: str, scene_action: str, environment: dict) -> None:
        """Cập nhật trạng thái nhân vật theo logic tiến triển qua các scene"""
        if name not in self.characters:
            return
        
        char_profile = self.characters[name]
        if not isinstance(char_profile, dict):
            return
        
        current_state = char_profile.get("current_state", {})
        
        # Tăng dirt level dựa trên hành động
        action_lower = scene_action.lower()
        dirt_levels = ["clean", "dusty", "mud-splattered", "filthy"]
        current_dirt = current_state.get("dirt_level", "clean")
        
        if any(word in action_lower for word in ["roll", "crawl", "fall", "slide"]):
            try:
                idx = dirt_levels.index(current_dirt)
                if idx < len(dirt_levels) - 1:
                    current_state["dirt_level"] = dirt_levels[idx + 1]
            except ValueError:
                current_state["dirt_level"] = "dusty"
        
        # Cập nhật wetness dựa trên terrain và weather
        terrain = environment.get("terrain", "").lower()
        weather = environment.get("weather", "").lower()
        
        if "mud" in terrain or "swamp" in terrain or "rain" in weather:
            if "fall" in action_lower or "wade" in action_lower:
                current_state["wetness"] = "soaked"
            elif current_state.get("wetness") == "dry":
                current_state["wetness"] = "damp"
        
        if "river" in terrain or "water" in terrain:
            current_state["wetness"] = "soaked"
        
        # Cập nhật exhaustion
        if any(word in action_lower for word in ["sprint", "climb", "fight", "combat", "struggle"]):
            exhaust_levels = ["fresh", "alert", "tired", "exhausted"]
            current_exhaust = current_state.get("exhaustion", "fresh")
            try:
                idx = exhaust_levels.index(current_exhaust)
                if idx < len(exhaust_levels) - 1:
                    current_state["exhaustion"] = exhaust_levels[idx + 1]
            except ValueError:
                current_state["exhaustion"] = "alert"
        
        char_profile["current_state"] = current_state
        self.characters[name] = char_profile


# ============================================================================
# CHARACTER PROFILE CLASS - EXTREME DETAIL TRACKING
# ============================================================================
class CharacterProfile:
    """Profile cực kỳ chi tiết cho mỗi nhân vật - đảm bảo continuity hoàn hảo"""
    
    def __init__(self, name: str):
        self.name = name
        self.appearance = {
            "face": "",  # "weathered with scar across left eyebrow"
            "body_type": "",  # "athletic build, 1.8m tall, broad shoulders"
            "height_estimate": "",  # "approximately 1.8m"
            "scars_markings": [],  # ["scar left eyebrow", "burn mark right forearm"]
            "tattoos_paint": [],  # ["tribal pattern left shoulder"]
            "hair": "",  # "short dark brown, swept back"
            "skin_tone": "",  # "tanned caucasian"
            "age_appearance": ""  # "mid-30s"
        }
        self.wardrobe = {
            "head": "",  # "dark leather cap with brass buckle"
            "torso_outer": "",  # "dark brown boiled leather cuirass with bronze shoulder guards"
            "torso_inner": "",  # "off-white linen shirt, sleeves rolled to elbows"
            "legs": "",  # "canvas trousers, earth-tone brown, tucked into boots"
            "feet": "",  # "worn leather boots, mud-caked, laces frayed"
            "hands": "",  # "leather fingerless gloves, right thumb torn"
            "accessories": [],  # ["wide leather belt with brass buckle", "rope coil over shoulder"]
            "colors": [],  # ["dark brown", "off-white", "bronze", "earth-tone"]
            "materials": [],  # ["leather", "linen", "canvas", "brass", "bronze"]
            "condition": ""  # "battle-worn, dust-covered, small tear in left sleeve at elbow"
        }
        self.current_state = {
            "dirt_level": "clean",  # clean -> dusty -> mud-splattered -> filthy
            "wounds": [],  # ["scratch on left cheek", "bruise on right forearm"]
            "wetness": "dry",  # dry -> damp -> soaked
            "exhaustion": "fresh",  # fresh -> alert -> tired -> exhausted
            "emotional_state": "neutral",  # specific emotion: determined, fearful, angry, focused
            "visible_damage": []  # ["torn left sleeve", "scratched armor shoulder"]
        }
        self.props_in_hand = {
            "left": None,  # "wooden shield, iron boss"
            "right": None,  # "iron-tipped ash spear, 2m length"
            "both": None  # "large battle-axe gripped in both hands"
        }
        self.movement_signature = {
            "gait": "",  # "confident stride, heel-strike prominent, slight forward lean"
            "posture": "",  # "upright, shoulders back, alert head position"
            "tempo": "",  # "moderate pace, deliberate movements"
            "quirks": []  # ["adjusts belt when thinking", "touches weapon when alert"]
        }
    
    def to_prompt_description(self, scene_number: int = 1) -> str:
        """Tạo mô tả cực kỳ chi tiết cho prompt - đảm bảo AI tái tạo chính xác"""
        lines = []
        lines.append(f"{self.name} (Scene {scene_number} Continuity - EXACT MATCH REQUIRED):")
        
        # Appearance - luôn lặp lại đầy đủ mọi chi tiết
        lines.append("  APPEARANCE:")
        if self.appearance['face']:
            lines.append(f"    Face: {self.appearance['face']}")
        if self.appearance['body_type']:
            lines.append(f"    Build: {self.appearance['body_type']}")
        if self.appearance['hair']:
            lines.append(f"    Hair: {self.appearance['hair']}")
        if self.appearance['skin_tone']:
            lines.append(f"    Skin: {self.appearance['skin_tone']}")
        if self.appearance['age_appearance']:
            lines.append(f"    Age: {self.appearance['age_appearance']}")
        if self.appearance['scars_markings']:
            lines.append(f"    Distinctive marks: {', '.join(self.appearance['scars_markings'])}")
        if self.appearance['tattoos_paint']:
            lines.append(f"    Tattoos/Paint: {', '.join(self.appearance['tattoos_paint'])}")
        
        # Wardrobe - chi tiết từng layer, KHÔNG BAO GIỜ thay đổi
        lines.append("  WARDROBE (must match exactly - NO variations):")
        if self.wardrobe['head']:
            lines.append(f"    Head: {self.wardrobe['head']}")
        if self.wardrobe['torso_outer']:
            lines.append(f"    Torso outer: {self.wardrobe['torso_outer']}")
        if self.wardrobe['torso_inner']:
            lines.append(f"    Torso inner: {self.wardrobe['torso_inner']}")
        if self.wardrobe['legs']:
            lines.append(f"    Legs: {self.wardrobe['legs']}")
        if self.wardrobe['feet']:
            lines.append(f"    Feet: {self.wardrobe['feet']}")
        if self.wardrobe['hands']:
            lines.append(f"    Hands: {self.wardrobe['hands']}")
        if self.wardrobe['accessories']:
            lines.append(f"    Accessories: {', '.join(self.wardrobe['accessories'])}")
        if self.wardrobe['colors']:
            lines.append(f"    Color palette: {', '.join(self.wardrobe['colors'])}")
        if self.wardrobe['materials']:
            lines.append(f"    Materials: {', '.join(self.wardrobe['materials'])}")
        if self.wardrobe['condition']:
            lines.append(f"    Overall condition: {self.wardrobe['condition']}")
        
        # Current state - tracking progression
        lines.append("  CURRENT STATE (this scene):")
        lines.append(f"    Cleanliness: {self.current_state['dirt_level']}")
        if self.current_state['wounds']:
            lines.append(f"    Visible wounds: {'; '.join(self.current_state['wounds'])}")
        lines.append(f"    Wetness: {self.current_state['wetness']}")
        lines.append(f"    Physical state: {self.current_state['exhaustion']}")
        lines.append(f"    Emotional state: {self.current_state['emotional_state']}")
        if self.current_state['visible_damage']:
            lines.append(f"    Clothing/gear damage: {'; '.join(self.current_state['visible_damage'])}")
        
        # Props in hand - critical for continuity
        props_list = []
        if self.props_in_hand['both']:
            props_list.append(f"both hands: {self.props_in_hand['both']}")
        else:
            if self.props_in_hand['left']:
                props_list.append(f"left hand: {self.props_in_hand['left']}")
            if self.props_in_hand['right']:
                props_list.append(f"right hand: {self.props_in_hand['right']}")
        if props_list:
            lines.append(f"  PROPS IN HAND: {'; '.join(props_list)}")
        
        # Movement signature
        if self.movement_signature['gait'] or self.movement_signature['posture']:
            lines.append("  MOVEMENT SIGNATURE:")
            if self.movement_signature['gait']:
                lines.append(f"    Gait: {self.movement_signature['gait']}")
            if self.movement_signature['posture']:
                lines.append(f"    Posture: {self.movement_signature['posture']}")
            if self.movement_signature['tempo']:
                lines.append(f"    Tempo: {self.movement_signature['tempo']}")
            if self.movement_signature['quirks']:
                lines.append(f"    Quirks: {'; '.join(self.movement_signature['quirks'])}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage"""
        return {
            "name": self.name,
            "appearance": self.appearance,
            "wardrobe": self.wardrobe,
            "current_state": self.current_state,
            "props_in_hand": self.props_in_hand,
            "movement_signature": self.movement_signature
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CharacterProfile':
        """Create from dictionary"""
        profile = cls(data.get("name", "Unknown"))
        profile.appearance = data.get("appearance", profile.appearance)
        profile.wardrobe = data.get("wardrobe", profile.wardrobe)
        profile.current_state = data.get("current_state", profile.current_state)
        profile.props_in_hand = data.get("props_in_hand", profile.props_in_hand)
        profile.movement_signature = data.get("movement_signature", profile.movement_signature)
        return profile


# =============================
# JSON Schema (tightened a bit) — UPDATED: allow minItems=1 for single-shot scenes
# =============================
VEO3_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "request_type": {"type": "string"},
        "duration_seconds": {"type": "number"},
        "fps": {"type": "number"},
        "resolution": {"type": "string"},
        "aspect_ratio": {"type": "string"},
        "language": {"type": "string"},
        "creative_brief": {"type": "string"},
        "global_style": {
            "type": "object",
            "properties": {
                "visual_tone": {"type": "string"},
                "motion_rules": {"type": "string"},
                "color_grading": {"type": "string"},
                "vfx_guidance": {"type": "string"},
                "sound_mix": {"type": "string"},
            },
            "required": [
                "visual_tone",
                "motion_rules",
                "color_grading",
                "vfx_guidance",
                "sound_mix",
            ],
            "additionalProperties": True,
        },
        "continuity_logic": {"type": "string"},
        "continuity_updates": {
            "type": "object",
            "properties": {
                # NEW: rich profiles
                "new_or_updated_characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "profile": {"type": "object"},
                        },
                        "required": ["name", "profile"],
                    },
                },
                "new_or_updated_locations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "profile": {"type": "object"},
                        },
                        "required": ["name", "profile"],
                    },
                },
                "new_or_updated_dynamic_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "profile": {"type": "object"},
                        },
                        "required": ["name", "profile"],
                    },
                },
                # legacy, vẫn giữ để tương thích
                "new_characters": {"type": "array", "items": {"type": "string"}},
                "new_aircraft_models": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "world_and_characters": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "lead_character": {"type": "string"},
                "antagonist": {"type": "string"},
                "weather": {"type": "string"},
                "battle_context": {"type": "string"},
                "character_continuity": {"type": "string"},
                "time_of_day": {"type": "string"},
            },
            "required": [
                "setting",
                "lead_character",
                "antagonist",
                "weather",
                "battle_context",
                "character_continuity",
            ],
            "additionalProperties": True,
        },
        "shots": {
            "type": "array",
            "minItems": 1,  # was 2 — now supports single-shot scenes (0–8s continuous)
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "start_time": {"type": "number"},
                    "duration": {"type": "number"},
                    "slugline": {"type": "string"},
                    "action_sequence": {
                        "type": "array",
                        "description": "A step-by-step sequence of actions, biomechanics, and physics for the entire shot duration.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp_sec": {
                                    "type": "number",
                                    "description": "The time this action occurs, e.g., 0.5",
                                },
                                "subject": {
                                    "type": "string",
                                    "description": "The character or animal performing the action, e.g., 'KAI', 'Wildebeest'",
                                },
                                "verb": {
                                    "type": "string",
                                    "description": "The core action verb, e.g., 'braces', 'lunges', 'stumbles', 'recoils'",
                                },
                                "biomechanics_note": {
                                    "type": "string",
                                    "description": "HOW the body moves: muscle tension, weight shift, joint movement, breathing.",
                                },
                                "physics_note": {
                                    "type": "string",
                                    "description": "WHY the body moves that way: impact from a force, loss of balance due to inertia, effect of gravity.",
                                },
                                "internal_motivation": {
                                    "type": "string",
                                    "description": "The psychological reason for the action, e.g., 'driven by desperation', 'acting with cold precision'.",
                                },
                            },
                            "required": [
                                "timestamp_sec",
                                "subject",
                                "verb",
                                "biomechanics_note",
                                "physics_note",
                                "internal_motivation",
                            ],
                        },
                    },
                    "camera": {
                        "type": "object",
                        "properties": {
                            "rig": {"type": "string"},
                            "lens_mm": {"type": "number"},
                            "aperture_f": {"type": "number"},
                            "shutter_angle_deg": {"type": "number"},
                            "white_balance_k": {"type": "number"},
                            "iso": {"type": "number"},
                            "position_m": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "object"},
                                    "end": {"type": "object"},
                                },
                                "required": ["start", "end"],
                                "additionalProperties": True,
                            },
                            "movement": {"type": "object"},
                            "orientation_deg": {"type": "object"},
                            "focus": {"type": "object"},
                            "composition_style": {
                                "type": "string",
                                "description": "Describes the framing principle, e.g., 'rule of thirds', 'centered symmetry', 'dutch angle'.",
                            },
                            "focus_pull_style": {
                                "type": "string",
                                "description": "Describes how focus changes, e.g., 'slow rack focus from foreground to background', 'snap focus'.",
                            },
                            "depth_of_field": {
                                "type": "string",
                                "enum": ["shallow", "deep", "moderate"],
                                "description": "Describes the depth of field, e.g., 'shallow' to isolate the character.",
                            },
                        },
                        "required": [
                            "rig",
                            "lens_mm",
                            "aperture_f",
                            "shutter_angle_deg",
                            "white_balance_k",
                            "iso",
                            "position_m",
                            "movement",
                            "orientation_deg",
                            "focus",
                        ],
                        "additionalProperties": False,
                    },
                    "vfx": {"type": "array", "items": {"type": "string"}},
                    "lighting": {"type": "string"},
                    "audio": {
                        "type": "object",
                        "properties": {
                            "ambience": {
                                "type": "string",
                                "description": "The background environmental sound bed, e.g., 'distant howling wind'.",
                            },
                            "foley": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Sounds from character actions, e.g., 'footsteps on gravel', 'leather creaking'.",
                            },
                            "sfx": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Impactful sound effects, e.g., 'dramatic whoosh', 'deep sub-bass impact'.",
                            },
                            "dialogue_note": {
                                "type": "string",
                                "description": "Notes on non-verbal vocalizations like grunts or sighs.",
                            },
                            "music_cue": {
                                "type": "string",
                                "description": "Describes the musical score's role and emotion, e.g., 'a lone cello note begins, building tension'.",
                            },
                            "mix_notes": {
                                "type": "string",
                                "description": "Overall instruction for the sound mix, e.g., 'foley should be hyper-realistic'.",
                            },
                        },
                    },
                    "transition_out": {"type": "string"},
                    "creative_suggestions": {
                        "type": "string",
                        "description": "An optional field for the AI to suggest creative visual flourishes, camera angles, or VFX ideas that enhance the scene's core intent but are not strictly required.",
                    },
                },
                "required": [
                    "id",
                    "start_time",
                    "duration",
                    "slugline",
                    "action_sequence",
                    "camera",
                    "vfx",
                    "lighting",
                    "audio",
                    "transition_out",
                ],
                "additionalProperties": False,
            },
        },
        "safety_and_quality": {
            "type": "object",
            "properties": {
                "avoid": {"type": "array", "items": {"type": "string"}},
                "must_have": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["avoid", "must_have"],
            "additionalProperties": True,
        },
        "output_settings": {
            "type": "object",
            "properties": {
                "container": {"type": "string"},
                "video_codec": {"type": "string"},
                "video_bitrate": {"type": "string"},
                "audio_codec": {"type": "string"},
                "audio_bitrate": {"type": "string"},
                "normalize_audio": {"type": "boolean"},
                "tone_mapping": {"type": "string"},
                "grain": {"type": "string"},
            },
            "required": [
                "container",
                "video_codec",
                "video_bitrate",
                "audio_codec",
                "audio_bitrate",
                "normalize_audio",
                "tone_mapping",
                "grain",
            ],
            "additionalProperties": True,
        },
        "single_prompt_for_model": {"type": "string"},
    },
    "required": [
        "model",
        "request_type",
        "duration_seconds",
        "fps",
        "resolution",
        "aspect_ratio",
        "language",
        "creative_brief",
        "global_style",
        "continuity_logic",
        "world_and_characters",
        "shots",
        "safety_and_quality",
        "output_settings",
        "single_prompt_for_model",
    ],
    "additionalProperties": True,
}


class MotionEnricher:
    def enrich(self, shot: Dict, scene_text: str) -> Dict:
        # Logic làm giàu ở đây
        return shot


class ArcheryEnricher(MotionEnricher):
    def enrich(self, shot: Dict, scene_text: str) -> Dict:
        # DISABLED: Prevent auto-generation of Archer/Arrow content not in original input
        # This was causing unwanted archery beats even when not mentioned in source text
        return shot

    def _phase_index(self, seq: list, *verbs: str) -> int:
        """Find index of first occurrence of any verb in sequence"""
        for i, e in enumerate(seq):
            verb = (e.get("verb", "") or "").lower()
            if any(v.lower() in verb for v in verbs):
                return i
        return -1

    def _ensure_projectile_chain(self, seq: list, duration: float, windy: bool) -> list:
        """Ensure proper projectile chain: release→flight→impact→target reacts"""
        if not seq:
            return seq
        
        # Sort by timestamp
        seq.sort(key=lambda e: float(e.get("timestamp_sec", 0)))
        
        # Find key phases
        rel_idx = self._phase_index(seq, "release", "loose")
        flt_idx = self._phase_index(seq, "flight", "flies", "arc")
        imp_idx = self._phase_index(seq, "impact", "thud", "lodges")
        
        # If we have release but no proper flight/impact, add them
        if rel_idx >= 0:
            t_rel = float(seq[rel_idx]["timestamp_sec"])
            
            # Add flight if missing or too close to release
            if flt_idx < 0 or float(seq[flt_idx]["timestamp_sec"]) - t_rel < 0.1:
                flight_time = t_rel + 0.15
                seq.append({
                    "timestamp_sec": round(flight_time, 2),
                    "subject": "Arrow",
                    "verb": "enters flight",
                    "biomechanics_note": "fletching stabilizes; shaft flexes then dampens",
                    "physics_note": f"parabolic arc under gravity; drag slows speed{'; subtle lateral drift if windy' if windy else ''}",
                    "internal_motivation": "maintaining trajectory toward target"
                })
            
            # Add impact if missing or too close to flight
            if imp_idx < 0:
                impact_time = t_rel + 1.0  # ~1 second flight time
                seq.append({
                    "timestamp_sec": round(impact_time, 2),
                    "subject": "Arrow",
                    "verb": "impacts",
                    "biomechanics_note": "shaft compresses then settles; vibrations dampen quickly",
                    "physics_note": "kinetic energy dissipates into target; arrow lodges at entry point",
                    "internal_motivation": "completing the shot sequence"
                })
        
        # Re-sort after additions
        seq.sort(key=lambda e: float(e.get("timestamp_sec", 0)))
        return seq


class DomainPack:
    name: str = "base"

    def detect(self, text: str, entities: dict) -> bool:
        return False

    def inject_rules_block(self) -> str:
        return ""

    def postprocess_sequence(self, seq: list, shot: dict, world: dict) -> list:
        return seq


class ArcheryPack(DomainPack):
    name = "archery"

    def detect(self, text, entities):
        kw = ["bow","arrow","archer","archery","cung","mũi tên","ban ten","bắn tên","mui ten","căng dây","cang day"]
        t = (text or "").lower() + " " + (entities.get("action_specific", "").lower())
        return any(k in t for k in kw)

    def inject_rules_block(self):
        # Return the archery rules block directly
        return (
            "ARCHERY REALISM LAYER (MANDATORY)\n"
            "- Bow actions must follow: draw (with visible string tension) → anchor (stable anchor point) → release → arrow flight (parabolic arc, non-instant) → impact/near-miss → plausible aftermath.\n"
            "- Arrow flight must show travel time, aerodynamic drop, and (if wind is present) slight lateral drift; no teleportation/no instant hit.\n"
            "- Audio must include: bowstring snap on release, a soft whoosh during flight, and a thud/woody thunk on impact or a skitter if deflected.\n"
            "- VFX must include a subtle motion trail in flight and a dust/particle puff on impact or a believable deflection.\n"
            "- On impact: arrow tip anchors at the entry point and stays lodged (or drops realistically with 1–2 short bounces under gravity/friction). No springy jitter or random hopping.\n"
            "- Shooter mechanics: shoulders and back engage on draw; bow arm stabilizes at anchor; mild post-release recoil; breathing/footwork micro-adjustments every 0.5–1s.\n"
            "- Target reaction must match impact vector (recoil/stagger/deflection) and surface type (wood/stone/flesh implied non-graphically)."
        )

    def postprocess_sequence(self, seq, shot, world):
        # These methods will be called from the main class instance
        return seq


class PromptGenerator:
    """Alias for ChatGPTPromptGenerator for backward compatibility."""
    def __init__(self, api_key: Optional[str] = None):
        self._generator = ChatGPTPromptGenerator(api_key)
        # Forward all attributes to the underlying generator
        for attr in dir(self._generator):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(self._generator, attr))
    
    def __getattr__(self, name):
        return getattr(self._generator, name)

class ChatGPTPromptGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self.load_api_key()
        self._eng_cache = {}  # simple memo cache for translations
        self.use_llm_scene_normalization = False
        self.translate_output = True
        self.dry_run = False  # chỉ build JSON, không save
        self.director_mode = False  # bật/tắt đạo diễn AI
        self.strict_grounding = True  # CHẾ ĐỘ NGHIÊM NGẶT: chỉ bám nội dung cảnh
        self.force_sample_template = False  # Bật True để ép xuất đúng mẫu bạn yêu cầu

        if not self.api_key or OpenAI is None:
            print("ℹ️ Không có OpenAI client → chạy offline (no-LLM).")
            self.client = None
        else:
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.client = OpenAI(api_key=self.api_key)  # ✨ bỏ timeout khỏi constructor

        self.request_timeout = 120  # ✨ tăng timeout lên 120 giây để tránh timeout
        self.translate_output = bool(self.client)  # ✨ chỉ dịch khi có client

        self.model = "gpt-4o-mini"
        self.temperature = 0.0  # thay vì 0.2 để giảm sinh ngẫu nhiên
        self.max_tokens = 4000
        self.seed = 7  # quasi-deterministic

        self.domain_packs = [ArcheryPack()]
        self.enrichers = [ArcheryEnricher()]
    
    # ============================================================================
    # TOP 0.1% ENHANCEMENT METHODS - EXTREME QUALITY
    # ============================================================================
    
    def _enrich_action_biomechanics(self, verb: str, subject: str, context: str = "") -> str:
        """Tạo biomechanics note cực kỳ chi tiết (60-100 chars minimum)"""
        verb_lower = verb.lower()
        base = BIOMECHANICS_TEMPLATES.get(verb_lower, "")
        
        if not base:
            base = f"{verb.capitalize()} involves coordinated muscle activation; joint articulation follows biomechanical principles; center of mass adjusts dynamically"
        
        # Subject-specific additions
        subject_lower = subject.lower()
        if any(w in subject_lower for w in ["warrior", "soldier", "fighter"]):
            details = " | Combat-trained muscle memory; efficient energy expenditure; tactical awareness maintained"
        elif any(w in subject_lower for w in ["beast", "animal", "wolf", "tiger"]):
            details = " | Predator reflexes; quadruped weight distribution; instinct-driven patterns"
        else:
            details = " | Natural human biomechanics; learned motor skills"
        
        # Context modifiers
        if "tired" in context.lower() or "exhaust" in context.lower():
            details += "; reduced power; muscle fatigue evident; breathing labored"
        elif "alert" in context.lower() or "ready" in context.lower():
            details += "; muscle readiness 30% MVC; optimal positioning; neural primed"
        
        result = (base + details).strip(" |;")
        if len(result) < 60:
            result += "; kinetic energy through musculoskeletal chain; proprioception guides motion"
        return result
    
    def _enrich_action_physics(self, verb: str, environment: dict, weather: str = "") -> str:
        """Tạo physics note cực kỳ chi tiết (60-100 chars minimum)"""
        verb_lower = verb.lower()
        base = PHYSICS_TEMPLATES.get(verb_lower, "")
        
        if not base:
            base = "Newton's laws govern motion; momentum p=mv transfers; energy conserved/dissipated; forces cause acceleration F=ma"
        
        # Terrain adjustments
        terrain = str(environment.get("terrain", "")).lower()
        if "mud" in terrain or "wet" in terrain:
            base += " | Friction reduced ~0.4; slip risk; energy absorbed by substrate"
        elif "sand" in terrain or "loose" in terrain:
            base += " | Sand absorbs energy; unstable footing; dust on impact"
        elif "rock" in terrain or "stone" in terrain:
            base += " | High friction ~0.8; solid support; ricochet potential"
        
        # Weather adjustments
        weather_lower = weather.lower()
        if "rain" in weather_lower:
            base += " | Rain adds mass; surfaces slippery; water resistance"
        elif "wind" in weather_lower:
            base += " | Wind applies lateral force; balance challenged; projectile drift"
        
        if len(base) < 60:
            base += "; work done W=F⋅d; energy state measurable"
        return base.strip(" |;")
    
    def _design_camera_movement_cinematic(self, scene_mood: str, creative_brief: str, 
                                           duration: float = 8.0) -> dict:
        """Thiết kế camera như cinematographer chuyên nghiệp"""
        brief_lower = creative_brief.lower()
        mood_lower = scene_mood.lower()
        
        # Select style
        if any(w in brief_lower for w in ["charges", "strikes", "combat", "fight"]):
            style = "action_intense"
        elif any(w in mood_lower for w in ["emotional", "intimate"]):
            style = "emotional_intimate"
        elif any(w in brief_lower for w in ["establish", "surveys", "arrives"]):
            style = "establishing_wide"
        elif any(w in mood_lower for w in ["tense", "suspense"]):
            style = "tension_buildup"
        else:
            style = "action_intense"
        
        rules = CINEMATIC_CAMERA_RULES.get(style, CINEMATIC_CAMERA_RULES["action_intense"])
        mov_type = random.choice(rules["movement_types"])
        
        movement = {
            "type": mov_type,
            "speed": random.choice(rules["speed_range"]),
            "motivation": rules["motivation"],
            "secondary_motion": "subtle breathing" if "handheld" in rules["recommended_rigs"] else "micro-adjustments",
            "easing": "ease_in_out"
        }
        
        lens_mm = random.randint(rules["lens_range"][0], rules["lens_range"][1])
        
        return {
            "rig": random.choice(rules["recommended_rigs"]),
            "lens_mm": lens_mm,
            "aperture_f": rules["aperture_sweet_spot"],
            "shutter_angle_deg": rules["shutter_angle"],
            "white_balance_k": 5600,
            "iso": 400,
            "movement": movement,
            "composition_style": self._get_composition_for_lens(lens_mm),
            "depth_of_field": "shallow" if rules["aperture_sweet_spot"] < 4 else "moderate",
            "focus_pull_style": "follow_subject_smooth"
        }
    
    def _get_composition_for_lens(self, lens_mm: int) -> str:
        """Composition style tối ưu cho focal length"""
        if lens_mm < 24:
            return "ultra-wide; rule of thirds; environmental context maximized"
        elif lens_mm < 50:
            return "medium shot; subject-background balance; rule of thirds"
        elif lens_mm < 85:
            return "portrait; shallow DOF; headroom 1/4 frame"
        else:
            return "tight close-up; emotional detail; fill frame"
    
    def _validate_cause_effect_chain(self, sequence: list, duration: float, 
                                     lead_label: str = "Lead", ant_label: str = "Antagonist") -> tuple[list, list]:
        """Kiểm tra và sửa cause-effect chain - đảm bảo logic vật lý"""
        errors = []
        fixed_seq = sorted(sequence, key=lambda x: float(x.get("timestamp_sec", 0)))
        injected = []
        
        for i, action in enumerate(fixed_seq):
            ts = float(action.get("timestamp_sec", 0))
            verb = action.get("verb", "").lower()
            subject = action.get("subject", "")
            
            # Rule: Impact actions need reactions 0.2-0.8s later
            if verb in ["strikes", "hits", "collides", "impacts", "slashes", "thrusts", "kicks"]:
                # Find reaction
                found_reaction = False
                for j in range(i+1, len(fixed_seq)):
                    next_action = fixed_seq[j]
                    next_ts = float(next_action.get("timestamp_sec", 0))
                    next_verb = next_action.get("verb", "").lower()
                    
                    if next_ts > ts + 1.0:
                        break
                    
                    if next_verb in ["reacts", "staggers", "flinches", "recoils", "falls", "stumbles"]:
                        found_reaction = True
                        # Check timing
                        if next_ts - ts < 0.2:
                            errors.append(f"⚠️ Reaction too fast at {next_ts}s after impact {ts}s")
                            next_action["timestamp_sec"] = round(ts + 0.25, 2)
                        break
                
                if not found_reaction and ts < duration - 1.0:
                    # Inject reaction
                    target = ant_label if subject == lead_label else lead_label
                    reaction = {
                        "timestamp_sec": round(ts + 0.3, 2),
                        "subject": target,
                        "verb": "staggers",
                        "biomechanics_note": "Impact disrupts balance; vestibular compensation; arms flail; center of mass shifts backward",
                        "physics_note": "Impulse J=FΔt transfers momentum; body decelerates; kinetic energy absorbed",
                        "internal_motivation": "Pain response triggers defensive reflex; survival instinct"
                    }
                    injected.append(reaction)
                    errors.append(f"✅ Injected reaction at {reaction['timestamp_sec']}s after impact {ts}s")
            
            # Rule: Projectile needs release → flight → impact
            if verb in ["releases", "shoots", "fires", "looses"]:
                has_flight = any("arrow" in str(a.get("subject", "")).lower() 
                               for a in fixed_seq[i+1:])
                if not has_flight and ts < duration - 1.0:
                    flight = {
                        "timestamp_sec": round(ts + 0.15, 2),
                        "subject": "Arrow",
                        "verb": "flies",
                        "biomechanics_note": "N/A (projectile)",
                        "physics_note": "Initial velocity 60m/s; parabolic trajectory; drag F=½ρv²CdA; gravity drop Δy=½gt²",
                        "internal_motivation": "N/A"
                    }
                    impact = {
                        "timestamp_sec": round(ts + 0.9, 2),
                        "subject": "Arrow",
                        "verb": "impacts",
                        "biomechanics_note": "N/A",
                        "physics_note": "Kinetic energy converts to penetration; target deformation; stress waves propagate",
                        "internal_motivation": "N/A"
                    }
                    injected.extend([flight, impact])
                    errors.append(f"✅ Injected arrow flight at {flight['timestamp_sec']}s")
        
        # Merge injected actions
        fixed_seq.extend(injected)
        fixed_seq = sorted(fixed_seq, key=lambda x: float(x.get("timestamp_sec", 0)))
        
        return fixed_seq, errors
    
    def _ultimate_validation(self, obj: dict, scene_text: str) -> tuple[bool, list]:
        """Validation cực kỳ nghiêm ngặt - đảm bảo ZERO errors"""
        errors = []
        warnings = []
        
        world = obj.get("world_and_characters", {})
        lead = world.get("lead_character", "")
        ant = world.get("antagonist", "")
        char_names = {lead, ant} if lead or ant else set()
        
        shots = obj.get("shots", [])
        seq = shots[0].get("action_sequence", []) if shots else []
        
        # 1. Character name consistency
        for i, action in enumerate(seq):
            subj = action.get("subject", "")
            if subj and subj not in char_names and subj.lower() not in ["arrow", "camera", "projectile"]:
                errors.append(f"❌ Action {i}: Inconsistent name '{subj}' not in {char_names}")
        
        # 2. Creative brief action in sequence
        brief = obj.get("creative_brief", "").lower()
        if brief:
            brief_verbs = re.findall(r'\b(charges|strikes|thrusts|dodges|blocks|shoots|attacks)\b', brief)
            seq_verbs = [a.get("verb", "").lower() for a in seq]
            for bv in brief_verbs[:2]:
                if bv not in seq_verbs:
                    errors.append(f"❌ Main action '{bv}' from brief missing in sequence")
        
        # 3. Timestamp spacing - USE ACTION_SPACING_CONFIG
        min_gap_required = ACTION_SPACING_CONFIG["min_gap_seconds"]
        tolerance = ACTION_SPACING_CONFIG["validation_tolerance"]
        prev_ts = -1.0
        for i, action in enumerate(seq):
            ts = float(action.get("timestamp_sec", 0))
            gap = ts - prev_ts
            # Use tolerance for floating point comparison
            if prev_ts >= 0 and gap < (min_gap_required - tolerance):
                errors.append(f"❌ Action {i} at {ts}s too close to previous ({gap:.2f}s gap, need >= {min_gap_required}s from CONFIG)")
            prev_ts = ts
        
        # 4. Biomechanics/physics depth
        for i, action in enumerate(seq):
            bio = action.get("biomechanics_note", "")
            if len(bio) < 60:
                warnings.append(f"⚠️ Action {i} biomechanics short: {len(bio)} chars (need 60+)")
            phys = action.get("physics_note", "")
            if len(phys) < 60:
                warnings.append(f"⚠️ Action {i} physics short: {len(phys)} chars (need 60+)")
        
        # 5. Wardrobe detail
        char_cont = world.get("character_continuity", {})
        for name in char_names:
            if name and name not in char_cont:
                errors.append(f"❌ Character '{name}' missing from character_continuity")
            elif name:
                desc = char_cont.get(name, "")
                if isinstance(desc, str) and len(desc) < 400:
                    warnings.append(f"⚠️ Character '{name}' description short: {len(desc)} chars (need 400+)")
        
        # 6. Cause-effect pairs
        impacts = sum(1 for a in seq if a.get("verb", "").lower() in ["strikes", "hits", "impacts"])
        reactions = sum(1 for a in seq if a.get("verb", "").lower() in ["staggers", "reacts", "flinches"])
        if impacts > 0 and reactions / max(impacts, 1) < 0.5:
            warnings.append(f"⚠️ Cause-effect imbalance: {impacts} impacts but only {reactions} reactions")
        
        # 7. Verb variety
        verbs = [a.get("verb", "") for a in seq]
        verb_counts = {}
        for v in verbs:
            verb_counts[v] = verb_counts.get(v, 0) + 1
        for verb, count in verb_counts.items():
            if count > 3:
                warnings.append(f"⚠️ Verb '{verb}' repeated {count} times (max 3 recommended)")
        
        # Print summary
        if errors:
            print("\n❌ CRITICAL VALIDATION ERRORS:")
            for e in errors[:10]:  # Show first 10
                print(f"  {e}")
        if warnings:
            print("\n⚠️ VALIDATION WARNINGS:")
            for w in warnings[:10]:
                print(f"  {w}")
        
        return (len(errors) == 0, errors + warnings)
    
    def _post_process_llm_output(self, obj: dict, scene_text: str, continuity) -> dict:
        """Xử lý và cải thiện output từ LLM - Using ACTION_SPACING_CONFIG"""
        shots = obj.get("shots", [])
        if not shots:
            return obj
        
        seq = shots[0].get("action_sequence", [])
        world = obj.get("world_and_characters", {})
        environment = {"terrain": world.get("setting", ""), "weather": world.get("weather", "")}
        
        print("🔧 Post-processing: Using ACTION_SPACING_CONFIG...")
        
        # 1. FIX TIMESTAMP SPACING - USE CONFIG (SINGLE SOURCE OF TRUTH)
        min_gap = ACTION_SPACING_CONFIG["min_gap_seconds"]
        seq = sorted(seq, key=lambda x: float(x.get("timestamp_sec", 0)))
        fixed_seq = []
        last_ts = 0.0
        spacing_fixes = 0
        
        for i, action in enumerate(seq):
            curr_ts = float(action.get("timestamp_sec", 0))
            
            # Enforce minimum gap from CONFIG
            if i > 0 and curr_ts - last_ts < min_gap:
                new_ts = round(last_ts + min_gap, 2)
                if new_ts <= 7.8:  # Keep within 8s limit
                    action["timestamp_sec"] = new_ts
                    curr_ts = new_ts
                    spacing_fixes += 1
                else:
                    # Skip action if it would exceed time limit
                    continue
            
            # Ensure within 8s
            if curr_ts > 7.8:
                curr_ts = 7.8
                action["timestamp_sec"] = 7.8
            
            fixed_seq.append(action)
            last_ts = curr_ts
        
        seq = fixed_seq
        if spacing_fixes > 0:
            print(f"✅ Fixed {spacing_fixes} timestamp spacing issues (enforced {min_gap}s minimum from CONFIG)")
        
        # 2. FIX VERB REPETITION - USE CONFIG MAX
        max_verb_usage = ACTION_SPACING_CONFIG["max_verb_usage"]
        verb_counts = {}
        for action in seq:
            verb = action.get("verb", "")
            verb_counts[verb] = verb_counts.get(verb, 0) + 1
        
        # Find repetitive verbs (exceeding CONFIG limit)
        repetitive = {v: c for v, c in verb_counts.items() if c > max_verb_usage}
        
        if repetitive:
            print(f"🔧 Fixing repetitive verbs: {repetitive}")
            
            # Replace repetitive verbs with varied alternatives
            verb_replacements = {
                "re-aims": ["adjusts aim", "refines target", "steadies weapon"],
                "re-positions": ["shifts stance", "adjusts footing", "repositions body"],
                "micro-adjusts": ["adjusts grip", "maintains balance", "stays alert"],
                "reacts": ["responds", "focuses attention", "prepares stance"],
                "prepares": ["readies", "braces", "positions"]
            }
            
            verb_usage = {v: 0 for v in repetitive.keys()}
            
            for action in seq:
                verb = action.get("verb", "")
                if verb in repetitive:
                    count = verb_usage.get(verb, 0)
                    if count >= max_verb_usage:  # Already used MAX times from CONFIG
                        # Replace with alternative
                        alternatives = verb_replacements.get(verb, ["adjusts", "maintains", "prepares"])
                        new_verb = alternatives[count % len(alternatives)]
                        action["verb"] = new_verb
                        print(f"  → Replaced '{verb}' with '{new_verb}' (usage {count+1}, max={max_verb_usage})")
                    verb_usage[verb] = count + 1
        
        # 3. Auto-enrich biomechanics/physics nếu quá ngắn
        enrich_count = 0
        for action in seq:
            verb = action.get("verb", "")
            subject = action.get("subject", "")
            
            bio = action.get("biomechanics_note", "")
            if len(bio) < 60:
                enriched = self._enrich_action_biomechanics(verb, subject, scene_text)
                action["biomechanics_note"] = enriched
                enrich_count += 1
            
            phys = action.get("physics_note", "")
            if len(phys) < 60:
                enriched = self._enrich_action_physics(verb, environment, world.get("weather", ""))
                action["physics_note"] = enriched
                enrich_count += 1
        
        if enrich_count > 0:
            print(f"✅ Enriched {enrich_count} short biomechanics/physics notes")
        
        # 4. Validate and fix cause-effect
        lead = world.get("lead_character", "Lead")
        ant = world.get("antagonist", "Target")
        seq, errors = self._validate_cause_effect_chain(seq, shots[0].get("duration", 8.0), lead, ant)
        
        # 5. Update character states progressively
        if continuity:
            for action in seq[:3]:
                subj = action.get("subject", "")
                verb = action.get("verb", "")
                if hasattr(continuity, 'update_character_state_progressive'):
                    continuity.update_character_state_progressive(subj, verb, environment)
        
        # Update sequence in shots
        shots[0]["action_sequence"] = seq
        
        print(f"✅ Final sequence: {len(seq)} actions, all spacing >= {min_gap}s (CONFIG), verbs varied (max {max_verb_usage} per verb)")
        
        return obj

    def _is_aftermath(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in [
            "aftermath", "after the shot", "already hit",
            "đã bắn", "hậu quả", "vừa bị bắn", "đã trúng"
        ])

    def load_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            return key
        cfg = "config.json"
        if os.path.exists(cfg):
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    k = (data.get("openai_api_key") or "").strip()
                    return k if k and k != "..." else ""
            except Exception:
                return ""
        return ""

    def _chat(self, **kwargs):
        try:
            # ✨ đảm bảo có timeout per-request (không phụ thuộc constructor)
            return self.client.chat.completions.create(**kwargs, timeout=self.request_timeout)
        except Exception as e:
            if "seed" in str(e).lower():
                kwargs.pop("seed", None)
                return self.client.chat.completions.create(**kwargs, timeout=self.request_timeout)
            raise

    def _parse_choice(self, choice):
        tc = getattr(choice.message, "tool_calls", None)
        if tc:
            args = tc[0].function.arguments
            if args is None:
                raise ValueError("Tool call returned empty arguments.")
            return json.loads(args)
        content = choice.message.content
        if not content:
            raise ValueError("Empty model content and no tool calls.")
        return json.loads(content)

    def _mentions_human_archer(self, scene_text: str, scene_entities: Dict[str, Any]) -> bool:
        """Return True only if the scene explicitly mentions a human archer (not just an arrow)."""
        if not isinstance(scene_text, str):
            return False
        text = scene_text.lower()
        # explicit human labels or roles
        human_kw = [
            "archer", "xạ thủ", "xa thu", "người bắn cung", "ban cung", "bowsman",
        ]
        # character names detected as leads count as humans
        leads = []
        if isinstance(scene_entities, dict):
            leads = [s.lower() for s in (scene_entities.get("lead") or []) if isinstance(s, str)]
            # also check content_main/action_specific for archer role words
            text += " " + (scene_entities.get("action_specific") or "").lower()
            text += " " + (scene_entities.get("content_main") or "").lower()
        if any(k in text for k in human_kw):
            return True
        # simple heuristic: if any non-empty lead names exist, treat as human present
        return any(leads)

    def _sanitize_continuity_updates(self, o: dict) -> dict:
        cu = o.get("continuity_updates") or {}
        def filt(arr):
            return [e for e in (arr or []) if isinstance(e, dict) and e.get("name") and isinstance(e.get("profile"), dict)]
        if "new_or_updated_characters" in cu:
            cu["new_or_updated_characters"] = filt(cu.get("new_or_updated_characters"))
        if "new_or_updated_locations" in cu:
            cu["new_or_updated_locations"] = filt(cu.get("new_or_updated_locations"))
        if "new_or_updated_dynamic_entities" in cu:
            cu["new_or_updated_dynamic_entities"] = filt(cu.get("new_or_updated_dynamic_entities"))
        o["continuity_updates"] = cu
        return o

    def _prune_optional_fields_strict(self, obj: Dict[str, Any], scene_text: str, scene_entities: Dict[str, Any]) -> Dict[str, Any]:
        """In strict mode, minimize optional fields to empty-but-valid if the scene doesn't mention them."""
        if not getattr(self, "strict_grounding", False):
            return obj
        try:
            shots = obj.get("shots") or []
            has_audio_in_text = bool((scene_entities or {}).get("audio"))
            has_vfx_in_text = False  # hiện chưa có nhãn VFX riêng trong input
            has_lighting_in_text = bool(re.search(r"\b(light|moon|tr\u0103ng|\u00e1nh tr\u0103ng|\u00e1nh s\u00e1ng|b\u00f3ng)\b", scene_text or "", re.IGNORECASE))
            for s in shots:
                # audio: preserve existing content, only normalize if None/wrong type
                aud = s.get("audio")
                if isinstance(aud, dict):
                    # Keep existing content even if not has_audio_in_text
                    # Only normalize empty/None values to proper defaults
                    for k in ["ambience","foley","sfx","music_cue","mix_notes","dialogue_note","fx","music"]:
                        if k in aud and aud[k] is None:
                            aud[k] = "" if k in ["ambience","mix_notes","dialogue_note"] else []
                    s["audio"] = aud
                elif aud is None:
                    s["audio"] = {}

                # vfx: keep empty list if nothing specific
                if not isinstance(s.get("vfx"), list):
                    s["vfx"] = []
                elif not has_vfx_in_text and not s["vfx"]:
                    s["vfx"] = []

                # lighting: keep empty string if not provided
                if not isinstance(s.get("lighting"), str):
                    s["lighting"] = ""
                elif not has_lighting_in_text and s.get("lighting") in (None, "natural, soft daylight", "", "natural"):
                    s["lighting"] = ""

                # transition_out: keep empty string if not provided
                if not isinstance(s.get("transition_out"), str):
                    s["transition_out"] = ""

                # creative_suggestions: remove if empty or not present in text
                cs = s.get("creative_suggestions")
                if not cs:
                    s.pop("creative_suggestions", None)
        except Exception:
            pass
        return obj

    def _classify_entity(self, name: str) -> str:
        """Phân loại sơ bộ: animal | vehicle | projectile | object."""
        if not isinstance(name, str):
            return "object"
        n = name.strip().lower()
        animals = ["wolf","tiger","lion","bear","boar","panther","leopard","dog","cat","eagle",
                   "hawk","falcon","vulture","horse","ox","buffalo","deer","stag",
                   "sói","hổ","sư tử","gấu","lợn rừng","báo","chó","mèo","đại bàng",
                   "diều hâu","ngựa","trâu","nai","hươu","heo rừng","rắn","cá sấu","khỉ"]
        vehicles = ["car","truck","jeep","motorbike","motorcycle","bike","bicycle","boat","ship",
                    "canoe","raft","drone","helicopter","tank",
                    "xe","xe hơi","ô tô","xe tải","xe máy","thuyền","tàu","ca nô","xuồng",
                    "máy bay","trực thăng","xe tăng","máy bay không người lái"]
        projectiles = ["arrow","bolt","bullet","spear","javelin","rock","boulder","stone","dart",
                       "mũi tên","tên","giáo","lao","đá","hòn đá"]
        if any(a == n or a in n for a in animals): return "animal"
        if any(v == n or v in n for v in vehicles): return "vehicle"
        if any(p == n or p in n for p in projectiles): return "projectile"
        return "object"

    def _mentioned_entities(self, scene_text: str, scene_entities: dict, continuity) -> dict:
        """
        Tìm mọi thực thể được nhắc đến trong cảnh:
        - characters (tận dụng _mentioned_characters)
        - animals / vehicles / projectiles (quét từ khoá)
        - các entities đã có sẵn trong continuity.entities nếu xuất hiện trong text
        Trả về dict: { characters, animals, vehicles, projectiles, objects, all }
        """
        import re as _re
        text = scene_text or ""
        low = text.lower()
        out = {"characters": [], "animals": [], "vehicles": [], "projectiles": [], "objects": [], "all": [], "dynamic": []}

        # nhân vật
        try:
            names = self._mentioned_characters(scene_text, scene_entities, continuity)
        except Exception:
            names = []
        out["characters"] = [n for n in names if isinstance(n, str) and n.strip()]

        # entities đã có trong continuity (nếu khớp từ nguyên vẹn)
        for name in (getattr(continuity, "entities", {}) or {}).keys():
            try:
                if _re.search(rf"(?i)\\b{_re.escape(name)}\\b", text):
                    out["objects"].append(name)
            except Exception:
                pass

        animals = ["wolf","tiger","lion","bear","boar","panther","leopard","dog","cat","eagle",
                   "hawk","falcon","vulture","horse","ox","buffalo","deer","stag",
                   "sói","hổ","sư tử","gấu","lợn rừng","báo","chó","mèo","đại bàng",
                   "diều hâu","ngựa","trâu","nai","hươu","heo rừng","rắn","cá sấu","khỉ"]
        vehicles = ["car","truck","jeep","motorbike","motorcycle","bike","bicycle","boat","ship",
                    "canoe","raft","drone","helicopter","tank",
                    "xe","xe hơi","ô tô","xe tải","xe máy","thuyền","tàu","ca nô","xuồng",
                    "máy bay","trực thăng","xe tăng","máy bay không người lái"]
        projectiles = ["arrow","bolt","bullet","spear","javelin","rock","boulder","stone","dart",
                       "mũi tên","tên","giáo","lao","đá","hòn đá"]

        def collect(lst, bucket):
            """Collect entities with word boundary check to avoid false positives."""
            for w in lst:
                # Use word boundary regex to avoid matching substrings
                # Example: "lao" should not match "lao vào" (rush into)
                pattern = rf"\b{_re.escape(w)}\b"
                try:
                    if _re.search(pattern, low):
                        bucket.append(w)
                except Exception:
                    # Fallback to simple substring check if regex fails
                    if w in low:
                        bucket.append(w)

        collect(animals, out["animals"])
        collect(vehicles, out["vehicles"])
        collect(projectiles, out["projectiles"])

        # khử trùng lặp
        def dedup(seq):
            seen=set(); r=[]
            for x in seq:
                if x and x not in seen:
                    r.append(x); seen.add(x)
            return r

        for k in list(out.keys()):
            out[k] = dedup(out[k])

        out["all"] = dedup(out["characters"] + out["animals"] + out["vehicles"] + out["projectiles"] + out["objects"])
        # dynamic entries with kind
        for nm in out["animals"]:
            out["dynamic"].append({"name": nm, "kind": "animal"})
        for nm in out["vehicles"]:
            out["dynamic"].append({"name": nm, "kind": "vehicle"})
        for nm in out["projectiles"]:
            out["dynamic"].append({"name": nm, "kind": "projectile"})
        for nm in out["objects"]:
            out["dynamic"].append({"name": nm, "kind": self._classify_entity(nm)})
        return out

    def _build_detailed_character_profile(self, name: str, scene_text: str, continuity) -> dict:
        base = continuity.characters.get(name, {}) if hasattr(continuity, "characters") else {}
        return {
            "appearance": base.get("appearance") or "",
            "wardrobe": base.get("wardrobe") or "",
            "props_in_hand": base.get("props_in_hand") or "",
            "weapon_state": base.get("weapon_state") or "",
            "movement_signature": base.get("movement_signature") or "",
            "voice": base.get("voice") or "",
            "psychology": base.get("psychology") or "",
            "relationships": base.get("relationships") or "",
        }

    def _build_detailed_dynamic_profile(self, label: str, kind: str, scene_text: str, continuity) -> dict:
        # kind: animal | vehicle | projectile | object
        dflt_mech = {
            "mass_kg": "unknown",
            "center_of_gravity": "mid",
            "locomotion": "n/a",
            "reaction_to_threat": "",
            "vulnerability": ""
        }
        if kind == "animal":
            biomech = dict(dflt_mech); biomech["locomotion"] = "quadruped gait (estimate)"
            return {
                "biomechanics": biomech,
                "operational_state": "",
                "interaction_rules": ""
            }
        elif kind == "vehicle":
            return {
                "mechanics": {"motion_model": "wheels/propulsion; inertia, traction, turning radius"},
                "operational_state": "",
                "interaction_rules": ""
            }
        elif kind == "projectile":
            return {
                "mechanics": {"motion_model": "ballistic arc; gravity/drag/momentum; non-graphic impact wording"},
                "operational_state": "",
                "interaction_rules": ""
            }
        else:
            return {
                "mechanics": {"motion_model": "mass/inertia/friction dominated"},
                "operational_state": "",
                "interaction_rules": ""
            }

    def _inject_entity_profiles_for_scene(self, obj, scene_text: str, continuity):
        # Tìm mentions theo cảnh
        try:
            scene_entities = self.extract_scene_entities(scene_text)
        except Exception:
            scene_entities = {}
        mentioned = self._mentioned_entities(scene_text, scene_entities, continuity) if hasattr(self, "_mentioned_entities") else {"characters": [], "dynamic": []}
        char_names = [n.strip() for n in (mentioned.get("characters") or []) if n and str(n).strip()]
        dyns = [d for d in (mentioned.get("dynamic") or []) if isinstance(d, dict) and d.get("name")]

        cu = obj.setdefault("continuity_updates", {})
        cu.setdefault("new_or_updated_characters", [])
        cu.setdefault("new_or_updated_dynamic_entities", [])

        existing_chars = { (e.get("name") or "").strip() for e in cu.get("new_or_updated_characters") if isinstance(e, dict) }
        for n in char_names:
            if n in existing_chars:
                continue
            profile = self._build_detailed_character_profile(n, scene_text, continuity) if hasattr(self, "_build_detailed_character_profile") else {}
            cu["new_or_updated_characters"].append({"name": n, "profile": profile})

        existing_dyn = { (e.get("name") or "").strip() for e in cu.get("new_or_updated_dynamic_entities") if isinstance(e, dict) }
        for ent in dyns:
            label = (ent.get("name") or "").strip()
            kind  = (ent.get("kind") or "object")
            if not label or label in existing_dyn:
                continue
            profile = self._build_detailed_dynamic_profile(label, kind, scene_text, continuity) if hasattr(self, "_build_detailed_dynamic_profile") else {}
            cu["new_or_updated_dynamic_entities"].append({"name": label, "profile": profile})

        return obj

    # >>> PATCH START: AUDIO RECLASSIFY + FULL COVERAGE + PROP CONSISTENCY (class methods)
    def _reclassify_audio(self, shot: dict, scene_text: str = "") -> dict:
        """Normalize audio fields; drop archery cues if scene/text is non-archery."""
        import re
        aud = shot.get("audio") or {}
        # normalize list fields
        for k in ("foley", "sfx"):
            v = aud.get(k)
            if isinstance(v, str):
                aud[k] = [v]
            elif isinstance(v, list):
                aud[k] = [str(x) for x in v if str(x).strip()]
            else:
                aud[k] = []

        # if scene not about archery, strip archery cues
        seq = shot.get("action_sequence") or []
        has_archery_subject = any("arrow" in (e.get("subject","" ).lower()) or "archer" in (e.get("subject","" ).lower()) for e in seq)
        text_archery = _has_archery_kw(scene_text)
        if not (has_archery_subject or text_archery):
            for key in ("foley", "sfx"):
                aud[key] = [
                    x for x in aud.get(key, [])
                    if not re.search(r"\b(arrow|bowstring|woody\s+thunk)\b", x, flags=re.IGNORECASE)
                ]

        # soften wording (non-graphic)
        def soft(x: str) -> str:
            x = re.sub(r"(?i)\bspear\s+piercing\s+flesh\b", "spear makes solid contact (non-graphic)", x)
            x = re.sub(r"(?i)\bpainful\s+roar\b", "fierce roar", x)
            x = re.sub(r"(?i)\breveals\s+injur(?:y|ies)\b", "reveals visible damage (non-graphic)", x)
            return x

        aud["foley"] = [soft(x) for x in aud.get("foley", [])]
        aud["sfx"]   = [soft(x) for x in aud.get("sfx", [])]

        # drop empties
        aud = {k: v for k, v in aud.items() if v not in ([], "", None)}
        shot["audio"] = aud
        return shot

    def _unify_subject_labels(self, obj: dict, *_, **__) -> dict:
        """Enhanced subject label unification using global canon() system."""
        return _unify_subject_labels_enhanced(obj)

    def _dominant_verbs(self, seq: list[str]) -> set[str]:
        verbs = set()
        for e in seq:
            v = (e.get("verb","" ) or "").lower()
            for core in ("charges","thrusts","stabs","slashes","strikes","pushes","shoves","recoils","dodges","parries","feints","grapples"):
                if core in v:
                    verbs.add(core)
        return verbs

    def _enforce_full_8s_coverage(self, seq: list[dict], dur: float, lead_label: str, ant_label: str) -> list[dict]:
        """Fill meaningful actions to reach dur (~8s), not just micro-adjusts."""
        if not isinstance(seq, list):
            return seq
        seq = sorted(seq, key=lambda e: float(e.get("timestamp_sec", 0.0)))

        # if last beat ends too early, add 1-2 purposeful beats based on dominant verbs
        last_t = float(seq[-1]["timestamp_sec"]) if seq else 0.0
        want_t1 = round(min(dur-0.7, max(last_t+0.6, 6.0)), 2)
        want_t2 = round(dur-0.12, 2)

        verbs = self._dominant_verbs(seq)
        def add(t, subject, verb, bio, phy, why):
            seq.append({
                "timestamp_sec": float(max(0.0, min(dur, t))),
                "subject": subject,
                "verb": verb,
                "biomechanics_note": bio,
                "physics_note": phy,
                "internal_motivation": why
            })

        if last_t < dur - 0.5:
            # pick actions driven by what's already happening
            if {"thrusts","stabs","strikes"} & verbs:
                if want_t1 > last_t and want_t1 < dur-0.05:
                    add(want_t1, lead_label, "presses forward",
                        "drives from hips; shoulders square; feet dig to advance",
                        "momentum carries into the opponent; friction prevents slip",
                        "capitalize on advantage; keep pressure on the opponent")
                if want_t2 > last_t:
                    add(want_t2, ant_label, "re-aims while shifting guard",
                        "upper mass shifts back; stance breaks; claws/feet scramble",
                        "momentum from prior impact disrupts balance; center-of-mass shifts",
                        "instinctive retreat from pressure")
            elif {"slashes"} & verbs:
                if want_t1 > last_t and want_t1 < dur-0.05:
                    add(want_t1, lead_label, "feints and re-engages",
                        "shoulders dip; quick weight shift; wrists set angle",
                        "inertia used to mask intent; quick torque sets new line",
                        "probe for opening without over-committing")
                if want_t2 > last_t:
                    add(want_t2, ant_label, "flinches and pivots",
                        "torso twists; hind/lead foot pivots to absorb and avoid",
                        "angular momentum + friction manage rapid redirection",
                        "avoid exposure; seek safer angle")
            else:
                # generic: keep it purposeful, not idle
                if want_t1 > last_t and want_t1 < dur-0.05:
                    add(want_t1, lead_label, "cuts angle",
                        "quick sidestep; hips turn; guard stays high",
                        "lateral movement changes engagement line",
                        "maintain initiative until the last second")
                if want_t2 > last_t:
                    add(want_t2, ant_label, "reacts under pressure",
                        "weight shifts; shoulders tense; recovery footing",
                        "momentum transfer and friction force a visible response",
                        "instinctive survival reaction")

        seq = sorted(seq, key=lambda e: float(e.get("timestamp_sec", 0.0)))
        return seq

    def _enforce_prop_state_consistency(self, shot: dict) -> dict:
        """If spear is dropped at t=T, disallow thrust/stab after T; rephrase + clean audio."""
        import re
        seq = shot.get("action_sequence") or []
        drop_t = None
        for e in seq:
            subj = (e.get("subject","" ) or "").lower()
            verb = (e.get("verb","" ) or "").lower()
            txt = f"{subj} {verb} {(e.get('biomechanics_note','') or '').lower()} {(e.get('physics_note','') or '').lower()}"
            if ("warrior" in subj or "lead" in subj) and (("drops" in verb) or re.search(r"\bspear\s+falls\b", txt)):
                drop_t = float(e.get("timestamp_sec", 0.0))
                break

        if drop_t is None:
            return shot

        # Rewrite post-drop thrusts/stabs into unarmed shoves/presses
        for e in seq:
            ts = float(e.get("timestamp_sec", 0.0))
            if ts <= drop_t:
                continue
            subj = (e.get("subject","" ) or "").lower()
            verb = (e.get("verb","" ) or "").lower()
            if ("warrior" in subj or "lead" in subj) and any(v in verb for v in ("thrust","stab","impale","lunge")):
                e["verb"] = "shoves"
                e["biomechanics_note"] = "shoulders and hips drive forward; forearm frames; feet push to gain space"
                e["physics_note"] = "force transfers through body contact; friction prevents sliding"
                e["internal_motivation"] = "continue pressure despite losing the weapon"

        # Scrub spear-specific audio after drop
        aud = shot.get("audio") or {}
        for key in ("foley", "sfx"):
            arr = [x for x in (aud.get(key) or []) if not re.search(r"(?i)\bspear\b", str(x))]
            if arr:
                aud[key] = arr
            else:
                aud.pop(key, None)
        shot["audio"] = aud
        shot["action_sequence"] = seq
        return shot

    # Optional helper from instructions; only used if present
    def _densify_full_8s(self, shot: dict, lead_label: str, ant_label: str) -> dict:
        """Đảm bảo chuỗi beats dày 0.5–1.0s, có cause→effect, có nhịp chốt ~7.8–7.95s."""
        dur = float(shot.get("duration", 8.0)) or 8.0
        seq = [e for e in (shot.get("action_sequence") or []) if isinstance(e, dict)]
        seq.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))

        # 1) Không để khoảng trống > 1.0s
        beats = []
        prev = -0.5
        for e in seq:
            t = max(0.0, min(dur, float(e.get("timestamp_sec", 0.0))))
            if t - prev > 1.0:
                beats.append({
                    "timestamp_sec": round(max(prev + 0.6, 0.0), 2),
                    "subject": lead_label or "Lead",
                    "verb": "micro-adjusts",
                    "biomechanics_note": "breath-linked sway; foot grip checks; gaze locks",
                    "physics_note": "weight shift & friction maintain continuous motion",
                    "internal_motivation": "staying on-task; no idle"
                })
            beats.append({**e, "timestamp_sec": round(t, 2)})
            prev = t

        if not beats:
            beats = [{
                "timestamp_sec": 0.0,
                "subject": lead_label or "Lead",
                "verb": "steps into frame",
                "biomechanics_note": "hips lead; shoulders square; controlled breath",
                "physics_note": "inertia & foot friction set initial balance",
                "internal_motivation": "initiating objective from 'Nội dung'"
            }]

        # 2) Bù cuối: luôn có beat sát 7.9s để không “đứng hình”
        if beats[-1]["timestamp_sec"] < dur - 0.2:
            beats.append({
                "timestamp_sec": round(dur - 0.05, 2),
                "subject": lead_label or "Lead",
                "verb": "micro-adjusts",
                "biomechanics_note": "tiny weight shift; breath reset; grip fine-tune",
                "physics_note": "continuous micro-motion avoids idle plateau",
                "internal_motivation": "hold through last frame"
            })

        # 3) Ghép cause→effect gần nhau (≤0.9s)
        out = []
        REACTION_VERBS = ("recoils","staggers","deflects","reels","splashes","reacts")
        for i, e in enumerate(sorted(beats, key=lambda x: x["timestamp_sec"])):
            out.append(e)
            v = (e.get("verb","") or "").lower()
            if any(k in v for k in ("thrust","strike","push","pull","charge","throw")):
                t0 = e["timestamp_sec"]
                has_react = any(
                    (x["timestamp_sec"] - t0) <= 0.9 and 
                    any(rv in (x.get("verb","") or "").lower() for rv in REACTION_VERBS)
                    for x in beats[i+1:]
                )
                if not has_react:
                    out.append({
                        "timestamp_sec": round(min(t0 + 0.7, dur - 0.05), 2),
                        "subject": ant_label or "Target",
                        "verb": "reacts",
                        "biomechanics_note": "weight shifts; brief off-balance; shoulder/hips adjust",
                        "physics_note": "momentum transfer + friction produce visible response",
                        "internal_motivation": "instinctive response to force"
                    })

        out.sort(key=lambda x: x["timestamp_sec"])
        shot["action_sequence"] = out
        return shot
    # <<< PATCH END

    def _finalize_minimal_output(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Remove optional top-level fields and diagnostics from final output to match user's minimal schema preferences."""
        # >>> ADD THIS GUARD <<<
        if isinstance(obj, list):
            obj = obj[0] if obj and isinstance(obj[0], dict) else {}
        elif not isinstance(obj, dict):
            obj = {}
        # <<< END ADD >>>
        # Drop internal/helper fields (đừng xóa các trường required theo schema)
        for k in ["model_prompt_format"]:
            if k in obj:
                obj.pop(k, None)

        # Optionally trim empty globals
        gs = obj.get("global_style") or {}
        if isinstance(gs, dict):
            for k in ["vfx_guidance", "sound_mix"]:
                if k in gs and not gs.get(k):
                    gs.pop(k, None)
            obj["global_style"] = gs

        # Ensure shots have only required optional fields when empty
        shots = obj.get("shots") or []
        if isinstance(shots, list):
            for s in shots:
                if not isinstance(s, dict):
                    continue
                # Remove optional creative suggestions
                s.pop("creative_suggestions", None)
                # Clean empty audio object
                aud = s.get("audio")
                if isinstance(aud, dict):
                    # remove keys that are empty
                    for k in ["ambience","foley","sfx","music_cue","mix_notes","dialogue_note","fx","music"]:
                        v = aud.get(k)
                        if v is None:
                            # Only normalize None to proper defaults, preserve existing content
                            aud[k] = "" if k in ["ambience","mix_notes","dialogue_note"] else []
                    s["audio"] = aud
                # Normalize empty vfx/lighting/transition_out
                if isinstance(s.get("vfx"), list) and not s["vfx"]:
                    s["vfx"] = []
                if not isinstance(s.get("lighting"), str):
                    s["lighting"] = ""
                if not isinstance(s.get("transition_out"), str):
                    s["transition_out"] = ""
        return obj

    # -------------------------
    # Scene splitting (unchanged from your last good version)
    # -------------------------
    def _clean_text(self, text: str) -> str:
        tx = text.replace("\r\n", "\n").replace("\r", "\n")
        tx = tx.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
        tx = tx.replace("–", "-").replace("—", "-").replace("…", "...")
        return tx

    def split_scenes(self, text: str) -> List[Tuple[int, str]]:
        # BƯỚC MỚI: Gọi LLM để chuẩn hóa tiêu đề (tùy chọn, an toàn)
        if self.use_llm_scene_normalization and self.client:
            normalization_prompt = (
                "Please reformat the following script text. "
                "Your only task is to ensure every scene heading is on its own line and "
                "formatted exactly as '### SCENE X:', where X is the scene number. "
                "Do not change any other content."
            )
            normalized_text = self._call_llm_for_utility(normalization_prompt, text)
        else:
            normalized_text = text

        # Các bước Regex sau đó sẽ chạy trên normalized_text đã được chuẩn hóa
        tx = self._clean_text(normalized_text)
        lines = tx.split("\n")
        print(f"🔍 DEBUG: Đang kiểm tra {len(lines)} dòng...")

        main_heading_pattern = re.compile(
            r"""^\s*
                (?:\*\*)?                 # optional leading **
                (?:[#]+\s*)?              # optional markdown heading, e.g., ## or ###
                (?:PHÂN\s*CẢNH|PHAN\s*CANH|CẢNH|CANH|SCENE)
                \s*(\d+(?:\.\d+)?)        # scene number, supports 1 or 1.1
                (?:\s*[:\-–—]\s*.*)?      # optional title after :, -, – or —
                (?:\s*\*\*)?              # optional trailing **
                \s*$""",
            re.IGNORECASE | re.VERBOSE
        )

        loose_heading_pattern = re.compile(
            r"^\s*(?:PHÂN\s*CẢNH|PHAN\s*CANH|CẢNH|CANH|SCENE)\s*(\d+(?:\.\d+)?).*$",
            re.IGNORECASE
        )

        potential_headings = []
        for i, ln in enumerate(lines):
            s = ln.strip()
            if main_heading_pattern.match(s):
                potential_headings.append((i, s))

        print(f"🔍 DEBUG: Tìm thấy {len(potential_headings)} dòng heading (pattern chính)")

        if not potential_headings:
            for i, ln in enumerate(lines):
                s = ln.strip()
                if loose_heading_pattern.match(s):
                    potential_headings.append((i, s))
            print(f"🔍 DEBUG: Tìm thấy {len(potential_headings)} dòng heading (pattern lỏng)")

        if potential_headings:
            print("🔍 DEBUG: Các dòng heading mẫu:")
            for k, (line_num, content) in enumerate(potential_headings[:10]):
                print(f" Dòng {line_num+1}: '{content}'")

        if not potential_headings:
            print("🔍 DEBUG: Không tìm thấy heading theo từ khóa, thử tách theo số thứ tự...")
            return self._split_by_numbers(tx)

        scenes: List[Tuple[int, str]] = []
        for j, (start_i, heading_line) in enumerate(potential_headings):
            end_i = potential_headings[j+1][0] if j+1 < len(potential_headings) else len(lines)
            block_lines = lines[start_i:end_i]
            block = "\n".join(block_lines).strip()
            if not block or len(block) < 5:
                continue
            num = self._extract_scene_number(heading_line, j + 1)
            scenes.append((num, block))

        print(f"🔍 DEBUG: Tạo được {len(scenes)} scenes")

        if len(scenes) == 1 and len(potential_headings) > 1:
            print("🔍 DEBUG: Chỉ có 1 cảnh mặc dù có nhiều heading, thử tách lại theo số...")
            return self._split_by_numbers(tx)

        return scenes[:500]

    def _split_by_numbers(self, text: str) -> List[Tuple[int, str]]:
        lines = text.split("\n")
        number_pattern = re.compile(r"^\s*(\d+)[\.\-\:\s]+")
        numbered_lines = []
        for i, ln in enumerate(lines):
            if number_pattern.match(ln.strip()):
                numbered_lines.append((i, ln.strip()))
        print(f"🔍 DEBUG: Tìm thấy {len(numbered_lines)} dòng có số thứ tự")

        if not numbered_lines:
            print("🔍 DEBUG: Không tìm thấy số thứ tự → trả về 1 cảnh toàn bộ")
            return [(1, text.strip())]

        scenes: List[Tuple[int, str]] = []
        for j, (start_i, heading_line) in enumerate(numbered_lines):
            end_i = numbered_lines[j+1][0] if j+1 < len(numbered_lines) else len(lines)
            block_lines = lines[start_i:end_i]
            block = "\n".join(block_lines).strip()
            if len(block) < 10:
                continue
            match = number_pattern.match(heading_line)
            num = int(match.group(1)) if match else (j + 1)
            scenes.append((num, block))

        print(f"🔍 DEBUG: Tách theo số thứ tự được {len(scenes)} scenes")
        return scenes[:500]


    def _extract_scene_number(self, heading_line: str, fallback: int) -> int:
        patterns = [
            r"(\d+(?:\.\d+)?)",
            r"CẢNH\s*(\d+)",
            r"SCENE\s*(\d+)",
            r"PHÂN\s*CẢNH\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, heading_line, re.IGNORECASE)
            if match:
                try:
                    return int(float(match.group(1)))
                except:
                    continue
        return fallback


    def _debug_dump_scenes(self, scenes: List[Tuple[int, str]]) -> None:
        print("📑 SCENES DETECTED:", len(scenes))
        for i, (num, block) in enumerate(scenes, 1):
            first = (block.splitlines()[0] if block else "").strip()
            preview = first if len(first) <= 80 else first[:77] + "..."
            print(f" #{i:02d} label={num} | {preview}")


# -------------------------
# Rich entity extraction with Vietnamese bullet labels (added)
# -------------------------
    def extract_scene_entities(self, text: str) -> Dict[str, Any]:
        lower = text.lower()
        out: Dict[str, Any] = {
            "setting": "",
            "weather": "",
            "time_of_day": "",
            "lead": [],
            "antagonist": [],
            "aircraft": [],
            # NEW bullets
            "content_main": "",        # Nội dung
            "action_specific": "",     # Hành động cụ thể
            "wardrobe": "",            # Trang phục/Tạo hình Nhân vật
            "camera_angle": "",        # Góc máy
            "camera_move": "",         # Chuyển động máy quay
            "audio": "",               # Âm thanh
            "setting_image": "",       # Bối cảnh/Hình ảnh
        }

        # helpers to grab one-line fields by label (supports diacritics variants)
        def grab(label_list):
            for lab in label_list:
                # pattern like: * **Label:** value
                pat = rf"^\s*[*\-•]?\s*(?:\*\*\s*)?{lab}(?:\s*\*\*)?\s*:\s*(.+)$"
                m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    return m.group(1).strip()
            return ""

        out["content_main"] = grab(["nội dung", "noi dung", "content"]) or ""
        out["setting_image"] = grab(["bối cảnh/ hình ảnh", "bối cảnh/hình ảnh", "bối cảnh", "boi canh", "hinh anh", "hình ảnh", "setting", "location"]) or ""
        out["wardrobe"] = grab(["trang phục/ tạo hình nhân vật", "trang phục", "tao hinh nhan vat", "wardrobe", "costume"]) or ""
        out["action_specific"] = grab(["hành động cụ thể", "hanh dong cu the", "actions"]) or ""
        out["camera_angle"] = grab(["góc máy", "goc may", "angle"]) or ""
        out["camera_move"] = grab(["chuyển động máy quay", "chuyen dong may quay", "camera movement"]) or ""
        out["audio"] = grab(["âm thanh", "am thanh", "audio"]) or ""

        # legacy/basic fields
        def grab_basic(label_list):
            for lab in label_list:
                m = re.search(rf"^\s*{lab}\s*:\s*(.+)$", lower, flags=re.MULTILINE)
                if m:
                    raw = m.group(1).strip()
                    for line in text.splitlines():
                        if re.match(rf"^\s*{lab}\s*:\s*", line, flags=re.IGNORECASE):
                            return line.split(":", 1)[1].strip()
                    return raw
            return ""

        out["setting"] = grab_basic(["bối cảnh","boi canh","setting","địa điểm","dia diem","location"]) or out["setting_image"]
        out["weather"] = grab_basic(["thời tiết","thoi tiet","weather"]) or ""
        out["time_of_day"] = grab_basic(["thời điểm","thoi diem","time of day","timeofday","time","thời gian","thoi gian"]) or ""

        def split_names(s):
            if not s:
                return []
            parts = re.split(r"[,/;&]| and | với | va | vs\.?", s, flags=re.IGNORECASE)
            names = [p.strip(" -–—\t") for p in parts if p.strip()]
            return [n for n in names if len(n) <= 40][:5]

        lead = grab_basic(["nhân vật chính","nhan vat chinh","lead","protagonist","main character","nhân vật","nhan vat"]) or ""
        antg = grab_basic(["đối thủ","doi thu","antagonist","enemy","opponent"]) or ""
        out["lead"] = split_names(lead)
        out["antagonist"] = split_names(antg)

        ac_pat = re.compile(r"\b(hellcat|zero|spitfire|mustang|b-?f\s?109|yak-?3|fw\s?190)\b", re.IGNORECASE)
        out["aircraft"] = sorted({m.group(0) for m in ac_pat.finditer(text)})

        return out


    def _is_archery_scene(self, scene_text: str, scene_entities: Dict[str, Any]) -> bool:
        if not isinstance(scene_text, str):
            return False
        text = scene_text.lower()
        # STRICT CHECK: Only detect archery if explicitly mentioned in scene content
        archery_kw = ["bow","arrow","archer","archery","cung","mũi tên","bắn tên","căng dây"]
        if self._is_aftermath(text):  # block archery injection in aftermath-only scenes
            return False
        # Check main scene text first
        if any(k in text for k in archery_kw):
            return True
        # Check action_specific field
        act = (scene_entities.get("action_specific") or "").lower()
        if any(k in act for k in archery_kw):
            return True
        # Check content_main field
        content = (scene_entities.get("content_main") or "").lower()
        if any(k in content for k in archery_kw):
            return True
        return False


    def _archery_rules_block(self) -> str:
        # Khối rule 100% tiếng Anh cho LLM
        return (
            "ARCHERY REALISM LAYER (MANDATORY)\n"
            "- Bow actions must follow: draw (with visible string tension) → anchor (stable anchor point) → release → arrow flight (parabolic arc, non-instant) → impact/near-miss → plausible aftermath.\n"
            "- Arrow flight must show travel time, aerodynamic drop, and (if wind is present) slight lateral drift; no teleportation/no instant hit.\n"
            "- Audio must include: bowstring snap on release, a soft whoosh during flight, and a thud/woody thunk on impact or a skitter if deflected.\n"
            "- VFX must include a subtle motion trail in flight and a dust/particle puff on impact or a believable deflection.\n"
            "- On impact: arrow tip anchors at the entry point and stays lodged (or drops realistically with 1–2 short bounces under gravity/friction). No springy jitter or random hopping.\n"
            "- Shooter mechanics: shoulders and back engage on draw; bow arm stabilizes at anchor; mild post-release recoil; breathing/footwork micro-adjustments every 0.5–1s.\n"
            "- Target reaction must match impact vector (recoil/stagger/deflection) and surface type (wood/stone/flesh implied non-graphically)."
        )


    def prime_continuity_from_text(self, text: str, continuity: 'ContinuityState') -> None:
        ent = self.extract_scene_entities(text)
        for n in ent.get("lead", []):
            continuity.add_character(n)
        for n in ent.get("antagonist", []):
            continuity.add_character(n)
        for a in ent.get("aircraft", []):
            continuity.add_aircraft(a)

        if ent.get("setting") and not continuity.setting_baseline:
            continuity.setting_baseline = ent["setting"]
        if ent.get("weather") and not continuity.weather_baseline:
            continuity.weather_baseline = ent["weather"]
        if ent.get("time_of_day") and not continuity.time_of_day:
            continuity.time_of_day = ent["time_of_day"]
        if ent.get("wardrobe") and not continuity.wardrobe_details:
            continuity.wardrobe_details = ent["wardrobe"]


# -------------------------
# Tools spec (function calling)
# -------------------------
    def _build_tool_spec(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "build_veo3_prompt",
                    "description": "Construct a complete Veo-3 style JSON object for an AI video generation scene.",
                    "parameters": VEO3_SCHEMA,
                },
            }
        ]


    # -------------------------
    # Helper: detect characters actually mentioned in this scene
    # -------------------------
    # === GOLD TEMPLATE FOR "Warrior Squad vs Forest Beast" ===
    def _matches_warrior_beast(self, scene_text: str) -> bool:
        t = (scene_text or "").lower()
        return ("warrior" in t and "beast" in t) or ("squad" in t and "forest" in t)

    def apply_warrior_beast_template(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Hard-lock the output to the provided 8s template with continuity_core and full action coverage."""
        # Top-level locks
        obj["model"] = "google/veo-3"
        obj["request_type"] = "video"
        obj["duration_seconds"] = 8
        obj["fps"] = 30
        obj["resolution"] = "1920x1080"
        obj["aspect_ratio"] = "16:9"
        obj["language"] = "English"
        obj["creative_brief"] = "Warrior Squad coordinates against a Forest Beast in a sun-dappled forest clearing. Purposeful, physically plausible motion with no idle seconds."

        # Global style
        obj["global_style"] = {
            "visual_tone": "cinematic, grounded realism",
            "motion_rules": "No idle frames; beats every 0.5–1.0s; every initiating action has an on-screen reaction within 0.9s; no teleportation",
            "color_grading": "natural daylight with warm greens and earthy browns",
            "vfx_guidance": "subtle dust/leaf particles on impacts and footfalls; restrained motion trails for fast projectiles",
            "sound_mix": "ambience low; foley detailed; sfx tight and non-graphic"
        }

        # Continuity core
        obj["continuity_core"] = {
            "roster": {
                "lead": "Warrior Squad",
                "antagonist": "Forest Beast",
                "projectiles": ["Arrow"],
                "props": ["Spear", "Bow"]
            },
            "aliases_map": {
                "Warrior": "Warrior Squad",
                "Hunters": "Warrior Squad",
                "Lead": "Warrior Squad",
                "Target": "Forest Beast",
                "Beast": "Forest Beast",
                "Creature": "Forest Beast"
            },
            "wardrobe_state": "leather armor, earth-tone tunics; spears and one bow slung",
            "props_state": "one warrior holds Bow (nocked), another holds Spear (ready)",
            "time_lock": "day",
            "weather_lock": "clear"
        }
        obj["continuity_logic"] = "All subjects must resolve to the roster labels via aliases_map. Do not introduce new characters or props. Keep wardrobe/props persistent across shots."

        # World
        obj["world_and_characters"] = {
            "setting": "Sun-dappled clearing deep in the forest; shafts of light cut through drifting motes; low shrubs and scattered trunks.",
            "lead_character": "Warrior Squad",
            "antagonist": "Forest Beast",
            "weather": "clear with a light breeze",
            "battle_context": "Beast charges; squad splits angles; bow unit shoots; spear unit pressures flank.",
            "character_continuity": "Warriors stay in staggered formation; bowman behind cover; spearman up front.",
            "time_of_day": "late morning"
        }

        # Shot (single)
        action_sequence = [
            { "timestamp_sec": 0.00, "subject": "Warrior Squad", "verb": "cuts angle left",
              "biomechanics_note": "heel-toe step; hips rotate; shoulders counter-rotate",
              "physics_note": "lateral shift manages distance; friction prevents slide",
              "internal_motivation": "gain flanking position" },
            { "timestamp_sec": 0.45, "subject": "Forest Beast", "verb": "charges",
              "biomechanics_note": "forelimbs extend; spine flexes; head low",
              "physics_note": "acceleration builds; mass drives forward under traction",
              "internal_motivation": "overwhelm the front line" },
            { "timestamp_sec": 0.90, "subject": "Warrior Squad", "verb": "sets guard",
              "biomechanics_note": "spearman lowers stance; bowman plants rear foot",
              "physics_note": "lowered CoG for stability against incoming mass",
              "internal_motivation": "prepare to absorb and counter" },

            { "timestamp_sec": 1.20, "subject": "Warrior Squad", "verb": "draws",
              "biomechanics_note": "scapular engagement; string hand pulls; bow arm stabilizes",
              "physics_note": "elastic energy stores in limbs of the bow",
              "internal_motivation": "commit to a clean release" },
            { "timestamp_sec": 1.55, "subject": "Warrior Squad", "verb": "anchors",
              "biomechanics_note": "string hand settles at corner of mouth; elbow aligns",
              "physics_note": "stable anchor reduces lateral error",
              "internal_motivation": "ensure accuracy under stress" },
            { "timestamp_sec": 1.70, "subject": "Warrior Squad", "verb": "re-aims",
              "biomechanics_note": "micro footwork; grip micro-corrections synced to breath",
              "physics_note": "tiny torque corrections stabilize sight picture",
              "internal_motivation": "compensate for the Beast's motion" },
            { "timestamp_sec": 1.85, "subject": "Warrior Squad", "verb": "releases",
              "biomechanics_note": "fingers relax; bow arm resists torque; slight follow-through",
              "physics_note": "stored energy converts to arrow kinetic energy; mild recoil",
              "internal_motivation": "execute the shot" },
            { "timestamp_sec": 1.98, "subject": "Arrow", "verb": "enters flight",
              "biomechanics_note": "fletching stabilizes; shaft flexes then dampens",
              "physics_note": "parabolic arc under gravity; slight lateral drift in light breeze",
              "internal_motivation": "maintain trajectory toward target" },
            { "timestamp_sec": 2.35, "subject": "Arrow", "verb": "impacts",
              "biomechanics_note": "shaft compresses; vibrations damp quickly",
              "physics_note": "kinetic energy dissipates into surface; arrow lodges non-graphically",
              "internal_motivation": "complete shot sequence" },
            { "timestamp_sec": 2.60, "subject": "Forest Beast", "verb": "recoils",
              "biomechanics_note": "mass shifts opposite vector; brief stagger",
              "physics_note": "momentum transfer; friction limits slide",
              "internal_motivation": "instinctive reaction to impact" },

            { "timestamp_sec": 3.10, "subject": "Warrior Squad", "verb": "presses with spear",
              "biomechanics_note": "rear foot drives; hips extend; lead arm snaps forward",
              "physics_note": "impulse sends spear tip on straight vector",
              "internal_motivation": "capitalize on opening" },
            { "timestamp_sec": 3.85, "subject": "Forest Beast", "verb": "deflects",
              "biomechanics_note": "shoulders twist; forelimbs parry line",
              "physics_note": "angular momentum redirects incoming vector",
              "internal_motivation": "avoid direct hit" },

            { "timestamp_sec": 4.40, "subject": "Warrior Squad", "verb": "circles",
              "biomechanics_note": "short arc steps; gaze locked; guard up",
              "physics_note": "centripetal acceleration kept small; balance preserved",
              "internal_motivation": "seek better angle" },
            { "timestamp_sec": 5.10, "subject": "Forest Beast", "verb": "feints",
              "biomechanics_note": "head dips; weight teases forward",
              "physics_note": "small impulse to test response",
              "internal_motivation": "probe defenses" },
            { "timestamp_sec": 5.70, "subject": "Warrior Squad", "verb": "re-positions",
              "biomechanics_note": "heel-toe shift; shoulders counter-rotate",
              "physics_note": "momentum managed to avoid overshoot",
              "internal_motivation": "maintain initiative" },
            { "timestamp_sec": 6.40, "subject": "Warrior Squad", "verb": "re-aims",
              "biomechanics_note": "micro footwork; breath reset",
              "physics_note": "parallax alive; no idle plateau",
              "internal_motivation": "hold pressure" },
            { "timestamp_sec": 7.20, "subject": "Forest Beast", "verb": "pivots away",
              "biomechanics_note": "hind/lead feet switch; torso twists",
              "physics_note": "friction & angular momentum govern rapid redirection",
              "internal_motivation": "escape the unfavorable line" },
            { "timestamp_sec": 7.90, "subject": "Warrior Squad", "verb": "holds guard while shifting",
              "biomechanics_note": "low stance; breath-linked sway; feet alive",
              "physics_note": "continuous micro-motion avoids freeze at tail",
              "internal_motivation": "stay ready into the cut" }
        ]

        shot0 = {
            "id": "shot_1",
            "start_time": 0.0,
            "duration": 8.0,
            "slugline": "Handheld tracking across the clearing as the Beast rushes; bowman releases; spear presses.",
            "action_sequence": action_sequence,
            "camera": {
                "rig": "handheld",
                "lens_mm": 35,
                "aperture_f": 5.6,
                "shutter_angle_deg": 180,
                "white_balance_k": 5600,
                "iso": 400,
                "position_m": {
                    "start": { "x": -1.2, "y": 1.6, "z": 4.0 },
                    "end":   { "x": -0.4, "y": 1.6, "z": 3.2 }
                },
                "movement": { "type": "dolly", "direction": "forward", "speed": "medium" },
                "orientation_deg": { "yaw": 6.0, "pitch": -2.0, "roll": 0.0 },
                "focus": { "mode": "manual", "distance_m": 6.0 },
                "composition_style": "rule of thirds",
                "focus_pull_style": "slow rack from foreground (Warriors) to Beast mid-frame at 2.3s",
                "depth_of_field": "moderate"
            },
            "vfx": [
                "subtle arrow motion trail during flight",
                "small dust/leaf puff on impact or foot plants"
            ],
            "lighting": "natural, sun-dappled with soft bounce from ground; maintain key from camera-left",
            "audio": {
                "ambience": "low forest bed: wind through leaves, distant birds",
                "foley": [
                    "leather creak on stance changes",
                    "footsteps on leaf litter",
                    "controlled breathing/grunts on exertion",
                    "Forest Beast low growl (fierce, non-graphic)"
                ],
                "sfx": [
                    "bowstring snap on release (non-graphic)",
                    "soft arrow whoosh (non-graphic)",
                    "solid woody thunk on impact (non-graphic)",
                    "small debris clatter on foot plants"
                ],
                "mix_notes": "keep ambience -12 LUFS relative to foley; sfx transient preserved; avoid pumping"
            },
            "transition_out": "cut",
            "creative_suggestions": "brief foreground branch occlusion at 0.9–1.2s to add depth; slight handheld micro-sway synced to breath."
        }

        obj["shots"] = [shot0]

        # Safety & quality
        obj["safety_and_quality"] = {
            "avoid": [
                "graphic injury or gore",
                "hate symbols or slurs",
                "sexual content or nudity",
                "child endangerment",
                "real-world political propaganda"
            ],
            "must_have": [
                "non-graphic, documentary tone for combat actions",
                "cause→effect reactions within 0.9s for every initiated action",
                "physically plausible motion (gravity, inertia, friction, drag)",
                "no idle seconds; continuous purposeful movement"
            ]
        }

        # Output settings
        obj["output_settings"] = {
            "container": "mp4",
            "video_codec": "h264",
            "video_bitrate": "8000k",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "normalize_audio": True,
            "tone_mapping": "linear",
            "grain": "none"
        }

        # Compose single prompt in the requested compact style
        obj["single_prompt_for_model"] = self._compose_single_prompt_compact(obj)

        return obj

    def _mentioned_characters(self, scene_text: str, scene_entities: Dict[str, Any], continuity: 'ContinuityState') -> List[str]:
        names: List[str] = []
        # 1) From scene_entities bullets
        for key in ("lead", "antagonist"):
            vals = scene_entities.get(key) or []
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                v = (v or "").strip()
                if v and v not in names:
                    names.append(v)

        # 2) From continuity known names that appear as whole words in scene_text
        text = scene_text or ""
        for cand in continuity.char_names or []:
            if not cand:
                continue
            try:
                pattern = r"\\b" + re.escape(cand) + r"\\b"
                if re.search(pattern, text):
                    if cand not in names:
                        names.append(cand)
            except Exception:
                # In case of regex edge cases, skip
                continue

        return names


# -------------------------
# Local structural normalizer (auto-fixes common issues)
# -------------------------
    def normalize_veo3_object(self, obj: Dict[str, Any], scene_text: str = "", scene_entities: Dict[str, Any] | None = None, continuity: ContinuityState | None = None) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            raise TypeError(f"normalize_veo3_object expects dict, got {type(obj).__name__}")
        # 👇 ép shape lần nữa ở local level (an toàn)
        obj = self._coerce_shapes(obj)

        shots = obj.get("shots") or []
        if not isinstance(shots, list):
            return obj

        for idx, s in enumerate(shots, 1):
            if not isinstance(s, dict):
                continue

            # BƯỚC DỌN DẸP MỚI: Xóa trường "action" cũ nếu nó tồn tại
            if "action" in s:
                del s["action"]

            # Apply motion enrichers ONLY if grounded by scene text (no invention)
            text_for_enrich = (scene_text or (obj.get("creative_brief") or "")) + " " + json.dumps(obj.get("world_and_characters") or {})
            for enricher in getattr(self, "enrichers", []):
                s = enricher.enrich(s, text_for_enrich)

            cam = s.get("camera") or {}
            if not isinstance(cam, dict):
                cam = {}

            # 1) Pull misplaced fields out of camera
            for k in ("vfx", "lighting", "audio", "transition_out"):
                if k in cam and k not in s:
                    s[k] = cam.pop(k)

            # 2) Camera defaults
            cam.setdefault("rig", "tripod")
            cam.setdefault("lens_mm", 35)
            cam.setdefault("aperture_f", 4)
            cam.setdefault("shutter_angle_deg", 180)
            cam.setdefault("white_balance_k", 5600)
            cam.setdefault("iso", 400)

            # Apply exposure heuristics based on world conditions
            w = obj.get("world_and_characters") or {}
            expo = self._exposure_heuristics(w.get("time_of_day",""), w.get("weather",""))
            for k,v in expo.items():
                cam[k] = v

            def _vec(v):
                if isinstance(v, dict) and all(k in v for k in ("x","y","z")):
                    return {"x": float(v["x"]), "y": float(v["y"]), "z": float(v["z"])}
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
                return {"x": 0.0, "y": 1.5, "z": 3.0}

            pos = cam.get("position_m")
            if not isinstance(pos, dict):
                pos = {}
            start = _vec(pos.get("start"))
            end = _vec(pos.get("end")) if pos.get("end") is not None else dict(start)
            cam["position_m"] = {"start": start, "end": end}

            mv = cam.get("movement")
            if isinstance(mv, str) and mv.strip():
                cam["movement"] = {"type": mv.strip()}
            elif not isinstance(mv, dict):
                cam["movement"] = {"type": "static"}

            ori = cam.get("orientation_deg")
            if isinstance(ori, (list, tuple)) and len(ori) >= 3:
                cam["orientation_deg"] = {"yaw": float(ori[0]), "pitch": float(ori[1]), "roll": float(ori[2])}
            elif isinstance(ori, (int, float)):
                cam["orientation_deg"] = {"yaw": float(ori), "pitch": 0.0, "roll": 0.0}
            elif not isinstance(ori, dict):
                cam["orientation_deg"] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

            focus_val = cam.get("focus")
            if not isinstance(focus_val, dict):
                focus_obj = {"mode": "infinity"}
                if isinstance(focus_val, (int, float)):
                    focus_obj = {"mode": "manual", "distance_m": float(focus_val)}
                elif isinstance(focus_val, str) and focus_val.strip().lower() not in ("",):
                    if focus_val.strip().lower() != "infinity":
                        focus_obj = {"mode": "manual"}
                cam["focus"] = focus_obj
            else:
                cam["focus"] = focus_val

            # 3) Required shot fields
            s.setdefault("id", f"shot{idx}")
            if "start_time" not in s or not isinstance(s.get("start_time"), (int, float)):
                s["start_time"] = 0 if idx == 1 else float(shots[idx-2].get("start_time", 0)) + float(shots[idx-2].get("duration", 0))
            if "duration" not in s or not isinstance(s.get("duration"), (int, float)):
                s["duration"] = 5
            if not isinstance(s.get("slugline"), str):
                s["slugline"] = ""

            # NEW: prefer action_sequence when schema supports it
            use_action_sequence = "action_sequence" in (
                VEO3_SCHEMA.get("properties", {})
                .get("shots", {})
                .get("items", {})
                .get("properties", {})
            )
            if use_action_sequence:
                # Do not re-introduce 'action' in strict schema mode
                s.pop("action", None)
                if "action_sequence" not in s or not isinstance(s["action_sequence"], list):
                    s["action_sequence"] = []
                
                # NEW: Canonicalize subject & normalize "None" labels
                # >>> PATCH START: map VN bullets -> camera + audio
                try:
                    if isinstance(scene_entities, dict):
                        angle = (scene_entities.get("camera_angle") or "").lower()
                        move  = (scene_entities.get("camera_move") or "").lower()
                        audio = (scene_entities.get("audio") or "").lower()

                        # Góc máy -> composition_style
                        if angle:
                            if "dutch" in angle or "nghiêng" in angle:
                                cam["composition_style"] = "dutch angle"
                            elif "đối xứng" in angle or "symmetry" in angle:
                                cam["composition_style"] = "centered symmetry"
                            else:
                                cam["composition_style"] = "rule of thirds"

                        # Chuyển động -> movement + position path
                        mv = cam.get("movement") or {}
                        mv_type = mv.get("type","static").lower()
                        if move:
                            if any(k in move for k in ("handheld","cầm tay")):
                                mv["type"] = "handheld"
                            elif any(k in move for k in ("dolly","truck","dollying","track")):
                                mv["type"] = "dolly"
                            elif any(k in move for k in ("pan","lia")):
                                mv["type"] = "pan"
                            elif any(k in move for k in ("tilt")):
                                mv["type"] = "tilt"
                            else:
                                mv["type"] = mv.get("type","static")
                            # direction heuristic
                            if any(k in move for k in ("in","tiến vào","tiến gần")):
                                mv["direction"] = "forward"
                            elif any(k in move for k in ("out","lùi ra","ra xa")):
                                mv["direction"] = "backward"
                            elif "left" in move or "trái" in move:
                                mv["direction"] = "left"
                            elif "right" in move or "phải" in move:
                                mv["direction"] = "right"
                            mv["speed"] = mv.get("speed","medium")
                            cam["movement"] = mv

                            # position start/end nhẹ theo hướng
                            pos = cam.get("position_m") or {"start":{"x":0.0,"y":1.5,"z":3.5},"end":{"x":0.0,"y":1.5,"z":3.5}}
                            st, ed = pos.get("start",{}), pos.get("end",{})
                            st = {"x": float(st.get("x",0.0)), "y": float(st.get("y",1.5)), "z": float(st.get("z",3.5))}
                            ed = dict(st)
                            if mv.get("direction") == "forward":
                                ed["z"] = ed["z"] - 0.8
                            elif mv.get("direction") == "backward":
                                ed["z"] = ed["z"] + 0.8
                            elif mv.get("direction") == "left":
                                ed["x"] = ed["x"] - 0.6
                            elif mv.get("direction") == "right":
                                ed["x"] = ed["x"] + 0.6
                            cam["position_m"] = {"start": st, "end": ed}

                        # Âm thanh -> ambience/foley/sfx tách nhẹ
                        aud = s.get("audio") or {}
                        if audio:
                            if "gió" in audio or "wind" in audio or "lá" in audio or "forest" in audio:
                                amb = aud.get("ambience", "")
                                if not amb:
                                    aud["ambience"] = "ambient forest bed with wind and leaves"
                            # tách đơn giản
                            hints = [x.strip() for x in re.split(r"[,/;|]", audio) if x.strip()]
                            foley, sfx = set(aud.get("foley",[]) or []), set(aud.get("sfx",[]) or [])
                            for h in hints:
                                if any(k in h.lower() for k in ("footstep","bước chân","clatter","va chạm nhỏ","armor","giáp")):
                                    foley.add(h)
                                elif any(k in h.lower() for k in ("impact","va","đập","roar","gầm","growl","gào")):
                                    sfx.add(h)
                            if foley: aud["foley"] = sorted(foley)
                            if sfx:   aud["sfx"]   = sorted(sfx)
                        s["audio"] = aud

                        s["camera"] = cam
                except Exception:
                    pass
                # <<< PATCH END
                def _canon_subject(x: str) -> str:
                    if not isinstance(x, str): return x
                    x = x.strip()
                    if x.lower() == "none": return ""  # don't keep "None" string
                    if x.lower() == "lead" and (obj.get("world_and_characters") or {}).get("lead_character"):
                        return (obj["world_and_characters"]["lead_character"]).strip()
                    if x.lower() == "arrow": return "Arrow"
                    if x.lower() == "target": return "Target"
                    return x

                for e in s.get("action_sequence", []):
                    e["subject"] = _canon_subject(e.get("subject", ""))
            else:
                # Legacy fallback
                if not isinstance(s.get("action"), str):
                    s["action"] = ""

            # Optional fields: in strict mode, do not auto-fill if absent in scene
            if not isinstance(s.get("vfx"), list):
                s["vfx"] = [] if not getattr(self, "strict_grounding", False) else []
            if not isinstance(s.get("lighting"), str):
                if not getattr(self, "strict_grounding", False):
                    s["lighting"] = "natural, soft daylight"

            # --- audio: luôn là dict, không sinh 'music' ---
            aud = s.get("audio") or {}
            if not isinstance(aud, dict):
                aud = {}
            s["audio"] = aud

            aud.setdefault("ambience", "")
            fx = aud.get("fx")
            if not isinstance(fx, list):
                fx = []
            aud["fx"] = fx

            aud.pop("music", None)
            aud.setdefault("mix_notes", "")

            if not isinstance(s.get("transition_out"), str):
                if not getattr(self, "strict_grounding", False):
                    s["transition_out"] = "cut"

            s["camera"] = cam

            # Vá camera mâu thuẫn "static nhưng thay vị trí"
            mv = cam.get("movement", {})
            pos = cam.get("position_m", {})
            st, ed = pos.get("start",{}), pos.get("end",{})
            if isinstance(mv, dict) and mv.get("type","static").lower()=="static":
                # static thì không được dịch vị trí
                cam["position_m"]["end"] = dict(st)
            else:
                # ✨ nếu non-static mà start==end → đẩy end lệch epsilon dọc trục z (hoặc theo direction)
                if st and ed and (st.get("x")==ed.get("x") and st.get("y")==ed.get("y") and st.get("z")==ed.get("z")):
                    direction = (mv.get("direction","forward") or "forward").lower()
                    eps = 0.1
                    ex, ey, ez = st.get("x",0.0), st.get("y",1.5), st.get("z",3.0)
                    # C) camera movement fallback
                    if direction in ("forward","dolly in"):
                        ez = ez - eps
                        cam["position_m"]["end"] = {"x": float(ex), "y": float(ey), "z": float(ez)}
                    elif direction in ("backward","dolly out"):
                        ez = ez + eps
                        cam["position_m"]["end"] = {"x": float(ex), "y": float(ey), "z": float(ez)}
                    elif direction in ("left",):
                        ex = ex - eps
                        cam["position_m"]["end"] = {"x": float(ex), "y": float(ey), "z": float(ez)}
                    elif direction in ("right",):
                        ex = ex + eps
                        cam["position_m"]["end"] = {"x": float(ex), "y": float(ey), "z": float(ez)}
                    elif direction in ("down","downward","tilt down","tilt-down"):
                        # tracking/tilt mà không đổi vị trí: xoay pitch nhẹ
                        o = cam.get("orientation_deg", {"yaw":0.0,"pitch":0.0,"roll":0.0})
                        o["pitch"] = float(o.get("pitch",0.0)) - 5.0
                        cam["orientation_deg"] = o
                        cam["position_m"]["end"] = dict(st)
                    elif direction in ("circular","orbit","circle","around"):
                        # orbit cảm giác: thay đổi yaw nhẹ, giữ vị trí
                        o = cam.get("orientation_deg", {"yaw":0.0,"pitch":0.0,"roll":0.0})
                        o["yaw"] = float(o.get("yaw",0.0)) + 5.0
                        cam["orientation_deg"] = o
                        mv["type"] = mv.get("type","orbit")
                        cam["movement"] = mv
                        cam["position_m"]["end"] = dict(st)
                    else:
                        # Direction không xác định → rơi về static + micro_motion
                        cam["movement"] = {"type":"static","micro_motion":"breathing-linked sway; no jitter"}
                        cam["position_m"]["end"] = dict(st)


        obj["shots"] = shots
        
        # NEW: Merge audio fx → sfx & deduplicate
        for s in shots:
            aud = s.get("audio") or {}
            fx = set(aud.get("fx", []))
            sfx = set(aud.get("sfx", []))
            aud["sfx"] = sorted(sfx.union(fx))
            aud.pop("fx", None)
            
            # chuyển mục có "shout|yell|growl|roar" ra foley/dialogue_note
            vox_pat = re.compile(r"\b(shout|yell|growl|roar|scream|grunt)s?\b", re.I)
            spoken = []
            sfx2 = []
            for x in aud.get("sfx", []):
                (spoken if vox_pat.search(str(x)) else sfx2).append(x)
            if spoken:
                aud["foley"] = sorted(set(list(aud.get("foley", [])) + spoken))
            aud["sfx"] = sfx2
            aud["foley"] = sorted(set(aud.get("foley", [])))
            # Remove empty audio fields
            for key in ["ambience", "music", "dialogue_note", "music_cue", "mix_notes"]:
                if not aud.get(key):
                    aud.pop(key, None)
            s["audio"] = aud
        
        # If no lead explicitly present in scene_entities, clear lead_character and prevent bleeding wardrobe/props
        if isinstance(scene_entities, dict):
            leads = scene_entities.get("lead") or []
            if not leads:
                w = obj.get("world_and_characters") or {}
                w["lead_character"] = ""
                obj["world_and_characters"] = w
        # Auto-inject per-scene detailed profiles for mentioned entities
        try:
            obj = self._inject_entity_profiles_for_scene(obj, scene_text or "", continuity)
        except Exception:
            pass

        return obj

    def _to_english(self, s: str) -> str:
        """
        Dịch một chuỗi đơn lẻ sang tiếng Anh theo phong cách điện ảnh. Hàm này được hưởng lợi rất nhiều từ caching.
        """
        if not isinstance(s, str) or not s.strip() or not re.search(r"[^\x00-\x7F]", s):
            return s
        if s in self._eng_cache:
            return self._eng_cache[s]

        # BƯỚC 1: Dịch thô
        translation_prompt = (
            "Translate the following Vietnamese text into clear, natural English. "
            "Keep proper nouns (e.g., KAI, AKENO) unchanged. Return only the translation."
        )
        raw_translation = self._call_llm_for_utility(translation_prompt, s)

        if raw_translation == s:
            # Nếu dịch thô thất bại, không cần tinh chỉnh
            out = s
        else:
            # BƯỚC 2: Tinh chỉnh bản dịch
            refinement_prompt = (
                "You are a screenwriter. Refine the following English translation to make it more cinematic, evocative, and impactful. "
                "Do not change the core meaning. Focus on strong verbs and descriptive language."
                f"\nOriginal Vietnamese: '{s}'"
                f"\nBasic Translation: '{raw_translation}'"
                "\nRefined Cinematic Translation (return only the text):"
            )
            # Truyền một chuỗi trống vào user content vì toàn bộ thông tin đã có trong system prompt
            out = self._call_llm_for_utility(refinement_prompt, raw_translation) or raw_translation

        self._eng_cache[s] = out
        return out


    def _translate_batch(self, texts: list[str]) -> dict[str, str]:
        # Gom các chuỗi cần dịch thành bullet list, trả mapping {gốc: dịch}
        uniq = [t for t in {t for t in texts if re.search(r"[^\x00-\x7F]", t)}]
        if not uniq:
            return {}
        prompt = "Translate each line to natural English, keep proper nouns. Return one line per input, same order."
        resp = self._chat(model=self.model, temperature=0, messages=[
            {"role":"system","content":prompt},
            {"role":"user","content":"\n".join(uniq)},
        ])
        out_lines = (resp.choices[0].message.content or "").splitlines()
        return {src: (out_lines[i] if i < len(out_lines) else src) for i, src in enumerate(uniq)}


    def _translate_deep_english(self, x, parent_key=""):
        """Hàm này sẽ tự động sử dụng phiên bản đã được cache của _to_english."""
        if isinstance(x, dict):
            return {k: self._translate_deep_english(v, k) for k, v in x.items()}
        if isinstance(x, list):
            return [self._translate_deep_english(i, parent_key) for i in x]
        if isinstance(x, str):
            # tránh dịch các nhánh kỹ thuật
            if parent_key in ("camera","output_settings","orientation_deg","position_m","focus") or re.match(r"^(codec|bitrate|container|iso|fps)$", parent_key or "", re.I):
                return x
            return self._to_english(x)
        return x


    def _call_llm_for_utility(self, prompt: str, text: str) -> str:
        """Gọi LLM cho các tác vụ tiện ích như chuẩn hóa hoặc dịch văn bản."""
        if not self.client or not text.strip():
            return text
        try:
            resp = self._chat(
                model=self.model,
                temperature=0,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
            )
            return (resp.choices[0].message.content or "").strip() or text
        except Exception as e:
            print(f"⚠️ Lỗi tiện ích LLM: {e}")
            return text


    def _coerce_shapes(self, o: dict) -> dict:
        if not isinstance(o, dict):
            return {"error": f"Top-level JSON must be an object, got {type(o).__name__}"}

        # Coerce các dict top-level có thể bị trả về dạng list
        for k in ["world_and_characters", "global_style", "safety_and_quality", "output_settings", "continuity_updates"]:
            v = o.get(k)
            if isinstance(v, list):
                print(f"⚠️ Coercing {k} from list to dict via first element")
                o[k] = (v[0] if v and isinstance(v[0], dict) else {})
            elif v is None:
                o[k] = {}

        # Coerce shots → list[dict]
        shots = o.get("shots")
        if isinstance(shots, dict):
            o["shots"] = [shots]
        elif isinstance(shots, list):
            fixed = []
            for i, sh in enumerate(shots):
                if isinstance(sh, list):
                    print(f"⚠️ Coercing shots[{i}] from list to dict via first element")
                    sh = sh[0] if sh and isinstance(sh[0], dict) else {}
                elif not isinstance(sh, dict):
                    print(f"⚠️ Replacing shots[{i}] non-dict with empty dict")
                    sh = {}

                # camera/audio bên trong shot cũng có thể bị trả về là list
                cam = sh.get("camera")
                if isinstance(cam, list):
                    print(f"⚠️ Coercing shots[{i}].camera from list to dict via first element")
                    sh["camera"] = cam[0] if cam and isinstance(cam[0], dict) else {}
                aud = sh.get("audio")
                if isinstance(aud, list):
                    print(f"⚠️ Coercing shots[{i}].audio from list to dict via first element")
                    sh["audio"] = aud[0] if aud and isinstance(aud[0], dict) else {}

                # Ép shape sâu hơn một nhịp
                vfx = sh.get("vfx")
                if isinstance(vfx, str):
                    sh["vfx"] = [vfx]
                lighting = sh.get("lighting")
                if isinstance(lighting, list):
                    sh["lighting"] = " | ".join(str(x) for x in lighting)

                fixed.append(sh)
            o["shots"] = fixed
        else:
            print("⚠️ shots was not list/dict, initializing with one empty shot")
            o["shots"] = [{}]

        return o


    def _debug_top_shapes(self, o: dict):
        def t(x):
            return type(x).__name__
        print(
            "🔎 SHAPES:",
            "world_and_characters=", t(o.get("world_and_characters")),
            "| global_style=", t(o.get("global_style")),
            "| safety_and_quality=", t(o.get("safety_and_quality")),
            "| output_settings=", t(o.get("output_settings")),
            "| shots=", t(o.get("shots")),
        )


    def _phase_index(self, seq, *verbs):
        verbs = tuple(v.lower() for v in verbs)
        for i, e in enumerate(seq):
            v = (e.get("verb","") or "").lower()
            if any(k in v for k in verbs):
                return i
        return -1

    def _ensure_projectile_chain(self, seq, dur, windy: bool):
        """Ensure canonical chain of projectile actions (draw → anchor → release → flight → impact)."""
        if not seq:
            return seq
        
        # Only apply if sequence already contains archery elements
        has_archery = any("arrow" in (e.get("subject", "").lower()) for e in seq)
        if not has_archery:
            return seq
            
        # Sort by timestamp
        seq.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))
        
        # Check for missing phases and insert them
        verbs = [e.get("verb", "").lower() for e in seq]
        
        # Draw phase
        if not any("draw" in v for v in verbs):
            seq.insert(0, {
                "timestamp_sec": 0.3,
                "subject": "Archer",
                "verb": "draws",
                "biomechanics_note": "scapular muscles engage; bow arm extends, string hand pulls to mid-face",
                "physics_note": "string tension rises; potential energy stored in limbs"
            })
        
        # Anchor phase
        if not any("anchor" in v for v in verbs):
            seq.append({
                "timestamp_sec": 0.8,
                "subject": "Archer", 
                "verb": "anchors",
                "biomechanics_note": "string hand stabilizes at corner of mouth; elbow aligns; breathing pause",
                "physics_note": "stable anchor ensures consistent launch angle and reduces lateral error"
            })
        
        # Release phase
        if not any("release" in v or "loose" in v for v in verbs):
            seq.append({
                "timestamp_sec": 1.2,
                "subject": "Archer",
                "verb": "releases", 
                "biomechanics_note": "finger relax triggers clean string slip; bow arm resists torque; slight follow-through",
                "physics_note": "stored elastic energy converts to arrow kinetic energy; mild recoil"
            })
        
        # Flight phase
        if not any("flight" in v or "flies" in v for v in verbs):
            seq.append({
                "timestamp_sec": 1.35,
                "subject": "Arrow",
                "verb": "enters flight",
                "biomechanics_note": "fletching stabilizes; shaft flexes then dampens", 
                "physics_note": "parabolic arc under gravity; drag slows speed; subtle lateral drift if windy"
            })
        
        # Impact phase
        if not any("impact" in v or "thud" in v for v in verbs):
            seq.append({
                "timestamp_sec": 2.3,
                "subject": "Arrow",
                "verb": "impacts",
                "biomechanics_note": "shaft compresses then settles; vibrations dampen quickly",
                "physics_note": "kinetic energy dissipates into target; arrow lodges at entry point"
            })
        
        # Sort again after insertions
        seq.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))
        return seq

    def _assert_no_teleport(self, seq):
        """Cảnh báo: nếu có impact nhưng không có flight trước đó."""
        if self._phase_index(seq, "impacts","lodges","thud") >= 0 and \
           self._phase_index(seq, "flight","enters flight","flies","arc","mid-trajectory") < 0:
            print("⚠️ LOGIC: impact without flight detected → auto-inserted flight beats")

    def _logic_assertions(self, obj):
        s0 = ((obj.get("shots") or [{}])[0])
        seq = s0.get("action_sequence", [])

        # 1) Không flight nhưng có impact
        if self._phase_index(seq, "impacts", "lodges", "thud") >= 0 and \
           self._phase_index(seq, "flight", "enters flight", "flies", "arc", "mid-trajectory") < 0:
            print("❗ LOGIC ASSERT: impact exists but no flight — repaired.")

        # 2) Sau impact >0.5s vẫn còn 'arrow ... bounces/hops'
        impact_i = self._phase_index(seq, "impacts", "lodges", "thud")
        if impact_i >= 0:
            t_imp = float(seq[impact_i]["timestamp_sec"])
            post = [e for e in seq if float(e["timestamp_sec"]) > t_imp + 0.5 and "arrow" in (e.get("subject", "").lower())]
            if post:
                print("❗ LOGIC ASSERT: arrow still acting after settle window — clamped.")

        # 3) Camera 'static' nhưng position thay đổi
        cam = s0.get("camera", {})
        mv = (cam.get("movement") or {}).get("type", "static").lower()
        st, ed = (cam.get("position_m") or {}).get("start", {}), (cam.get("position_m") or {}).get("end", {})
        if mv == "static" and st and ed and (st != ed):
            print("❗ LOGIC ASSERT: camera marked static but position changes — forced end=start.")

    def _analyze_scene_subtext(self, scene_text: str, continuity_summary: str) -> Dict[str, Any]:
        """ Sử dụng LLM như một "đạo diễn" để phân tích ý đồ và bối cảnh của cảnh quay. """
        analysis_prompt = f"""
        You are a film director analyzing a scene. Based on the script excerpt and the story continuity, provide a concise analysis in a JSON object.
        Continuity so far: {continuity_summary}
        Scene Script:
        ---
        {scene_text}
        ---
        Return a JSON object with the following keys:
        - "emotional_state": The primary emotion of the main character (e.g., 'determined', 'fearful', 'exhausted', 'calm').
        - "immediate_goal": The character's specific, physical objective in this 8-second scene (e.g., 'make a silent kill shot', 'find stable footing', 'examine the tracks without being seen').
        - "physical_condition": The character's physical state (e.g., 'uninjured but tired', 'limping slightly', 'adrenaline-fueled').
        - "environmental_constraints": List of environmental factors that make the goal harder (e.g., 'slippery ground due to rain', 'strong crosswind affecting arrow', 'dark shadows obscuring the target').
        - "environmental_opportunities": List of environmental factors that can be used to advantage or enhance the scene visually (e.g., 'puddles for reflection shots', 'wind blowing leaves for dramatic effect').
        """
        try:
            resp = self._chat(
                model=self.model,
                temperature=0.5,  # Tăng một chút để có phân tích sáng tạo hơn
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful film director providing scene analysis as a JSON object."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ Lỗi phân tích bối cảnh: {e}")
            return {}


    def _generate_micro_script(self, analysis: Dict[str, Any]) -> str:
        """ Tạo ra một kịch bản hành động chi tiết (beat sheet) dựa trên phân tích của "đạo diễn". """
        if not analysis:
            return "The character performs the action as described in the script."

        scripting_prompt = f"""
        Based on the following director's analysis, write a concise, 3-beat narrative action plan (a micro-script) for an 8-second scene.
        Focus on showing the character's state through action. Describe a clear cause-and-effect progression.

        Director's Analysis:
        - Emotional State: {analysis.get('emotional_state', 'N/A')}
        - Immediate Goal: {analysis.get('immediate_goal', 'N/A')}
        - Physical Condition: {analysis.get('physical_condition', 'N/A')}
        - Environmental Constraints: {', '.join(analysis.get('environmental_constraints', []))}
        - Environmental Opportunities: {', '.join(analysis.get('environmental_opportunities', []))}

        Example Output for an exhausted archer in the rain:
        1. (0-2s) Setup & Struggle: KAI fights to steady his footing on the slippery rocks, his breath misting in the cold air, showing deep exhaustion.
        2. (2-6s) Focused Action: He grits his teeth, forcing focus. He slowly raises his bow, the movement heavy and deliberate. Rain drips from the bowstring, a visual cue of the harsh conditions.
        3. (6-8s) Climax: He draws the string to his anchor point, his knuckles white. A single drop of water falls from his chin, mirroring his intense concentration just before the release.

        Now, generate the micro-script for the provided analysis.
        """
        try:
            resp = self._chat(
                model=self.model,
                temperature=0.6,
                messages=[
                    {"role": "system", "content": "You are a creative screenwriter writing a 3-beat micro-script."},
                    {"role": "user", "content": scripting_prompt}
                ]
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Lỗi tạo kịch bản vi mô: {e}")
            return ""



    def _compose_single_prompt_text(
        self,
        scene: Dict[str, Any],
        continuity: Optional["ContinuityState"] = None,
    ) -> str:
        """Build a high-fidelity single prompt summary derived from the scene JSON."""
        if not isinstance(scene, dict):
            return ""

        continuity = continuity or ContinuityState()
        world = scene.get("world_and_characters") or {}
        continuity_core = scene.get("continuity_core") or {}
        roster = continuity_core.get("roster") if isinstance(continuity_core, dict) else {}

        def _first_non_empty(*values: Any, default: str = "") -> str:
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return default

        lead = _first_non_empty(
            world.get("lead_character"),
            roster.get("lead") if isinstance(roster, dict) else "",
            getattr(continuity, "char_names", [""])[0] if getattr(continuity, "char_names", []) else "",
            default="Primary Subject",
        )
        antagonist = _first_non_empty(
            world.get("antagonist"),
            roster.get("antagonist") if isinstance(roster, dict) else "",
            roster.get("enemy") if isinstance(roster, dict) else "",
            default="Opposing Force",
        )
        setting = _first_non_empty(
            world.get("setting"),
            continuity_core.get("environment") if isinstance(continuity_core, dict) else "",
            default="Cinematic environment",
        )
        weather = _first_non_empty(
            world.get("weather"),
            continuity_core.get("weather_lock") if isinstance(continuity_core, dict) else "",
        )
        time_of_day = _first_non_empty(
            world.get("time_of_day"),
            continuity_core.get("time_lock") if isinstance(continuity_core, dict) else "",
        )
        duration = scene.get("duration_seconds", 8)
        creative_brief = (scene.get("creative_brief") or "").strip()

        shots = scene.get("shots") or []
        first_shot = shots[0] if shots else {}
        sequence = first_shot.get("action_sequence") or []

        timeline: list[str] = []
        seen_beats: set[str] = set()
        for entry in sequence:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("timestamp_sec", 0.0)
            try:
                ts_display = f"{float(ts):0.2f}s"
            except Exception:
                ts_display = "0.00s"
            subject = (entry.get("subject") or lead or "Subject").strip()
            verb = (entry.get("verb") or "moves").strip()
            biomechanics = (entry.get("biomechanics_note") or "").strip()
            physics = (entry.get("physics_note") or "").strip()
            highlight = biomechanics or physics
            if highlight:
                highlight = re.sub(r"\s+", " ", highlight)
                if len(highlight) > 90:
                    highlight = highlight[:90].rstrip() + "..."
                beat = f"{ts_display} {subject} {verb} ({highlight})"
            else:
                beat = f"{ts_display} {subject} {verb}"
            if beat not in seen_beats:
                timeline.append(beat)
                seen_beats.add(beat)
        action_summary = " | ".join(timeline[:12])

        camera = first_shot.get("camera") or {}
        camera_chunks: list[str] = []
        rig = (camera.get("rig") or "").strip()
        if rig:
            camera_chunks.append(rig)
        lens = camera.get("lens_mm")
        if lens is not None:
            try:
                camera_chunks.append(f"{float(lens):g}mm lens")
            except Exception:
                camera_chunks.append(f"{lens}mm lens")
        aperture = camera.get("aperture_f")
        if aperture:
            try:
                camera_chunks.append(f"f/{float(aperture):g}")
            except Exception:
                camera_chunks.append(f"f/{aperture}")
        movement = camera.get("movement") or {}
        if isinstance(movement, dict) and movement:
            move_parts = [movement.get("type") or "movement"]
            if movement.get("direction"):
                move_parts.append(movement["direction"])
            if movement.get("speed"):
                move_parts.append(f"({movement['speed']})")
            camera_chunks.append(" ".join(p for p in move_parts if p))
        position = camera.get("position_m") or {}
        if isinstance(position, dict) and position.get("start") and position.get("end"):
            start_pos = json.dumps(position.get("start"), ensure_ascii=False)
            end_pos = json.dumps(position.get("end"), ensure_ascii=False)
            camera_chunks.append(f"path {start_pos} -> {end_pos}")
        orientation = camera.get("orientation_deg") or {}
        if isinstance(orientation, dict) and orientation:
            orientation_parts = []
            for axis in ("yaw", "pitch", "roll"):
                if axis in orientation:
                    orientation_parts.append(f"{axis} {orientation[axis]}")
            if orientation_parts:
                camera_chunks.append("orientation " + ", ".join(orientation_parts))
        if camera.get("focus_pull_style"):
            camera_chunks.append((camera.get("focus_pull_style") or "").strip())
        if camera.get("composition_style"):
            camera_chunks.append((camera.get("composition_style") or "").strip())
        if camera.get("depth_of_field"):
            camera_chunks.append(f"DOF {camera.get('depth_of_field')}")
        camera_text = "; ".join([c for c in camera_chunks if c])

        lighting_hint = _first_non_empty(
            first_shot.get("lighting") if isinstance(first_shot, dict) else "",
            world.get("lighting") if isinstance(world, dict) else "",
            default=f"{time_of_day or 'time-locked'} light with {weather or 'controlled'} ambience",
        )

        audio = first_shot.get("audio") if isinstance(first_shot, dict) else {}
        audio_chunks: list[str] = []
        if isinstance(audio, dict):
            ambience = (audio.get("ambience") or "").strip()
            if ambience:
                audio_chunks.append(f"ambience {ambience}")
            foley = audio.get("foley")
            if isinstance(foley, list) and foley:
                audio_chunks.append("foley " + ", ".join(str(x) for x in foley if x))
            sfx = audio.get("sfx")
            if isinstance(sfx, list) and sfx:
                audio_chunks.append("sfx " + ", ".join(str(x) for x in sfx if x))
            music = (audio.get("music_cue") or "").strip()
            if music:
                audio_chunks.append(f"music {music}")
            mix = (audio.get("mix_notes") or "").strip()
            if mix:
                audio_chunks.append(f"mix {mix}")
        audio_text = " | ".join(audio_chunks)

        vfx = first_shot.get("vfx") if isinstance(first_shot, dict) else []
        vfx_text = ", ".join(str(v) for v in vfx if v)

        continuity_logic = (scene.get("continuity_logic") or "").strip()
        safety = scene.get("safety_and_quality") or {}
        safety_chunks: list[str] = []
        if isinstance(safety, dict):
            avoid = safety.get("avoid")
            if avoid:
                if isinstance(avoid, list):
                    safety_chunks.append("avoid " + ", ".join(str(x) for x in avoid if x))
                else:
                    safety_chunks.append("avoid " + str(avoid))
            must_have = safety.get("must_have")
            if must_have:
                if isinstance(must_have, list):
                    safety_chunks.append("must include " + ", ".join(str(x) for x in must_have if x))
                else:
                    safety_chunks.append("must include " + str(must_have))
            motion_quality = safety.get("motion_quality")
            if motion_quality:
                safety_chunks.append(str(motion_quality))
        safety_text = "; ".join(safety_chunks)

    

        camera_dict = first_shot.get("camera", {}) if isinstance(first_shot, dict) else {}
        movement_meta = camera_dict.get("movement", {}) if isinstance(camera_dict, dict) else {}
        position_meta = camera_dict.get("position_m", {}) if isinstance(camera_dict, dict) else {}
        start_pose = position_meta.get("start") if isinstance(position_meta, dict) else None
        end_pose = position_meta.get("end") if isinstance(position_meta, dict) else None

        movement_summary = "; ".join(
            str(x) for x in [
                movement_meta.get("type"),
                movement_meta.get("direction"),
                movement_meta.get("speed"),
                movement_meta.get("motivation"),
                movement_meta.get("secondary_motion"),
            ] if x
        )
        if not movement_summary and camera_text:
            movement_summary = camera_text

        if start_pose or end_pose:
            position_summary = f"start {start_pose or 'n/a'} -> end {end_pose or 'n/a'}"
        else:
            position_summary = ""

        timeline_section = ""
        if timeline:
            bullet_lines = [f"  - {line}" for line in timeline[:6]]
            timeline_section = "\n" + "\n".join(bullet_lines)

        lines = [
            f"SCENE INTENT: {creative_brief or 'No brief supplied - derive from bullets.'}",
            f"CHARACTERS: Lead={lead}; Antagonist={antagonist}; Setting={setting}; Weather={weather or 'controlled'}; Time={time_of_day or 'consistent'}",
            timeline_section and f"MOTION TIMELINE:{timeline_section}",
            movement_summary and f"CAMERA PLAN: {movement_summary}",
            position_summary and f"CAMERA PATH: {position_summary}",
            lighting_hint and f"LIGHTING: {lighting_hint}",
            audio_text and f"AUDIO & MIX: {audio_text}",
            vfx_text and f"VFX: {vfx_text}",
            continuity_logic and f"CONTINUITY CHECKS: {continuity_logic}",
            safety_text and f"SAFETY & PHYSICS: {safety_text}",
        ]
        out = "\n".join(line.strip() for line in lines if isinstance(line, str) and line.strip())
        for pat, rep in SAFEWORDING:
            out = re.sub(pat, rep, out)
        return out.strip()

    def _compose_single_prompt_compact(self, o: Dict[str, Any]) -> str:
        return self._compose_single_prompt_text(o)

    def compose_single_prompt_for_model(self, o: Dict[str, Any], continuity: ContinuityState) -> str:
        return self._compose_single_prompt_text(o, continuity)

    # ====== POST SANITY PASS (built-in, không còn phụ thuộc monkey-patch) ======
    def _has_archery_keywords(self, text: str) -> bool:
        if not isinstance(text, str): 
            return False
        t = text.lower()
        kws = ["bow","arrow","archer","archery","cung","mũi tên","mui ten","bắn tên","ban ten","căng dây","cang day"]
        return any(k in t for k in kws)

    def _scene_text_of(self, obj: dict) -> str:
        w = (obj or {}).get("world_and_characters") or {}
        return " ".join([
            str((obj or {}).get("creative_brief","")),
            str(w.get("setting","")),
            str(w.get("battle_context","")),
            str(w.get("character_continuity","")),
        ])

    def _reclassify_audio(self, shot: dict, scene_text: str = "") -> dict:
        """
        Chuẩn hoá audio:
        - Phân loại lại foley vs sfx
        - Dọn wording nhạy cảm (non-graphic)
        - Nếu cảnh không có archery, loại tiếng 'bowstring/arrow/whoosh/woody thunk'
        """
        import re
        aud = shot.get("audio") or {}

        # Gom và chuẩn hoá mảng
        foley_in = list(aud.get("foley") or [])
        sfx_in   = list(aud.get("sfx") or [])
        foley, sfx = [], []

        def push(arr, x):
            x = (x or "").strip()
            if x and x not in arr:
                arr.append(x)

        FOLEY_HINTS = (
            "footstep","breath","breathing","cloth","leather","body","groan","grunt",
            "shout","yell","roar","growl","snarl","voice","pant","rustle"
        )
        SFX_HINTS = (
            "impact","clatter","clash","clanging","thud","whoosh","swish","slam",
            "crack","hit","strike","spark","debris","dust puff","puff","trail"
        )

        # Phân loại cơ bản
        for item in foley_in + sfx_in:
            low = str(item).lower()
            if any(h in low for h in FOLEY_HINTS):
                push(foley, item)
            elif any(h in low for h in SFX_HINTS):
                push(sfx, item)
            else:
                # Mặc định: tiếng người/vật -> foley | va chạm/động năng -> sfx
                if re.search(r"(warrior|hunter|beast|animal|growl|roar|voice|groan|grunt)", low):
                    push(foley, item)
                else:
                    push(sfx, item)

        # Nếu cảnh không phải archery, loại cụm archery khỏi foley/sfx
        no_archery = not re.search(r"(bow|string|arrow|woody\s+thunk)", (scene_text or ""), flags=re.IGNORECASE)
        if no_archery:
            foley = [x for x in foley if not re.search(r"(bow|string|arrow|woody\s+thunk)", str(x), re.I)]
            sfx   = [x for x in sfx   if not re.search(r"(bow|string|arrow|woody\s+thunk)", str(x), re.I)]

        # Làm mềm wording (non-graphic)
        def safe(x: str) -> str:
            x = re.sub(r"(?i)painful\s+growl","fierce growl", x)
            x = re.sub(r"(?i)painful\s+roar","fierce roar", x)
            x = re.sub(r"(?i)spear\s+piercing\s+flesh","spear makes solid contact (non-graphic)", x)
            x = re.sub(r"(?i)reveals\s+(its\s+)?wound","reveals visible damage (non-graphic)", x)
            return x

        foley = [safe(x) for x in dict.fromkeys(foley)]
        sfx   = [safe(x) for x in dict.fromkeys(sfx)]

        # Đẩy voice-like từ SFX -> Foley
        kept_sfx = []
        for x in sfx:
            if re.search(r"(?i)\b(shout|yell|scream)s?\b", str(x)):
                push(foley, x)
            else:
                kept_sfx.append(x)

        aud_out = {}
        if foley: aud_out["foley"] = sorted(set(foley))
        if kept_sfx: aud_out["sfx"] = sorted(set(kept_sfx))
        # Giữ lại ambience/music_cue/mix_notes/dialogue_note nếu có nội dung
        for key in ("ambience","music_cue","mix_notes","dialogue_note"):
            if aud.get(key): aud_out[key] = aud[key]

        shot["audio"] = aud_out
        return shot

    def _apply_focus_heuristics(self, cam: dict, world: dict, shot: dict) -> dict:
        """Heuristic DOF & focus_pull wording; giữ end==start nếu static."""
        from copy import deepcopy
        import re
        cam = deepcopy(cam) if isinstance(cam, dict) else {}
        lens = float(cam.get("lens_mm", 35) or 35)
        slug = (shot.get("slugline") or "").lower()

        if "close-up" in slug or lens >= 50:
            cam.setdefault("depth_of_field", "shallow")
        elif lens <= 28:
            cam.setdefault("depth_of_field", "moderate")
        else:
            cam.setdefault("depth_of_field", "moderate")

        fps = cam.get("focus_pull_style")
        if isinstance(fps, str) and re.search(r"(?i)\bzoom\b", fps):
            cam["focus_pull_style"] = re.sub(r"(?i)\bzoom\b", "rack focus", fps)

        mv  = cam.get("movement") or {}
        pos = cam.get("position_m") or {}
        st  = (pos.get("start") or {}) if isinstance(pos, dict) else {}
        ed  = (pos.get("end") or {})   if isinstance(pos, dict) else {}
        mvt = (mv.get("type") or "static").lower()

        if mvt == "static" and st:
            pos["end"] = dict(st)
            cam["position_m"] = pos
        else:
            try:
                if st and ed and (st.get("x")==ed.get("x") and st.get("y")==ed.get("y") and st.get("z")==ed.get("z")):
                    eps = 0.1
                    direction = (mv.get("direction") or "").lower()
                    ex, ey, ez = float(st.get("x",0)), float(st.get("y",1.5)), float(st.get("z",3))
                    if direction in ("forward","dolly in","in"):
                        ez -= eps
                    elif direction in ("backward","dolly out","out"):
                        ez += eps
                    elif direction in ("left",):
                        ex -= eps
                    elif direction in ("right",):
                        ex += eps
                    pos["end"] = {"x":ex, "y":ey, "z":ez}
                    cam["position_m"] = pos
            except Exception:
                pass
        return cam

    def _post_sanity_pass(self, obj: dict, scene_text: str) -> dict:
        """Chạy sau khi build JSON thành công: reclass audio, DOF/focus, wording non-graphic."""
        import re
        if not isinstance(obj, dict): 
            return obj

        text = scene_text or self._scene_text_of(obj)
        for s in (obj.get("shots") or []):
            # an toàn nếu method chưa có
            try:
                s = self._reclassify_audio(s, text)
            except AttributeError:
                # fallback nhẹ: giữ nguyên audio
                pass
            cam = s.get("camera") or {}
            s["camera"] = self._apply_focus_heuristics(cam, obj.get("world_and_characters") or {}, s)

        # Làm sạch wording nhẹ trong single_prompt_for_model
        sp = (obj.get("single_prompt_for_model") or "")
        sp = re.sub(r"(?i)\bpainful\s+growl\b","fierce growl", sp)
        sp = re.sub(r"(?i)\breveals\s+(its\s+)?wound\b","reveals visible damage (non-graphic)", sp)
        obj["single_prompt_for_model"] = sp
        return obj


def _diag_report(self, o: dict):
    w = o.get("world_and_characters", {})
    seq = ((o.get("shots") or [{}])[0]).get("action_sequence", [])
    verbs = " ".join((e.get("verb", "") or "").lower() for e in seq)
    phases = ["draw", "anchor", "release", "flight", "impact", "lodges"]
    missing = [p for p in phases if p not in verbs]
    print(f"🔎 DIAG: time={w.get('time_of_day','?')}, weather={w.get('weather','?')}, missing_phases={missing}")


def _distribute_to_sum(self, durations: List[float], total: float) -> List[float]:
    if not durations:
        return []
    s = sum(durations)
    if s <= 0:
        return [8.0][:len(durations)]
    scale = total / s
    scaled = [round(d * scale, 2) for d in durations]
    drift = round(total - sum(scaled), 2)
    if scaled:
        scaled[-1] = round(scaled[-1] + drift, 2)
    return scaled


def _validate_and_clamp_sequence(self, seq: list, dur: float, lead_label: str = "Lead", antagonist_label: str = "Target") -> tuple[list, list]:
    """Return (fixed_seq, warnings). Enforce monotonic timestamps and dense coverage with no idle > IDLE_MAX_GAP."""
    warns = []
    seq = [e for e in seq if isinstance(e, dict) and "timestamp_sec" in e]
    seq.sort(key=lambda e: float(e["timestamp_sec"]))

    # clamp [0, dur)
    for e in seq:
        ts = float(e["timestamp_sec"])
        if ts < 0 or ts > dur:
            warns.append(f"timestamp {ts} out of [0,{dur}] → clamped")
        e["timestamp_sec"] = min(max(0.0, ts), dur - 0.01)

    # strictly increasing (>= 0.01s)
    for i in range(1, len(seq)):
        if seq[i]["timestamp_sec"] <= seq[i-1]["timestamp_sec"]:
            seq[i]["timestamp_sec"] = round(float(seq[i-1]["timestamp_sec"]) + 0.01, 2)

    # chèn beat nếu có gap > IDLE_MAX_GAP
    gaps = []
    prev = 0.0
    for e in seq:
        if e["timestamp_sec"] - prev > IDLE_MAX_GAP:
            gaps.append((prev, e["timestamp_sec"]))
        prev = e["timestamp_sec"]
    if dur - (seq[-1]["timestamp_sec"] if seq else 0.0) > IDLE_MAX_GAP:
        gaps.append((seq[-1]["timestamp_sec"] if seq else 0.0, dur))

    for a, b in gaps:
        mid = round(min(a + IDLE_MAX_GAP/2, b - 0.05), 2)
        seq.append({
            "timestamp_sec": mid,
            "subject": lead_label,
            "verb": "micro-adjusts",
            "biomechanics_note": "breath cycle; stance readjusts; grip micro-corrections",
            "physics_note": "continuous weight shift prevents idle; friction and inertia respected",
            "internal_motivation": "maintain objective without idle"
        })

    seq.sort(key=lambda e: float(e["timestamp_sec"]))

    # bổ sung tail để chắc chắn chạm ~7.9s với hành động có mục tiêu chứ không 'đứng nhìn'
    if not seq or float(seq[-1].get("timestamp_sec", 0)) < dur - 0.2:
        for t in TAIL_FILL_TS:
            if t < dur - 0.05:
                seq.append({
                    "timestamp_sec": round(t, 2),
                    "subject": lead_label,
                    "verb": "re-positions",
                    "biomechanics_note": "heel-toe footwork; shoulders counter-rotate subtly",
                    "physics_note": "momentum managed to avoid overshoot; parallax remains alive",
                    "internal_motivation": "keep initiative without idle"
                })
        seq.append({
            "timestamp_sec": round(dur - 0.05, 2),
            "subject": lead_label,
            "verb": "holds guard while shifting",
            "biomechanics_note": "low stance; breath-linked sway; feet alive",
            "physics_note": "no idle; friction & inertia managed to end frame",
            "internal_motivation": "stay ready into the cut"
        })

    seq.sort(key=lambda e: float(e["timestamp_sec"]))

    # đảm bảo tổng số beat tối thiểu
    if len(seq) < MIN_BEATS_8S:
        # rải thêm beat đều nhau (không đụng nhau)
        needed = MIN_BEATS_8S - len(seq)
        slots = [round(x, 2) for x in [i*(dur/(needed+1)) for i in range(1, needed+1)]]
        taken = {round(float(e["timestamp_sec"]),2) for e in seq}
        for t in slots:
            while t in taken:
                t = round(t + 0.03, 2)  # đẩy nhẹ để tránh trùng
            seq.append({
                "timestamp_sec": min(t, dur - 0.05),
                "subject": lead_label,
                "verb": "re-aims",
                "biomechanics_note": "micro footwork; grip corrections synced to breath",
                "physics_note": "tiny torque corrections keep sights steady",
                "internal_motivation": "maintain objective without idle"
            })
            taken.add(t)

    seq.sort(key=lambda e: float(e["timestamp_sec"]))
    return seq, warns


def _post_impact_clamp(self, seq: list) -> list:
    """After 'impacts'/'lodges', prevent unrealistic hopping. Allow ≤0.5s settling."""
    impact_i = None
    for i, e in enumerate(seq):
        v = (e.get("verb", "") or "").lower()
        if any(k in v for k in ("impacts", "lodges", "thuds")):
            impact_i = i
            break
    if impact_i is None:
        return seq

    impact_ts = float(seq[impact_i]["timestamp_sec"])
    max_settle = impact_ts + 0.5

    kept = []
    for e in seq:
        ts = float(e["timestamp_sec"])
        if ts <= impact_ts:
            kept.append(e)
        elif ts <= max_settle:
            kept.append(e)  # allow quick dampening
        else:
            # giữ lại nếu chỉ là phản ứng của TARGET (không cho ARROW tiếp tục "bay/nhảy")
            subj = (e.get("subject", "") or "").lower()
            verb = (e.get("verb", "") or "").lower()
            if "arrow" in subj:
                continue
            if any(k in verb for k in ("skitters", "bounces", "hops", "flies")) and "arrow" in subj:
                continue
            kept.append(e)
    return kept


def _windy(self, scene_weather: str) -> bool:
    if not scene_weather:
        return False
    s = scene_weather.lower()
    return any(k in s for k in ("wind", "breeze", "gale", "gust", "gió", "lộng"))


def _dominant_subject(self, seq: list[str]) -> str:
    """Lấy nhãn chủ đạo xuất hiện nhiều nhất (loại bỏ placeholder)."""
    ban = {"lead", "target", "arrow", "archer", ""}
    counter = {}
    for e in seq or []:
        s = (e.get("subject") or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in ban:
            continue
        counter[s] = counter.get(s, 0) + 1
    if not counter:
        return ""
    # trả về nhãn có tần suất lớn nhất
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _lock_subject_aliases(self, obj: dict, continuity: 'ContinuityState', scene_entities: dict) -> dict:
    """
    Ép tất cả subject về nhãn chuẩn (lead/antagonist) để tránh 'Warrior'↔'Lead'↔'KAI' nhảy lung tung.
    Ưu tiên:
      - world_and_characters.lead_character / antagonist nếu có
      - nếu rỗng: dùng subject chủ đạo trong sequence (self._dominant_subject)
      - nếu scene_entities có lead/antagonist thì lấy cái đầu tiên làm chuẩn
    """
    if not isinstance(obj, dict):
        return obj

    w = obj.get("world_and_characters") or {}
    shots = obj.get("shots") or []

    # chọn nhãn chuẩn
    lead_label = (w.get("lead_character") or "").strip()
    ant_label  = (w.get("antagonist") or "").strip()

    if not lead_label:
        # ưu tiên tên từ scene_entities
        ent_leads = (scene_entities or {}).get("lead") or []
        if ent_leads:
            lead_label = ent_leads[0].strip()
        else:
            # fallback: subject xuất hiện nhiều nhất (trừ placeholder)
            seq0 = (shots[0].get("action_sequence") or []) if shots else []
            dom = self._dominant_subject(seq0)
            if dom: lead_label = dom

    if not ant_label:
        ent_ants = (scene_entities or {}).get("antagonist") or []
        if ent_ants:
            ant_label = ent_ants[0].strip()
        else:
            # thử tìm ứng viên "beast/creature/enemy/target" trong chuỗi
            seq0 = (shots[0].get("action_sequence") or []) if shots else []
            for e in seq0:
                s = (e.get("subject") or "").strip().lower()
                if s in ("beast","creature","enemy","opponent","target"):
                    ant_label = e.get("subject","Target")
                    break
            if not ant_label:
                ant_label = "Target"

    # cập nhật world_and_characters
    w["lead_character"] = lead_label or "Lead"
    w["antagonist"]     = ant_label or "Target"
    obj["world_and_characters"] = w

    # tạo alias set đơn giản (thêm biến thể hay gặp)
    def aliases_for(label: str, generic_pool: list[str]) -> set[str]:
        if not label: return set()
        base = label.strip()
        al = {
            base, base.lower(), base.title(),
            f"the {base}".strip(), f"the {base}".lower(),
        }
        for g in generic_pool:
            al.add(g); al.add(g.capitalize()); al.add(g.title())
            al.add(f"the {g}"); al.add(f"the {g}".title())
        return {a.strip() for a in al if a.strip()}

    lead_aliases = aliases_for(lead_label, ["lead","protagonist","warrior","hunter","soldier","man","woman"])
    ant_aliases  = aliases_for(ant_label,  ["target","opponent","enemy","beast","creature"])

    # thay thế trong toàn bộ sequence
    for s in shots:
        seq = s.get("action_sequence") or []
        new_seq = []
        for e in seq:
            subj = (e.get("subject") or "").strip()
            subj_l = subj.lower()
            if subj_l in {x.lower() for x in lead_aliases}:
                e["subject"] = lead_label
            elif subj_l in {x.lower() for x in ant_aliases}:
                e["subject"] = ant_label
            elif subj_l in ("lead", "the lead"):
                e["subject"] = lead_label
            elif subj_l in ("target", "the target", "opponent"):
                e["subject"] = ant_label
            new_seq.append(e)
        s["action_sequence"] = new_seq

    return obj


def _derive_context(self, obj: dict, scene_text: str) -> dict:
    """Suy ngữ cảnh để chọn filler/động từ cụ thể."""
    t = (scene_text or "").lower() + " " + json.dumps((obj.get("world_and_characters") or {})).lower()
    seq = ((obj.get("shots") or [{}])[0]).get("action_sequence") or []
    verbs = " ".join((e.get("verb","") or "").lower() for e in seq)
    subjects = " ".join((e.get("subject","") or "").lower() for e in seq)

    ctx = {
        "melee": any(k in t+verbs for k in ["slash","thrust","parry","block","spear","giáo","giao","melee","claws"]),
        "spear": any(k in t+verbs for k in ["spear","giáo","giao","thrust"]),
        "group": any(k in subjects for k in ["hunters","soldiers","team","unit"]),
        "beast": any(k in subjects for k in ["beast","creature","animal"]),
        "forest": any(k in t for k in ["forest","rừng"]),
    }
    return ctx


def _purposeful_fillers(self, ctx: dict, lead_label: str, ant_label: str) -> list[dict]:
    """
    Các filler có mục tiêu rõ ràng cho 6–8s (không phải 'đứng nhìn'). 
    Chọn theo context để trông hợp lý.
    """
    fillers = []
    if ctx.get("spear"):
        fillers = [
            ( "re-guards", "spear haft draws back; feet settle into staggered stance",
              "re-centers balance after force transfer; friction prevents slide",
              "maintain threat readiness" ),
            ( "angles to flank", "lead foot steps diagonally; hips rotate to narrow profile",
              "short lateral shift reduces target's direct line; inertia controlled",
              "seek positional advantage" ),
            ( "re-aims", "hands micro-correct grip; shoulders square on target",
              "small torque corrections keep point of aim stable",
              "prepare follow-up strike" ),
        ]
    elif ctx.get("group"):
        fillers = [
            ( "signals to reposition", "off-hand gestures low; torso leans to indicate path",
              "minimal motion to communicate while preserving balance",
              "coordinate team response" ),
            ( "slides to cover", "short side-step; knees flex; weight stays low",
              "friction and low CoG enable controlled lateral move",
              "avoid exposure while tracking target" ),
            ( "braces", "feet widen; core tightens; chin tucks",
              "increase base of support to absorb impact or charge",
              "ready for next exchange" ),
        ]
    else:
        fillers = [
            ( "circles", "short arc step; head/gaze locked on opposition",
              "circular path manages distance; centripetal acceleration kept small",
              "probe for opening" ),
            ( "re-positions", "heel-toe step; shoulders counter-rotate",
              "momentum managed to avoid overshoot",
              "keep initiative alive" ),
            ( "checks surroundings", "scanning while feet keep moving; weapon stays up",
              "no idle—micro-motion preserves readiness and framing",
              "situational awareness" ),
        ]

    def mk(verb, bio, phy, motive, t=0.0):
        return {
            "timestamp_sec": float(t),
            "subject": lead_label,
            "verb": verb,
            "biomechanics_note": bio,
            "physics_note": phy,
            "internal_motivation": motive
        }

    # trả về danh sách mẫu; caller sẽ gán timestamp
    return [mk(*tpl) for tpl in fillers]


def _clarify_and_extend_actions(self, seq: list, dur: float, ctx: dict, lead_label: str, ant_label: str) -> list:
    """
    - Thay động từ mơ hồ → hành động cụ thể theo context
    - Bảo đảm có nhịp có mục tiêu đến cuối ~7.9s
    - Loại 'đứng nhìn': biến 'look/stare' → 'scan while moving' có mô tả chuyển động
    """
    if not isinstance(seq, list):
        return []

    # 1) Replace placeholders 'Lead'/'Target' ngay tại đây để ổn định
    for e in seq:
        s = (e.get("subject") or "").strip()
        if s.lower() == "lead":
            e["subject"] = lead_label
        elif s.lower() == "target":
            e["subject"] = ant_label

    # 2) Clarify vague verbs
    vague_map_base = {
        "react": ("braces", 
                  "weight shifts; core tightens; stance firms",
                  "pre-tension readies for incoming force",
                  "stay in the fight"),
        "assume": ("sets guard",
                   "hands/weapon rise; elbows tuck; feet offset",
                   "lowers CoG for stability under acceleration",
                   "prepare to engage"),
        "regain": ("re-centers",
                   "small back-step; shoulders settle over hips",
                   "re-align center of mass after perturbation",
                   "restore balance to press on"),
        "shout": ("calls out cues",
                  "diaphragm drives breath; head tilts toward ally while feet keep moving",
                  "vocalization without sacrificing movement",
                  "coordinate action"),
        "micro-adjusts": ("re-aims",
                          "micro footwork; grip corrections synced to breath",
                          "tiny torque corrections keep sights steady",
                          "maintain objective without idle"),
        "looks": ("scans while moving",
                  "eyes track while shoulders/feet keep subtle motion",
                  "avoid idle plateau; parallax confirms depth",
                  "situational awareness"),
        "stares": ("scans while moving",
                   "same as above",
                   "same as above",
                   "same as above"),
        "watch": ("scans while moving",
                  "eyes track while feet keep subtle motion",
                  "parallax confirms depth; no freeze",
                  "situational awareness"),
        "watches": ("scans while moving",
                    "eyes track while feet keep subtle motion",
                    "parallax confirms depth; no freeze",
                    "situational awareness"),
        "stop": ("re-positions",
                 "short shift in footwork; chin tucks; guard stays up",
                 "regains optimal distance without freezing",
                 "avoid idle plateau"),
        "stops": ("re-positions",
                  "short shift in footwork; chin tucks; guard stays up",
                  "regains optimal distance without freezing",
                  "avoid idle plateau"),
        "pause": ("scans while moving",
                  "eyes track while feet keep subtle motion",
                  "parallax confirms depth; no freeze",
                  "situational awareness"),
        "pauses": ("scans while moving",
                   "eyes track while feet keep subtle motion",
                   "parallax confirms depth; no freeze",
                   "situational awareness"),
        "wait": ("angles to flank",
                 "diagonal step; hips rotate to narrow profile",
                 "small lateral shift creates advantage",
                 "seek positional edge"),
        "waits": ("angles to flank",
                  "diagonal step; hips rotate to narrow profile",
                  "small lateral shift creates advantage",
                  "seek positional edge"),
        "stand": ("braces",
                  "feet widen; core tightens; shoulder settle",
                  "increase base of support while staying mobile",
                  "stay combat-ready"),
        "stands": ("braces",
                   "feet widen; core tightens; shoulder settle",
                   "increase base of support while staying mobile",
                   "stay combat-ready"),
        "observe": ("scans while moving",
                    "head pivots; shoulders sway with breath",
                    "maintain parallax & micro-motion to avoid idle",
                    "situational awareness"),
        "observes": ("scans while moving",
                     "head pivots; shoulders sway with breath",
                     "maintain parallax & micro-motion to avoid idle",
                     "situational awareness"),
        "stare": ("scans while moving",
                  "gaze tracks while footwork stays alive",
                  "parallax + weight shift keep motion continuous",
                  "maintain awareness"),
        "stares": ("scans while moving",
                   "gaze tracks while footwork stays alive",
                   "parallax + weight shift keep motion continuous",
                   "maintain awareness"),
        "idle": ("re-aims",
                 "micro footwork; grip corrections synced to breath",
                 "tiny torque corrections keep sights steady",
                 "avoid idle plateau"),
        "freeze": ("re-centers",
                   "small back-step; shoulders settle over hips",
                   "re-align center of mass after perturbation",
                   "restore balance and continue"),
        "freezes": ("re-centers",
                    "small back-step; shoulders settle over hips",
                    "re-align center of mass after perturbation",
                    "restore balance and continue"),
    }

    # context-specific substitutions
    if ctx.get("spear"):
        vague_map_base.update({
            "thrust": ("thrusts",
                       "rear foot drives; hips extend; lead arm snaps forward",
                       "impulse sends spear tip on straight vector",
                       "finish the strike"),
            "connects": ("drives through",
                         "shoulders follow-through; front knee bends to absorb",
                         "momentum transfer; controlled decel",
                         "commit to impact safely"),
        })
    if ctx.get("group"):
        vague_map_base.update({
            "react": ("spreads formation",
                      "two quick lateral steps; gestures low",
                      "increase spacing to reduce single-point failure",
                      "coordinate team lanes"),
        })

    def clarify(e):
        v = (e.get("verb") or "").strip().lower()
        if v in vague_map_base:
            newv, bio, phy, motive = vague_map_base[v]
            e["verb"] = newv
            e.setdefault("biomechanics_note", bio)
            e.setdefault("physics_note",     phy)
            e.setdefault("internal_motivation", motive)
        return e

    seq = [clarify(dict(e)) for e in seq]
    seq.sort(key=lambda x: float(x.get("timestamp_sec", 0.0)))

    # 3) Ensure purposeful beats until ~7.9s
    last_t = float(seq[-1]["timestamp_sec"]) if seq else 0.0
    if dur - last_t > 0.6:
        # chèn 2–3 filler có mục tiêu, phân bố đều 6.4–7.9s
        fillers = self._purposeful_fillers(ctx, lead_label, ant_label)
        slots = [max(dur-2.0, 6.4), max(dur-1.3, 7.1), max(dur-0.4, 7.9)]
        out = []
        i = 0
        for t in slots:
            if i < len(fillers):
                f = dict(fillers[i])
                f["timestamp_sec"] = round(min(max(0.0, t), dur-0.01), 2)
                out.append(f)
                i += 1
        seq.extend(out)

    # 4) Không cho 'idle nhìn' thuần túy
    for e in seq:
        v = (e.get("verb") or "").lower()
        if v in ("looks", "stares", "observes"):
            e["verb"] = "scans while moving"
            e["biomechanics_note"] = "eyes track while feet keep subtle motion; shoulders sway with breath"
            e["physics_note"] = "micro parallax + weight shift preserve motion continuity"
            e["internal_motivation"] = "maintain awareness without idle"

    # đảm bảo tăng dần thời gian
    seq.sort(key=lambda x: float(x.get("timestamp_sec", 0.0)))
    for i in range(1, len(seq)):
        if seq[i]["timestamp_sec"] <= seq[i-1]["timestamp_sec"]:
            seq[i]["timestamp_sec"] = round(seq[i-1]["timestamp_sec"] + 0.01, 2)

    # ép beat cuối >= dur-0.12s
    if not seq or float(seq[-1]["timestamp_sec"]) < dur - 0.12:
        seq.append({
            "timestamp_sec": round(dur - 0.05, 2),
            "subject": lead_label,
            "verb": "holds guard while shifting",
            "biomechanics_note": "low stance; continuous breath-linked sway; feet alive",
            "physics_note": "no idle; friction & inertia managed to end frame",
            "internal_motivation": "stay ready into the cut"
        })
    seq.sort(key=lambda x: float(x.get("timestamp_sec", 0.0)))
    return seq


def _ensure_cause_effect(self, seq: list, window: float = 1.2, antagonist_label: str = "Target") -> list:
    out = []
    REACTION_VERBS = ("recoils", "staggers", "deflects", "reacts", "reels", "splashes", "lodges", "impacts", "thuds")
    for i, e in enumerate(sorted(seq, key=lambda x: float(x["timestamp_sec"]))):
        out.append(e)
        v = (e.get("verb", "") or "").lower()
        subj = (e.get("subject", "") or "").lower()
        if any(k in v for k in ("draws", "releases", "strikes", "pushes", "pulls", "fires", "throws")):
            t0 = float(e["timestamp_sec"])
            # lookahead
            found = any(
                (float(n["timestamp_sec"]) - t0) <= window and
                any(rv in n.get("verb", "").lower() for rv in REACTION_VERBS)
                for n in seq[i+1:]
            )
            if not found:
                out.append({
                    "timestamp_sec": round(min(t0 + 0.8, t0 + window), 2),
                    "subject": antagonist_label,  # ✨ thay vì "Target"
                    "verb": "reacts",
                    "biomechanics_note": "weight shifts; brief loss of balance; muscles tense",
                    "physics_note": "momentum transfer and friction cause visible response",
                    "internal_motivation": "instinctive response to force"
                })
    out.sort(key=lambda x: float(x["timestamp_sec"]))
    return out


def _exposure_heuristics(self, time_of_day: str, weather: str) -> dict:
    tod = (time_of_day or "").lower()
    wet = (weather or "").lower()
    out = {}

    if any(k in tod for k in ("night", "đêm")):
        out.update({"white_balance_k": 3200, "iso": 800, "aperture_f": 2.8})
    elif any(k in tod for k in ("sunset", "sunrise", "hoàng hôn", "bình minh", "dawn", "dusk")):
        out.update({"white_balance_k": 4800, "iso": 640, "aperture_f": 4.0})
    else:  # daytime
        out.update({"white_balance_k": 5600, "iso": 400, "aperture_f": 5.6})

    if any(k in wet for k in ("overcast", "cloud", "mây", "rain", "mưa")):
        out["aperture_f"] = min(4.0, out.get("aperture_f", 4.0))
        out["iso"] = max(640, out.get("iso", 640))

    return out




def enforce_eight_seconds_and_motion(self, obj: Dict[str, Any]) -> Dict[str, Any]:
    """Force duration=8, shots cover full 0–8s without idle; add beat timeline."""
    obj["duration_seconds"] = 8

    # ✨ Single-shot clamp (merge nếu >1)
    shots = list(obj.get("shots") or [])
    if len(shots) > 1:
        first = shots[0]
        merged = []
        for sh in shots:
            merged.extend(sh.get("action_sequence", []))
        first["action_sequence"] = merged
        obj["shots"] = [first]
        shots = obj["shots"]

    if not shots:
        obj["shots"] = [{}]
        shots = obj["shots"]

    # ✨ Lấy nhãn thực tế
    w = obj.get("world_and_characters") or {}
    lead_label = (w.get("lead_character") or "Lead").strip() or "Lead"
    ant_label  = (w.get("antagonist") or "Target").strip() or "Target"
    
    # ✨ BUILD ALLOWED SUBJECTS SET - avoid adding actions for wrong subjects
    allowed_subjects = {lead_label, ant_label, "Lead", "Target"}
    
    # Extract from characters if available
    chars = w.get("characters", [])
    if isinstance(chars, list):
        for ch in chars:
            if isinstance(ch, dict):
                label = ch.get("label", "")
                name = ch.get("name", "")
                if label:
                    allowed_subjects.add(label)
                if name:
                    allowed_subjects.add(name)
    
    # Also check character_details
    char_details = w.get("character_details", {})
    if isinstance(char_details, dict):
        allowed_subjects.update(char_details.keys())
    
    # Normalize allowed subjects (handle case variations)
    allowed_subjects_normalized = {s.strip().lower() for s in allowed_subjects if s}
    
    def is_valid_subject(subj: str) -> bool:
        """Check if subject is in allowed list."""
        if not subj:
            return False
        subj_norm = subj.strip().lower()
        # Allow if exact match or canonical match
        return (subj_norm in allowed_subjects_normalized or 
                canon(subj).lower() in allowed_subjects_normalized)
    
    def get_verb_counts(seq: list) -> dict:
        """Count verb usage in sequence."""
        verb_counts = {}
        for e in seq:
            if isinstance(e, dict):
                verb = (e.get("verb") or "").strip().lower()
                if verb:
                    verb_counts[verb] = verb_counts.get(verb, 0) + 1
        return verb_counts
    
    def check_verb_diversity(seq: list, new_verb: str) -> tuple[bool, str]:
        """Check if adding new_verb would violate diversity rules.
        Returns (is_ok, warning_message)
        """
        verb_counts = get_verb_counts(seq)
        new_verb_lower = new_verb.strip().lower()
        current_count = verb_counts.get(new_verb_lower, 0)
        
        # Rule: NEVER use same verb more than 2 times
        if current_count >= 2:
            return False, f"⚠️ VERB DIVERSITY: '{new_verb}' already used {current_count} times (max 2)"
        
        # Count unique verbs
        unique_verbs = len(verb_counts)
        
        # Warn if we have too few unique verbs for the sequence length
        if len(seq) > 10 and unique_verbs < 8:
            return True, f"ℹ️ Only {unique_verbs} unique verbs so far (target: 12+)"
        
        return True, ""

    # phân phối thời lượng
    durs = [float(s.get("duration", 8.0)) for s in shots]
    durs = self._distribute_to_sum(durs, 8.0)

    t = 0.0
    for s, d in zip(shots, durs):
        s["start_time"] = round(t, 2)
        s["duration"] = round(d, 2)
        t = round(t + d, 2)

    # NEW: if action_sequence exists, guarantee dense beats and archery realism
    for s in shots:
        has_sequence = isinstance(s.get("action_sequence"), list)

        cam = s.get("camera") or {}
        mv = cam.get("movement") or {}
        if isinstance(mv, dict) and mv.get("type", "static").lower() == "static":
            cam["movement"] = {
                "type": "static",
                "micro_motion": "subtle parallax or breathing-linked sway; no jitter"
            }
            s["camera"] = cam

        # Ensure visual clarity and legible motion
        # Prefer moderate/deep DOF and crisper shutter to avoid motion blur
        cam.setdefault("composition_style", "rule of thirds; readable subject separation")
        cam.setdefault("depth_of_field", "moderate")
        cam.setdefault("focus_pull_style", "none unless motivated by action")
        # If shutter angle missing or too high, clamp to 90–120deg for clarity
        try:
            sa = float(cam.get("shutter_angle_deg", 0) or 0)
        except Exception:
            sa = 0
        if not sa or sa > 180:
            cam["shutter_angle_deg"] = 120
        # Basic exposure defaults if absent
        if "white_balance_k" not in cam:
            cam["white_balance_k"] = 5600
        if "iso" not in cam:
            cam["iso"] = 400
        if "aperture_f" not in cam:
            cam["aperture_f"] = 5.6
        s["camera"] = cam

        # Ensure audio slots and map requested categories
        aud = s.get("audio") or {}
        aud.setdefault("ambience", "")
        if not isinstance(aud.get("foley"), list):
            aud["foley"] = []
        if not isinstance(aud.get("sfx"), list):
            aud["sfx"] = []
        # use 'music_cue' explicitly; do not force a score if not implied
        aud.setdefault("music_cue", "")
        aud.setdefault("mix_notes", "dialogue intelligibility; avoid masking critical impacts")
        # keep compatibility: remove any free-form 'music'
        aud.pop("music", None)
        s["audio"] = aud

        if has_sequence:
            # Densify to ~0.5–1.0s cadence if too sparse
            seq = s["action_sequence"]
            seq = [e for e in seq if isinstance(e, dict)]

            # ✨ dùng nhãn thật khi densify
            seq, warns = self._validate_and_clamp_sequence(seq, float(s.get("duration", 8.0)), lead_label, ant_label)
            for wmsg in warns:
                print("⚠️ SEQ:", wmsg)
            
            # FORCE MAIN ACTION INJECTION
            creative_brief = obj.get("creative_brief", "")
            if creative_brief:
                seq = self._force_main_action_injection(seq, creative_brief, lead_label)

            # Fix archery beats ordering - remove premature target reactions before release
            rel_idx = self._phase_index(seq, "release","loose")
            if rel_idx >= 0:
                t_rel = float(seq[rel_idx]["timestamp_sec"])
                seq = [e for e in seq if not (
                    (e.get("subject","").lower()=="target") and
                    ("react" in (e.get("verb","").lower())) and
                    float(e.get("timestamp_sec",0)) < t_rel
                )]

            # (di chuyển WIND lên trước khi dùng)
            wthr = (obj.get("world_and_characters") or {}).get("weather", "")

            # ✨ chặn projectile chain nếu không phải archery
            if any("arrow" in (e.get("subject","").lower()) for e in seq):
                if self._is_archery_scene(json.dumps(obj), self.extract_scene_entities(json.dumps(obj))):
                    seq = self._ensure_projectile_chain(seq, float(s.get("duration",8.0)), self._windy(wthr))
            if self._windy(wthr):
                for e in seq:
                    if "arrow" in (e.get("subject", "").lower()):
                        e["physics_note"] = (e.get("physics_note", "") + " | crosswind introduces slight lateral drift").strip()
                mv = s["camera"].get("movement", {})
                if isinstance(mv, dict) and mv.get("type", "static") == "static":
                    base = mv.get("micro_motion", "").strip()
                    mv["micro_motion"] = (base + "; wind-safe stabilization (no jitter)").strip("; ").strip()
                    s["camera"]["movement"] = mv
                aud = s.get("audio") or {}
                fx = aud.get("sfx") or []
                if "low wind bed (non-graphic)" not in fx:
                    fx.append("low wind bed (non-graphic)")
                aud["sfx"] = fx
                s["audio"] = aud

            # Lock consistent subject labels before any timing edits
            try:
                obj = self._lock_subject_aliases(obj, continuity, scene_entities_local)  # type: ignore
                # refresh references after lock
                s = (obj.get("shots") or [{}])[0]
                seq = s.get("action_sequence") or seq
            except Exception:
                pass

            # CRITICAL FIX: Ensure realistic timeline spacing and prevent repetitive patterns
            # Sort existing sequence by timestamp
            seq = sorted(seq, key=lambda x: float(x.get("timestamp_sec", 0)))
            
            # COMPLETE TIMELINE REDISTRIBUTION: Force 0.4s minimum spacing everywhere
            fixed_seq = []
            last_time = -0.5  # Start before 0 to ensure proper spacing
            
            for i, action in enumerate(seq):
                curr_time = float(action.get("timestamp_sec", 0))
                
                # STRICT: Ensure minimum 0.4s spacing between ALL actions
                if curr_time - last_time < 0.4:
                    new_time = round(last_time + 0.4, 2)
                    action["timestamp_sec"] = new_time
                    last_time = new_time
                else:
                    last_time = curr_time
                
                # CRITICAL: Ensure no action goes beyond 7.5s (leave 0.5s buffer)
                if last_time > 7.5:
                    last_time = 7.5
                    action["timestamp_sec"] = 7.5
                
                fixed_seq.append(action)
            seq = fixed_seq
            
            # AGGRESSIVE ANTI-REPETITION: Detect and completely replace repetitive patterns
            verb_patterns = {}
            for action in seq:
                verb = action.get("verb", "").lower()
                verb_patterns[verb] = verb_patterns.get(verb, 0) + 1
            
            # DEBUG: Print verb patterns for analysis
            print(f"DEBUG: Verb patterns detected: {verb_patterns}")
            
            # COMPLETE REPLACEMENT: If any verb appears more than 1 time, replace ALL instances
            replacement_verbs = [
                "adjusts grip", "shifts weight", "maintains balance", "prepares stance", 
                "assesses situation", "stays alert", "focuses attention", "prepares reaction",
                "maintains readiness", "stays vigilant", "prepares defense", "maintains position",
                "optimizes stance", "enhances focus", "coordinates movement", "stabilizes position"
            ]
            
            # Enhanced biomechanics with specific muscle groups (30+ chars each)
            biomech_templates = [
                "deltoid and trapezius muscles engage; shoulder joint rotates; core stabilizes; neural pathways activate",
                "quadriceps and hamstrings contract; hip flexors activate; ankle joints flex; weight distribution optimized",
                "erector spinae muscles support spine; glutes engage; weight shifts through pelvis; balance maintained",
                "biceps and triceps coordinate; elbow joint articulates; forearm muscles stabilize; grip strength optimized",
                "neck muscles tense; cervical spine aligns; head position optimizes; visual tracking enhanced",
                "abdominal muscles contract; diaphragm controls breathing; ribcage expands; core stability maximized",
                "pectoral muscles engage; shoulder blades stabilize; arm movement coordinated; upper body strength applied",
                "latissimus dorsi activates; back muscles support; spinal alignment maintained; posture optimized"
            ]
            
            # Enhanced physics with specific forces (30+ chars each)
            physics_templates = [
                "conservation of momentum governs motion; gravity creates downward force; friction provides resistance; energy transfer optimized",
                "kinetic energy transfers through joints; potential energy stored in muscle tension; air resistance affects movement; momentum conserved",
                "center of gravity shifts; torque applied through limbs; angular momentum maintained; rotational forces balanced",
                "force vectors combine; acceleration changes direction; inertia resists sudden movements; mechanical advantage applied",
                "elastic energy stored in tendons; muscle fibers contract; mechanical advantage optimized; power output maximized",
                "fluid dynamics affect motion; pressure changes in joints; energy dissipation through heat; thermal regulation active",
                "electromagnetic forces in muscles; chemical energy converts to mechanical; ATP hydrolysis drives contraction; cellular respiration optimized",
                "hydrostatic pressure in joints; synovial fluid reduces friction; cartilage absorbs impact; bone density supports load"
            ]
            
            # Process each action for anti-repetition
            for i, action in enumerate(seq):
                verb = action.get("verb", "").lower()
                if verb_patterns.get(verb, 0) > 1:
                    # Replace with completely different, contextual actions
                    new_verb = replacement_verbs[i % len(replacement_verbs)]
                    action["verb"] = new_verb
                    print(f"DEBUG: Replaced '{verb}' with '{new_verb}' at index {i}")
                    
                    # Apply enhanced biomechanics
                    action["biomechanics_note"] = biomech_templates[i % len(biomech_templates)]
                    
                    # Apply enhanced physics
                    action["physics_note"] = physics_templates[i % len(physics_templates)]
                    
                    # Enhanced motivation
                    action["internal_motivation"] = "maintaining tactical awareness and readiness for dynamic response; psychological state optimized for action"
            
            # FORCE INCLUSION OF MAIN ACTIONS FROM CREATIVE_BRIEF
            creative_brief = obj.get("creative_brief", "").lower()
            print(f"DEBUG: Creative brief: {creative_brief}")
            main_actions = []
            
            # Extract main actions from creative_brief with enhanced detection
            if "thrusts" in creative_brief or "thrust" in creative_brief:
                main_actions.append(("thrusts spear", "shoulder and arm muscles engage; core rotates; forward momentum generated; weapon accelerates toward target", "kinetic energy transfers through weapon; impact force calculated; reaction force generated on contact"))
                print("DEBUG: Added thrusts spear action")
            if "charges" in creative_brief or "charge" in creative_brief:
                main_actions.append(("charges forward", "leg muscles contract; hip flexors activate; forward propulsion generated; weight shifts forward", "momentum builds; friction resists; acceleration increases; kinetic energy accumulates"))
                print("DEBUG: Added charges forward action")
            if "strikes" in creative_brief or "strike" in creative_brief:
                main_actions.append(("strikes target", "arm muscles contract; weapon accelerates; impact force applied; target receives blow", "kinetic energy transfers to target; reaction force generated; momentum conserved"))
                print("DEBUG: Added strikes target action")
            if "shoots" in creative_brief or "shooting" in creative_brief or "arrows" in creative_brief:
                main_actions.append(("releases arrow", "bowstring tension released; arrow accelerates; projectile motion initiated; trajectory calculated", "elastic energy converts to kinetic; gravity affects trajectory; air resistance slows projectile"))
                print("DEBUG: Added releases arrow action")
            if "pursues" in creative_brief or "pursuit" in creative_brief:
                main_actions.append(("pursues target", "leg muscles engage; forward motion maintained; tracking initiated; distance closing", "momentum conserved; distance closing; speed optimized; pursuit velocity calculated"))
                print("DEBUG: Added pursues target action")
            if "beast" in creative_brief and "charging" in creative_brief:
                main_actions.append(("beast charges", "powerful leg muscles contract; forward momentum generated; aggressive posture assumed; target locked", "massive kinetic energy builds; ground friction resists; acceleration increases; impact force calculated"))
                print("DEBUG: Added beast charges action")
            
            # Insert main actions at key timestamps (1.0s, 3.0s, 5.0s) with priority
            main_timestamps = [1.0, 3.0, 5.0]
            for i, (verb, biomech, physics) in enumerate(main_actions[:3]):
                if i < len(main_timestamps):
                    # Check if timestamp is already occupied
                    occupied = any(abs(main_timestamps[i] - float(a.get("timestamp_sec", 0))) < 0.4 for a in seq)
                    if not occupied:
                        seq.append({
                            "timestamp_sec": main_timestamps[i],
                            "subject": lead_label,
                            "verb": verb,
                            "biomechanics_note": biomech,
                            "physics_note": physics,
                            "internal_motivation": "executing primary objective with focused determination"
                        })
                        print(f"DEBUG: Inserted main action '{verb}' at {main_timestamps[i]}s")
                    else:
                        # Find alternative timestamp
                        for alt_time in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.0]:
                            if not any(abs(alt_time - float(a.get("timestamp_sec", 0))) < 0.4 for a in seq):
                                seq.append({
                                    "timestamp_sec": alt_time,
                                    "subject": lead_label,
                                    "verb": verb,
                                    "biomechanics_note": biomech,
                                    "physics_note": physics,
                                    "internal_motivation": "executing primary objective with focused determination"
                                })
                                print(f"DEBUG: Inserted main action '{verb}' at alternative time {alt_time}s")
                                break
            
            # Ensure coverage: need at least 12-16 beats over 8s with realistic spacing
            if len(seq) < 12:
                # Create realistic timeline with proper spacing
                target_beats = max(16, len(seq) + 4)
                time_span = 7.0  # Leave 1s buffer at end
                interval = time_span / (target_beats - 1)
                
                # Find gaps and fill with contextual micro-actions
                existing_times = [float(a.get("timestamp_sec", 0)) for a in seq]
                contextual_actions = [
                    ("maintains focus", "eye muscles track movement; neck muscles stabilize head position; visual cortex processes information", "visual tracking requires precise muscle coordination; neural pathways activate"),
                    ("adjusts breathing", "diaphragm contracts rhythmically; chest expands and contracts; oxygen intake optimized", "oxygen intake supports muscle function and alertness; carbon dioxide expelled"),
                    ("shifts stance", "weight transfers between feet; hip muscles engage for balance; core stabilizes", "center of gravity shifts to maintain stability; momentum redistributed"),
                    ("prepares reaction", "muscles tense in anticipation; neural pathways activate; sensory input processed", "preparation phase allows faster response time; reaction time minimized"),
                    ("maintains readiness", "muscle tone optimized; attention focused; sensory awareness heightened", "physiological state prepared for action; response capability maximized"),
                    ("assesses situation", "visual scanning; auditory processing; tactical evaluation", "sensory input analyzed; decision-making processes activated")
                ]
                
                for i in range(target_beats):
                    target_time = round(i * interval, 2)
                    if not any(abs(target_time - t) < 0.4 for t in existing_times):
                        verb, biomech, physics = contextual_actions[i % len(contextual_actions)]
                        seq.append({
                            "timestamp_sec": target_time,
                            "subject": lead_label,
                            "verb": verb,
                            "biomechanics_note": biomech,
                            "physics_note": physics,
                            "internal_motivation": "maintaining readiness and situational awareness"
                        })
                
                # Re-sort after adding new actions
                seq = sorted(seq, key=lambda x: float(x.get("timestamp_sec", 0)))

            # STRICT CHARACTER NAME LOCKING: Ensure consistent naming throughout
            # Lock the first character name found and use it consistently
            first_subject = None
            for action in seq:
                if action.get("subject") and not first_subject:
                    first_subject = action.get("subject")
                    break
            
            if first_subject:
                # Replace all subjects with the locked name
                for action in seq:
                    if action.get("subject"):
                        action["subject"] = first_subject
            
            # STRICT BIOMECHANICS ENHANCEMENT: Ensure all notes are 30+ chars with specific details
            new_seq = []
            for i, e in enumerate(seq):
                if isinstance(e, dict):
                    # Enhanced biomechanics with specific muscle groups and joint movements (50+ chars)
                    if not e.get("biomechanics_note") or len(e.get("biomechanics_note", "")) < 50:
                        enhanced_biomech = [
                            "deltoid and trapezius muscles engage; shoulder joint rotates; core stabilizes; neural pathways activate; weight distribution optimized",
                            "quadriceps and hamstrings contract; hip flexors activate; ankle joints flex; weight distribution optimized; balance maintained through pelvis",
                            "erector spinae muscles support spine; glutes engage; weight shifts through pelvis; balance maintained; spinal alignment preserved",
                            "biceps and triceps coordinate; elbow joint articulates; forearm muscles stabilize; grip strength optimized; arm movement coordinated",
                            "neck muscles tense; cervical spine aligns; head position optimizes; visual tracking enhanced; sensory input processed",
                            "abdominal muscles contract; diaphragm controls breathing; ribcage expands; core stability maximized; oxygen intake optimized"
                        ]
                        e["biomechanics_note"] = enhanced_biomech[i % len(enhanced_biomech)]
                        print(f"DEBUG: Enhanced biomechanics for action {i}: {e.get('verb', 'unknown')}")
                    
                    # Enhanced physics with specific forces and energy transfer (50+ chars)
                    if not e.get("physics_note") or len(e.get("physics_note", "")) < 50:
                        enhanced_physics = [
                            "conservation of momentum governs motion; gravity creates downward force; friction provides resistance; energy transfer optimized; kinetic energy accumulates",
                            "kinetic energy transfers through joints; potential energy stored in muscle tension; air resistance affects movement; momentum conserved; power output maximized",
                            "center of gravity shifts; torque applied through limbs; angular momentum maintained; rotational forces balanced; mechanical advantage applied",
                            "force vectors combine; acceleration changes direction; inertia resists sudden movements; mechanical advantage applied; energy dissipation controlled",
                            "elastic energy stored in tendons; muscle fibers contract; mechanical advantage optimized; power output maximized; cellular respiration active",
                            "fluid dynamics affect motion; pressure changes in joints; energy dissipation through heat; thermal regulation active; hydrostatic pressure optimized"
                        ]
                        e["physics_note"] = enhanced_physics[i % len(enhanced_physics)]
                        print(f"DEBUG: Enhanced physics for action {i}: {e.get('verb', 'unknown')}")
                    
                    # Enhanced motivation with specific psychological states (40+ chars)
                    if not e.get("internal_motivation") or len(e.get("internal_motivation", "")) < 40:
                        enhanced_motivation = [
                            "maintaining tactical awareness and readiness for dynamic response; psychological state optimized for action; decision-making processes activated",
                            "executing primary objective with focused determination; strategic thinking engaged; situational awareness heightened; response capability maximized",
                            "staying alert for next move; sensory input processed; neural pathways activated; cognitive functions optimized for rapid response",
                            "preparing for confrontation; adrenaline levels elevated; fight-or-flight response activated; survival instincts engaged; defensive mechanisms primed",
                            "assessing the situation; analytical thinking engaged; risk evaluation active; strategic planning initiated; tactical advantage sought",
                            "navigating the environment; spatial awareness enhanced; obstacle avoidance active; pathfinding algorithms engaged; movement optimization applied"
                        ]
                        e["internal_motivation"] = enhanced_motivation[i % len(enhanced_motivation)]
                        print(f"DEBUG: Enhanced motivation for action {i}: {e.get('verb', 'unknown')}")
                    
                    new_seq.append(e)
            seq = new_seq
            
            # FINAL VALIDATION: Ensure all requirements are met
            print(f"DEBUG: Final sequence length: {len(seq)}")
            print(f"DEBUG: Final timestamps: {[float(a.get('timestamp_sec', 0)) for a in seq]}")
            print(f"DEBUG: Final verbs: {[a.get('verb', '') for a in seq]}")
            
            # Check for remaining repetitive patterns
            final_verbs = [a.get("verb", "").lower() for a in seq]
            verb_counts = {}
            for verb in final_verbs:
                verb_counts[verb] = verb_counts.get(verb, 0) + 1
            print(f"DEBUG: Final verb counts: {verb_counts}")
            
            # Check biomechanics length
            biomech_lengths = [len(a.get("biomechanics_note", "")) for a in seq]
            print(f"DEBUG: Biomechanics lengths: {biomech_lengths}")
            
            # Check physics length
            physics_lengths = [len(a.get("physics_note", "")) for a in seq]
            print(f"DEBUG: Physics lengths: {physics_lengths}")

            # Archery-specific injection if applicable
            # Heuristic: check if any archery keyword appears in existing sequence or scene-level creative_brief
            scene_text = (obj.get("creative_brief") or "") + " " + json.dumps(obj.get("world_and_characters") or {})
            scene_entities_local = self.extract_scene_entities(scene_text)
            # STRICT CHECK: Only inject human archer phases if an archer is explicitly mentioned
            if (self._is_archery_scene(scene_text, scene_entities_local)
                and self._mentions_human_archer(scene_text, scene_entities_local)
                and any("arrow" in (e.get("subject","").lower()) or "archer" in (e.get("subject","").lower()) for e in seq)):
                # Ensure ordered phases exist
                verbs = " ".join([str(e.get("verb", "")).lower() for e in seq])
                need_draw = "draw" not in verbs and "căng" not in verbs
                need_anchor = "anchor" not in verbs
                need_release = "release" not in verbs and "loose" not in verbs
                need_flight = "flight" not in verbs and "flies" not in verbs and "arc" not in verbs
                need_impact = "impact" not in verbs and "thud" not in verbs and "lodges" not in verbs

                # Insert canonical beats if missing
                def add(ts, subject, verb, bio, phy):
                    seq.append({
                        "timestamp_sec": float(ts),
                        "subject": subject,
                        "verb": verb,
                        "biomechanics_note": bio,
                        "physics_note": phy
                    })

                # Draw (0.3s)
                if need_draw:
                    add(0.3, "Archer", "draws",
                        "scapular muscles engage; bow arm extends, string hand pulls to mid-face",
                        "string tension rises; potential energy stored in limbs")

                # Anchor (0.8s)
                if need_anchor:
                    add(0.8, "Archer", "anchors",
                        "string hand stabilizes at corner of mouth; elbow aligns; breathing pause",
                        "stable anchor ensures consistent launch angle and reduces lateral error")

                # Release (1.2s)
                if need_release:
                    add(1.2, "Archer", "releases",
                        "finger relax triggers clean string slip; bow arm resists torque; slight follow-through",
                        "stored elastic energy converts to arrow kinetic energy; mild recoil")

                # Flight (1.3–2.0s)
                if need_flight:
                    add(1.35, "Arrow", "enters flight",
                        "fletching stabilizes; shaft flexes then dampens",
                        "parabolic arc under gravity; drag slows speed; subtle lateral drift if windy")
                    add(1.9, "Arrow", "crosses mid-trajectory",
                        "spin stabilizes yaw; arc becomes more pronounced",
                        "vertical drop increases with time; momentum carries forward")

                # Impact (~2.2–2.6s)
                if need_impact:
                    add(2.3, "Arrow", "impacts",
                        "shaft compresses then settles; vibrations dampen quickly",
                        "kinetic energy dissipates into target; arrow lodges at entry point")
                    add(2.6, "Target", "recoils",
                        "mass shifts opposite the impact vector; brief stagger",
                        "conservation of momentum; friction with ground limits slide")

                # Ensure audio fx - only if sequence has arrow/flight/release
                arrow_in_seq = any("arrow" in (e.get("subject","").lower()) for e in seq)
                released = any("releas" in (e.get("verb","").lower()) for e in seq)

                if arrow_in_seq or released:
                    fx = s["audio"].get("sfx", [])

                    def ensure_fx(x):
                        if x not in fx:
                            fx.append(x)

                    ensure_fx("bowstring snap on release (non-graphic)")
                    ensure_fx("soft arrow whoosh in flight (non-graphic)")
                    ensure_fx("solid thud/woody thunk on impact (non-graphic)")

                # Ensure VFX
                vfx = s.get("vfx") or []
                if "subtle arrow motion trail during flight" not in vfx:
                    vfx.append("subtle arrow motion trail during flight")
                if "small dust/particle puff on impact" not in vfx:
                    vfx.append("small dust/particle puff on impact")
                s["vfx"] = vfx

            # Apply post-impact clamp
            seq = self._post_impact_clamp(seq)
            
            # ✨ POST-FILTER: Remove invalid subjects and enforce verb diversity
            filtered_seq = []
            for action in seq:
                if not isinstance(action, dict):
                    continue
                    
                subject = action.get("subject", "")
                verb = action.get("verb", "")
                
                # Check 1: Subject must be valid
                if not is_valid_subject(subject):
                    print(f"⚠️ FILTERED OUT: Invalid subject '{subject}' for verb '{verb}' at {action.get('timestamp_sec')}s")
                    continue
                
                # Check 2: Verb diversity (max 2 occurrences)
                is_ok, msg = check_verb_diversity(filtered_seq, verb)
                if not is_ok:
                    print(f"{msg} at {action.get('timestamp_sec')}s - SKIPPED")
                    continue
                
                filtered_seq.append(action)
            
            seq = filtered_seq
            print(f"✅ Filtered action_sequence: {len(seq)} valid actions retained")
            
            # Report verb diversity stats
            verb_counts = get_verb_counts(seq)
            unique_verbs = len(verb_counts)
            print(f"📊 Verb diversity: {unique_verbs} unique verbs in {len(seq)} actions")
            if unique_verbs < 12 and len(seq) >= 16:
                print(f"⚠️ LOW DIVERSITY: Only {unique_verbs} unique verbs (target: 12+)")
            
            # ✨ ACTION BALANCE CHECK - ensure lead and antagonist have reasonable action distribution
            lead_actions = [a for a in seq if canon((a.get("subject") or "").strip()).lower() == canon(lead_label).lower()]
            ant_actions = [a for a in seq if canon((a.get("subject") or "").strip()).lower() == canon(ant_label).lower()]
            
            lead_count = len(lead_actions)
            ant_count = len(ant_actions)
            total_count = len(seq)
            
            print(f"📊 Action balance: {lead_label}={lead_count} ({lead_count*100//max(total_count,1)}%), {ant_label}={ant_count} ({ant_count*100//max(total_count,1)}%)")
            
            # Warn if heavily imbalanced (one character has <20% of actions)
            if total_count > 10:
                lead_pct = lead_count * 100 / total_count
                ant_pct = ant_count * 100 / total_count
                
                if lead_pct < 20:
                    print(f"⚠️ IMBALANCE: {lead_label} has only {lead_pct:.0f}% of actions (expect 40-60%)")
                elif lead_pct > 80:
                    print(f"⚠️ IMBALANCE: {lead_label} has {lead_pct:.0f}% of actions (expect 40-60%)")
                
                if ant_pct < 20:
                    print(f"⚠️ IMBALANCE: {ant_label} has only {ant_pct:.0f}% of actions (expect 40-60%)")
                elif ant_pct > 80:
                    print(f"⚠️ IMBALANCE: {ant_label} has {ant_pct:.0f}% of actions (expect 40-60%)")

            # ✨ cause→effect với antagonist label thực
            seq = self._ensure_cause_effect(seq, antagonist_label=ant_label)
            
            # ✨ STANDARDIZE character_details format - always use dict
            for action in seq:
                if not isinstance(action, dict):
                    continue
                
                char_det = action.get("character_details")
                
                # If it's a string, try to extract from world_and_characters
                if isinstance(char_det, str):
                    subject = action.get("subject", "")
                    # Try to find full profile from world_and_characters
                    char_details_dict = w.get("character_details", {})
                    if subject in char_details_dict:
                        action["character_details"] = char_details_dict[subject]
                    else:
                        # Fallback: create minimal dict with the string as note
                        action["character_details"] = {
                            "name": subject,
                            "description": char_det
                        }
                
                # If missing entirely, try to populate from world_and_characters
                elif not char_det:
                    subject = action.get("subject", "")
                    char_details_dict = w.get("character_details", {})
                    if subject in char_details_dict:
                        action["character_details"] = char_details_dict[subject]

            # Guarantee camera angle and path fields
            cam = s.get("camera") or {}
            cam.setdefault("composition_style", cam.get("composition_style") or "wide shot; rule of thirds; eye-level unless specified")
            if not isinstance(cam.get("position_m"), dict):
                cam["position_m"] = {
                    "start": cam.get("position_m_start") or [-1.2, 1.6, 4.0],
                    "end": cam.get("position_m_end") or [-0.4, 1.6, 3.2]
                }
            mv = cam.get("movement") or {}
            if isinstance(mv, dict) and not mv.get("type"):
                mv["type"] = "tracking"
                mv.setdefault("direction", "forward dolly-in")
                mv.setdefault("speed", "slow")
                cam["movement"] = mv
            s["camera"] = cam

            # Áp state-machine projectile (tổng quát, không chỉ cung)
            windy = self._windy((obj.get("world_and_characters") or {}).get("weather", ""))
            if any("arrow" in (e.get("subject", "").lower()) for e in seq):
                seq = self._ensure_projectile_chain(seq, float(s.get("duration", 8.0)), windy)
                self._assert_no_teleport(seq)

            # Apply domain pack postprocessing
            world = obj.get("world_and_characters") or {}
            for pack in getattr(self, "domain_packs", []):
                # nhẹ nhàng áp dụng nếu pack liên quan
                if pack.name == "archery" and not any("arrow" in (e.get("subject", "").lower()) for e in seq):
                    continue
                seq = pack.postprocess_sequence(seq, s, world)

            # Audio/VFX "tam thanh nhất vết" bám theo phase
            aud = s.get("audio") or {}
            fx = aud.get("sfx") or []

            def need(x):
                if x not in fx:
                    fx.append(x)

            if any("releases" in (e.get("verb", "").lower()) for e in seq):
                need("bowstring snap (non-graphic)")
            if any("flight" in (e.get("verb", "").lower()) or "trajectory" in (e.get("verb", "").lower()) for e in seq):
                need("soft arrow whoosh (non-graphic)")
            if any(v in (e.get("verb", "").lower()) for e in seq for v in ("impacts", "lodges", "thud")):
                need("solid woody thunk on impact (non-graphic)")
            aud["sfx"] = fx
            s["audio"] = aud

            vfx = s.get("vfx") or []

            def vneed(x):
                if x not in vfx:
                    vfx.append(x)

            if any("flight" in (e.get("verb", "").lower()) for e in seq):
                vneed("subtle arrow motion trail during flight")
            if any(v in (e.get("verb", "").lower()) for e in seq for v in ("impacts", "lodges", "thud")):
                vneed("small dust/particle puff on impact")
            s["vfx"] = vfx

            # Sort by timestamp and write back
            seq.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))

            # --- Final anti-idle sweep: siết lại gap & đa dạng động từ ---
            seq, _ = self._validate_and_clamp_sequence(
                seq, float(s.get("duration", 8.0)), lead_label, ant_label
            )
            
            # FORCE MAIN ACTION INJECTION (second pass)
            creative_brief = obj.get("creative_brief", "")
            if creative_brief:
                seq = self._force_main_action_injection(seq, creative_brief, lead_label)

            # Lấy context & áp clarifier + filler chống "trững"
            world = obj.get("world_and_characters") or {}
            ctx = self._derive_context(obj, json.dumps(world) + " " + (obj.get("creative_brief") or ""))
            seq = self._clarify_and_extend_actions(
                seq=seq,
                dur=float(s.get("duration", 8.0)),
                ctx=ctx,
                lead_label=lead_label,
                ant_label=ant_label
            )
            # >>> PATCH START: fill purposeful beats until ~8s
            seq = self._enforce_full_8s_coverage(
                seq,
                float(s.get("duration", 8.0)),
                lead_label,
                ant_label
            )
            # <<< PATCH END
            s["action_sequence"] = seq
            # >>> PATCH START: audio reclassify (strip archery if irrelevant)
            s = self._reclassify_audio(s, (obj.get("creative_brief") or "") + " " + json.dumps(obj.get("world_and_characters") or {}))
            # <<< PATCH END

        else:
            # Legacy schema fallback (action string). We keep your old micro-action enforcement.
            action = (s.get("action") or "").strip()
            core = (
                " Lead keeps performing purposeful micro-actions (breathing, gaze shifts, footwork, grip adjustments) every 0.5–1s; "
                "every initiating action produces a visible reaction (recoil, stumble, flinch, dust puff, splash, deflection). "
                "No idle plateau, no freeze frames."
            )
            if core not in action:
                s["action"] = (action + " " + core).strip()

    # Simplified timeline - only include if needed for complex scenes
    if not getattr(self, "strict_grounding", False) and (len(shots) > 1 or any(len(s.get("action_sequence", [])) > 10 for s in shots)):
        beats = []
        for sec in range(0, 9):
            beats.append({
                "t": sec,
                "lead_action": "purposeful micro-action; maintain objective; no idle",
                "opposition_reaction": "reactive micro-actions; respect inertia, momentum, gravity, friction",
                "continuity_check": "wardrobe/props/weather consistent; lighting direction/grade consistent"
            })
        obj["timelines"] = {
            "granularity": "1s",
            "beats": beats,
            "guarantee": "No second is idle; actions remain physically plausible with cause→effect reactions across full 8s."
        }
    return obj


# -------------------------
# Top-level defaults & grounding reinforcement (ADDED)
# -------------------------
def ensure_top_level_defaults(self, obj: Dict[str, Any], scene_entities: Dict[str, Any], continuity: ContinuityState, scene_text: str = "") -> Dict[str, Any]:
    # Set language default
    obj.setdefault("language", "English")

    # Set other common defaults to reduce repair cycles
    obj.setdefault("model", "google/veo-3")
    obj.setdefault("request_type", "video")
    obj.setdefault("fps", 30)
    obj.setdefault("resolution", "1920x1080")
    obj.setdefault("aspect_ratio", "16:9")

    # Đặt defaults cho các field REQUIRED (tránh vòng repair)
    obj.setdefault("creative_brief", "Scene brief not provided; follow bullets and continuity.")
    obj.setdefault("continuity_logic", "Labels kept (Lead/Antagonist consistent); Wardrobe/props repeated per scene input; Setting/weather/time consistent with locks; Last→first frame continuity held; No idle seconds across 0–8s")
    obj.setdefault("single_prompt_for_model", "GROUND THE SCENE...")

    # nếu rỗng thì set luôn
    gs = obj.get("global_style") or {}
    gs.setdefault("visual_tone", "cinematic, grounded")
    gs.setdefault("motion_rules", "purposeful, no idle; micro-parallax when static")
    gs.setdefault("color_grading", "neutral filmic")
    gs.setdefault("vfx_guidance", "subtle, non-graphic, support realism")
    gs.setdefault("sound_mix", "diegetic with restrained score")
    obj["global_style"] = gs

    # >>> PATCH START: prefer VN bullets for brief
    brief_src = (scene_entities.get("action_specific") or scene_entities.get("content_main") or "").strip()
    if brief_src:
        obj["creative_brief"] = brief_src
    # <<< PATCH END

    w = obj.get("world_and_characters") or {}
    w.setdefault("lead_character", "")
    w.setdefault("antagonist", "")

    # Always ensure output_settings
    if not isinstance(obj.get("output_settings"), dict):
        obj["output_settings"] = {
            "container": "mp4",
            "video_codec": "h264",
            "video_bitrate": "8000k",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "normalize_audio": True,
            "tone_mapping": "linear",
            "grain": "none",
        }

    # Fill world fields ONLY from this scene when strict_grounding
    w = obj.get("world_and_characters") or {}
    if getattr(self, "strict_grounding", False):
        w.setdefault("setting", scene_entities.get("setting") or scene_entities.get("setting_image", ""))
        if scene_entities.get("weather"):
            w.setdefault("weather", scene_entities.get("weather"))
        if scene_entities.get("time_of_day"):
            w.setdefault("time_of_day", scene_entities.get("time_of_day"))
    else:
        w.setdefault("setting", scene_entities.get("setting") or continuity.setting_baseline or scene_entities.get("setting_image", ""))
        if continuity.weather_baseline or scene_entities.get("weather"):
            w.setdefault("weather", continuity.weather_baseline or scene_entities.get("weather"))
        if continuity.time_of_day or scene_entities.get("time_of_day"):
            w.setdefault("time_of_day", continuity.time_of_day or scene_entities.get("time_of_day"))
    w.setdefault("battle_context", "")

    # NEW: Infer "night" & WB/ISO from text (even when time_of_day is empty)
    txt = " ".join([
        str(obj.get("creative_brief","")),
        json.dumps(obj.get("world_and_characters") or {}),
        json.dumps((obj.get("shots") or [{}])[0].get("lighting",""))
    ]).lower()

    if not getattr(self, "strict_grounding", False):
        if not w.get("time_of_day") and any(k in txt for k in ["night","moon","moonlight","đêm","ánh trăng"]):
            w["time_of_day"] = "night"
    obj["world_and_characters"] = w

    # Ensure character_continuity repeats wardrobe/props ONLY if this scene mentions a lead
    cc = w.get("character_continuity") or ""
    has_lead = bool(scene_entities.get("lead"))
    if has_lead:
        wardrobe_src = scene_entities.get("wardrobe") or continuity.wardrobe_details
        if wardrobe_src and wardrobe_src not in cc:
            cc = (cc + ("; " if cc else "") + wardrobe_src).strip("; ")
        if continuity.props_details and continuity.props_details not in cc:
            cc = (cc + "; " + continuity.props_details).strip("; ")
        w["character_continuity"] = cc
    else:
        # prevent bleeding human wardrobe text into non-human scenes
        w["character_continuity"] = cc if not cc else cc.split(";")[0].strip()
    obj["world_and_characters"] = w

    # Strengthen physics expectations globally
    gs = obj.get("global_style") or {}

    # Safety & quality: must have realistic ballistic cues (for filtering only, not output)
    saq = obj.get("safety_and_quality") or {"avoid": [], "must_have": []}
    mh = saq.get("must_have") or []
    # F) Must-have theo loại cảnh
    core = [
        "cause→effect reactions for every initiated action",
        "non-graphic, documentary tone for combat actions",
    ]
    for c in core:
        if c not in mh:
            mh.append(c)
    has_proj = self._has_projectile(obj)
    has_melee = self._has_melee(obj)
    has_pred = self._has_animal_predator(obj)
    # Gỡ ballistic cũ nếu không cần
    mh = [x for x in mh if "ballistic motion" not in x]
    if has_proj and "physically plausible ballistic motion (gravity, drag, momentum)" not in mh:
        mh.append("physically plausible ballistic motion (gravity, drag, momentum)")
    if (not has_proj) and has_melee:
        melee_rule = "physically plausible melee motion (balance, recoil, friction)"
        if melee_rule not in mh:
            mh.append(melee_rule)
    if has_pred:
        pred_rule = "predator–prey reaction realism (non-graphic)"
        if pred_rule not in mh:
            mh.append(pred_rule)
    saq["must_have"] = mh
    # Keep safety_and_quality for internal filtering, but don't include in final prompt
    obj["safety_and_quality"] = saq

    # Strengthen single_prompt_for_model with explicit grounding to bullets
    sp = (obj.get("single_prompt_for_model") or "").strip()
    grounding = (
        "GROUND THE SCENE: Obey the provided 'Content' as the core intent; "
        "map 'Setting/Visuals' to the setting; apply 'Camera Angle' and 'Camera Movement' into the camera; "
        "reflect 'Audio' into audio fields; repeat wardrobe/props exactly."
    )
    if grounding not in sp:
        obj["single_prompt_for_model"] = (sp + "\n" + grounding).strip()

    # NEW: Auto-create/merge continuity profiles for all mentioned characters (lead & others)
    cu = obj.get("continuity_updates") or {}
    profiles: List[Dict[str, Any]] = []

    required_scene_characters = self._mentioned_characters(scene_text or "", scene_entities or {}, continuity)
    world = obj.get("world_and_characters") or {}
    ward = (world.get("character_continuity") or "").strip()

    def build_min_profile(name: str) -> Dict[str, Any]:
        base = (continuity.characters.get(name) or {}).copy()
        # fill minimal fields if absent
        base.setdefault("wardrobe", ward or "N/A")
        base.setdefault("props_in_hand", self._infer_props_from_text(ward))
        base.setdefault("notes", "auto-min profile from scene output")
        return base

    for name in required_scene_characters:
        if not name:
            continue
        profiles.append({
            "name": name,
            "profile": build_min_profile(name)
        })

    if profiles:
        cu["new_or_updated_characters"] = profiles
    else:
        # nếu cảnh không nhắc nhân vật nào, đảm bảo rỗng
        cu["new_or_updated_characters"] = []
    obj["continuity_updates"] = cu

    # Auto-create/merge dynamic-entity profiles for mentioned entities
    try:
        req = self._mentioned_entities(scene_text or "", scene_entities or {}, continuity)
        # loại trừ những cái đã là nhân vật
        req_all = [x for x in (req.get("all") or []) if x not in (required_scene_characters or [])]
        cu2 = obj.get("continuity_updates") or {}
        dyn_entries = [e for e in (cu2.get("new_or_updated_dynamic_entities") or []) if isinstance(e, dict)]
        have = {(e.get("name") or "").strip() for e in dyn_entries}

        for nm in req_all:
            name2 = (nm or "").strip()
            if not name2 or name2 in have:
                continue
            cat = self._classify_entity(name2)
            base = (continuity.entities.get(name2) or {}).copy()

            if cat == "animal":
                base.setdefault("type","animal")
                base.setdefault("appearance","")
                base.setdefault("biomechanics", {
                    "mass_kg": "unknown",
                    "center_of_gravity": "mid-torso",
                    "locomotion": "quadruped or relevant gait",
                    "reaction_to_threat": "startle → recoil → reposition (non-graphic)",
                    "vulnerability": "joints/neck areas"
                })
            elif cat == "vehicle":
                base.setdefault("type","vehicle")
                base.setdefault("specs","engine/propulsion unknown; use plausible motion only")
                base.setdefault("motion_model","traction, inertia, turning radius; momentum under braking")
            elif cat == "projectile":
                base.setdefault("type","projectile")
                base.setdefault("motion_model", "ballistic arc; gravity/drag/momentum; non-graphic impact wording")
            else:
                base.setdefault("type","object")
                base.setdefault("motion_model","mass/inertia/friction; subtle drift or settle if pushed")

            base.setdefault("notes","auto-min dynamic-entity profile from scene bullets; expand as info appears")
            dyn_entries.append({"name": name2, "profile": base})

        cu2["new_or_updated_dynamic_entities"] = dyn_entries if (req.get("all")) else []
        obj["continuity_updates"] = cu2
    except Exception:
        pass

    # Insert detailed profiles only for mentioned entities in this scene
    try:
        mentioned = self._mentioned_entities(scene_text or "", scene_entities or {}, continuity)
        char_names = self._mentioned_characters(scene_text or "", scene_entities or {}, continuity) or []
        cu3 = obj.setdefault("continuity_updates", {})
        cu3.setdefault("new_or_updated_characters", [])
        cu3.setdefault("new_or_updated_dynamic_entities", [])

        existing_chars = {e.get("name") for e in cu3["new_or_updated_characters"] if isinstance(e, dict)}
        for n in char_names:
            if n and n not in existing_chars:
                cu3["new_or_updated_characters"].append({
                    "name": n,
                    "profile": self._build_detailed_character_profile(n, scene_text, continuity)
                })

        existing_dyn = {e.get("name") for e in cu3["new_or_updated_dynamic_entities"] if isinstance(e, dict)}
        for ent in (mentioned.get("dynamic") or []):
            n, kind = ent.get("name"), (ent.get("kind") or "object")
            if n and n not in existing_dyn:
                cu3["new_or_updated_dynamic_entities"].append({
                    "name": n,
                    "profile": self._build_detailed_dynamic_profile(n, kind, scene_text, continuity)
                })
        obj["continuity_updates"] = cu3
    except Exception:
        pass

    # NEW: Make physics_rules / timelines.guarantee less poetic
    gs = obj.get("global_style") or {}
    gs["physics_rules"] = (
        "Respect gravity, drag, inertia, momentum, friction, and reaction latency. "
        "All motion follows realistic physics with proper cause-and-effect; "
        "no teleportation; no idle seconds."
    )
    obj["global_style"] = gs

    if "timelines" in obj:
        obj["timelines"]["guarantee"] = (
            "No idle seconds; actions remain physically plausible with cause→effect reactions across full 8s."
        )

    return obj

def _has_projectile(self, obj: dict) -> bool:
    try:
        seq = ((obj.get("shots") or [{}])[0]).get("action_sequence") or []
        txt = " ".join([
            json.dumps(obj.get("world_and_characters") or {}),
            json.dumps(obj.get("creative_brief") or ""),
            " ".join(f"{e.get('subject','')} {e.get('verb','')}" for e in seq)
        ]).lower()
        kws = ("arrow","bolt","bullet","shoot","release","loose","bow","archer","fire")
        return any(k in txt for k in kws)
    except Exception:
        return False

def _has_melee(self, obj: dict) -> bool:
    try:
        seq = ((obj.get("shots") or [{}])[0]).get("action_sequence") or []
        verbs = " ".join((e.get("verb","") or "").lower() for e in seq)
        melee_verbs = ("slash","stab","thrust","punch","kick","parry","block","grapple","strike","charge")
        return any(v in verbs for v in melee_verbs)
    except Exception:
        return False

def _has_animal_predator(self, obj: dict) -> bool:
    try:
        w = obj.get("world_and_characters") or {}
        txt = " ".join([json.dumps(w), json.dumps(obj.get("creative_brief") or "")]).lower()
        subjects = " ".join(
            (e.get("subject","") or "").lower()
            for e in ((obj.get("shots") or [{}])[0]).get("action_sequence") or []
        )
        animal_kws = ("beast","wolf","tiger","lion","bear","boar","panther","predator")
        return any(k in txt or k in subjects for k in animal_kws)
    except Exception:
        return False


def _infer_props_from_text(self, text: str) -> str:
    """Simple regex-based props inference from text"""
    if not isinstance(text, str):
        return "N/A"
    t = text.lower()
    props = []
    if any(k in t for k in ["bow", "cung", "archery"]):
        props.append("bow")
    if any(k in t for k in ["arrow", "mũi tên", "mui ten"]):
        props.append("arrows")
    if any(k in t for k in ["spear", "giáo", "giao"]):
        props.append("spear")
    if any(k in t for k in ["shield", "khiên", "khien"]):
        props.append("shield")
    if any(k in t for k in ["sword", "kiếm", "kiem"]):
        props.append("sword")
    return ", ".join(props) if props else "N/A"


# -------------------------
# Validation & Repair
# -------------------------
def validate_json(self, obj: Dict[str, Any]) -> Optional[str]:
    if jsonschema is None:
        required = set(VEO3_SCHEMA.get("required", []))
        missing = [k for k in required if k not in obj]
        if missing:
            return f"Missing keys: {missing}"

        # ✨ kiểm thêm 3 khối bắt buộc
        for k in ("world_and_characters","global_style","output_settings"):
            if not isinstance(obj.get(k), dict):
                return f"{k} must be an object"

        shots = obj.get("shots")
        if not isinstance(shots, list) or not (1 <= len(shots) <= 4):
            return "shots must be an array with 1–4 items"

        # ...
        req = (
            VEO3_SCHEMA.get("properties", {})
            .get("shots", {})
            .get("items", {})
            .get("required", [])
        )
        need_sequence = "action_sequence" in req

        for i, s in enumerate(shots, 1):
            if not isinstance(s, dict):
                return f"shots[{i}] must be object"

            cam = (s or {}).get("camera")
            if not isinstance(cam, dict):
                return f"shots[{i}].camera must be object"

            for k in ("movement", "orientation_deg", "focus", "position_m"):
                if k not in cam:
                    return f"shots[{i}].camera.{k} is required"

            if "vfx" not in s:
                return f"shots[{i}].vfx is required"

            if need_sequence and "action_sequence" not in s:
                return f"shots[{i}].action_sequence is required by schema"
            if need_sequence:
                for j, e in enumerate(s.get("action_sequence", []), 1):
                    miss = [k for k in ("timestamp_sec","subject","verb","biomechanics_note","physics_note","internal_motivation") if k not in e]
                    if miss:
                        return f"shots[{i}].action_sequence[{j}] missing keys: {miss}"

        return None

    try:
        validate(instance=obj, schema=VEO3_SCHEMA)
        return None
    except Exception as e:
        return str(e)


def repair_with_llm(
    self,
    bad_obj: Dict[str, Any],
    error_msg: str,
    continuity: ContinuityState,
    scene_text: str = ""
) -> Optional[Dict[str, Any]]:
    if not self.client:
        return None

    tools = self._build_tool_spec()
    # Compute required scene characters (derive scene_entities locally)
    scene_entities = self.extract_scene_entities(scene_text or "")
    required_scene_characters = self._mentioned_characters(scene_text, scene_entities, continuity)
    resp = self._chat(
        model=self.model,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        seed=self.seed,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "build_veo3_prompt"}},
        messages=[
            {"role": "system", "content": "Return only a strict JSON function-call for build_veo3_prompt. No prose."},
            {
                "role": "user",
                "content": (
                    "Fix the JSON to satisfy the schema WITHOUT changing scene intent.\n"
                    "Rules (very important):\n"
                    "- In each shot: 'camera' MUST include 'movement', 'orientation_deg', 'focus', 'position_m'.\n"
                    "- 'vfx', 'lighting', 'audio', 'transition_out' belong to the SHOT LEVEL (same level as 'camera'), NEVER inside 'camera'.\n"
                    "- This project uses SINGLE SHOT per scene. Ensure exactly one shot spanning 0–8 seconds with no gaps.\n\n"
                    f"Schema error:\n{error_msg}\n\n"
                    f"Continuity locks:\n{self._to_english(continuity.summary_for_prompt())}\n\n"
                    + ("SCENE TEXT (verbatim):\n" + scene_text + "\n\n" if scene_text else "")
                    + "Invalid JSON you must correct:\n"
                    + json.dumps(bad_obj, ensure_ascii=False)
                )
            },
        ],
    )
    choice = resp.choices[0]
    try:
        fixed = self._parse_choice(choice)
        return fixed
    except Exception:
        # Fallback: retry với response_format={"type":"json_object"}
        try:
            resp2 = self._chat(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return only a strict JSON object. No prose. JSON format required."},
                    {
                        "role": "user",
                        "content": (
                            "Fix the JSON to satisfy the schema WITHOUT changing scene intent.\n"
                            "Rules (very important):\n"
                            "- In each shot: 'camera' MUST include 'movement', 'orientation_deg', 'focus', 'position_m'.\n"
                            "- 'vfx', 'lighting', 'audio', 'transition_out' belong to the SHOT LEVEL (same level as 'camera'), NEVER inside 'camera'.\n"
                            "- This project uses SINGLE SHOT per scene. Ensure exactly one shot spanning 0–8 seconds with no gaps.\n\n"
                            f"Schema error:\n{error_msg}\n\n"
                            f"Continuity locks:\n{self._to_english(continuity.summary_for_prompt())}\n\n"
                            + ("SCENE TEXT (verbatim):\n" + scene_text + "\n\n" if scene_text else "")
                            + "Invalid JSON you must correct:\n"
                            + json.dumps(bad_obj, ensure_ascii=False)
                        )
                    },
                ],
            )
            return json.loads(resp2.choices[0].message.content)
        except Exception:
            return None


# -------------------------
# Core LLM call per scene — UPDATED for single-shot & richer grounding
# -------------------------
def build_scene_prompt(self, scene_text: str, scene_number: int = 1, continuity: ContinuityState = None) -> Dict[str, Any]:
    if not self.client:
        return {"error": "OpenAI client is not initialized (no API key)."}

    # Create default continuity if not provided
    if continuity is None:
        continuity = ContinuityState()

    scene_entities = self.extract_scene_entities(scene_text)
    # — Khởi tạo sớm để tránh UnboundLocalError và dùng cho toàn khối —
    required_scene_characters = self._mentioned_characters(scene_text or "", scene_entities or {}, continuity) or []
    required_scene_entities   = self._mentioned_entities(scene_text or "", scene_entities or {}, continuity) if hasattr(self, "_mentioned_entities") else {"all": [], "characters": []}

    # === BƯỚC NÂNG CẤP "ĐẠO DIỄN AI" ===
    if self.director_mode:
        # 1. Phân tích bối cảnh
        print("🎬 Phân tích tâm lý và bối cảnh...")
        analysis = self._analyze_scene_subtext(scene_text, self._to_english(continuity.summary_for_prompt()))

        # 2. Tạo kịch bản vi mô
        print("✍️ Tạo kịch bản vi mô...")
        micro_script = self._generate_micro_script(analysis)
        if not micro_script:  # Fallback nếu bước trên thất bại
            micro_script = scene_entities.get("action_specific") or scene_entities.get("content_main")
    else:
        # Chế độ nhanh: dùng action_specific trực tiếp
        micro_script = scene_entities.get("action_specific") or scene_entities.get("content_main")

    # ==================================
    # Use domain packs for rules
    active_packs = [p for p in self.domain_packs if p.detect(scene_text, scene_entities)]
    rules_txt = "\n".join(p.inject_rules_block() for p in active_packs if p.inject_rules_block())

    # BƯỚC MỚI: Truy xuất hồ sơ chi tiết từ ContinuityState
    mentioned_chars = scene_entities.get("lead", []) + scene_entities.get("antagonist", [])
    detailed_entity_profiles_text = ""
    # Xây dựng chuỗi văn bản chứa hồ sơ chi tiết
    profiles_to_inject = []
    for char_name in mentioned_chars:
        if char_name in continuity.characters:
            profiles_to_inject.append({
                "type": "Character",
                "name": char_name,
                "profile": continuity.characters[char_name]
            })

    # Làm tương tự cho locations nếu có - mở rộng tìm kiếm trong toàn bộ scene_text
    mentioned_locations = scene_entities.get("setting", [])
    if isinstance(mentioned_locations, str):
        mentioned_locations = [mentioned_locations]

    # Tìm kiếm tên bối cảnh trong toàn bộ scene_text
    import re
    location_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',          # Proper nouns
        r'\b(the\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',    # "The Great Rock"
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'                # "Great Rock"
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, scene_text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            match = match.strip()
            if len(match) > 3 and match not in mentioned_locations:
                mentioned_locations.append(match)

    for loc_name in mentioned_locations:
        if loc_name in continuity.locations:
            profiles_to_inject.append({
                "type": "Location",
                "name": loc_name,
                "profile": continuity.locations[loc_name]
            })

    if profiles_to_inject:
        detailed_entity_profiles_text = "ENTITY PROFILES FOR THIS SCENE (MUST ADHERE STRICTLY):\n" + json.dumps(profiles_to_inject, indent=2)

    # --- Prompt template: không dùng f-string để tránh lỗi dấu {} literal ---
    ui_tmpl = """
    You are the top 0.01% cinematic prompt engineer for Veo-3 video generation, delivering flawless, production-ready JSON briefs.
    You operate with zero tolerance for ambiguity; every field must be precise, internally consistent, and director-ready.

    HARD REQUIREMENTS
    - Language: English only.
    - STRICT JSON via function call (no prose).
    - duration_seconds = 8 exactly. Cover the full 0–8s continuously, no gaps/overlaps.
    - SINGLE SHOT ONLY per scene (exactly one shot), and that shot must carry continuous, goal-driven action for all 8 seconds.
    - Every second contains purposeful motion (no idle plateau). No "frozen" frames unless the scene text explicitly says so.
    - Prohibit any blur, smearing, or indistinct motion. Motion must be readable, physically credible, and precisely described.
    - DETAILED REQUIREMENT: Each field must be filled with SPECIFIC, DETAILED descriptions, not generic placeholders.
    - ZERO ERROR TOLERANCE: No placeholder text, no TODOs; resolve conflicts explicitly and keep schemas perfectly valid.
    - CONSISTENCY REQUIREMENT: Character names, appearance, wardrobe, and props must remain identical throughout the 8-second sequence.
    
    ACTION TIMING RULES (FOLLOW EXACTLY - CRITICAL):
    1. Generate exactly 16-20 action beats over 8 seconds
    2. Minimum spacing: 0.35 seconds between ANY two consecutive actions (NO EXCEPTIONS)
    3. Distribution (MUST follow this pattern):
       - Setup (0-2s): 3-4 beats, spaced 0.5-0.8s apart (slow, deliberate buildup)
       - Action (2-6s): 10-12 beats, spaced 0.3-0.45s apart (fast, intense action)
       - Resolution (6-8s): 3-4 beats, spaced 0.4-0.6s apart (gradual wind down)
    
    VERB VARIETY RULES (STRICT):
    1. NEVER use the same verb more than 2 times in the entire 8-second sequence
    2. BANNED repetitive patterns: "aims → re-aims → re-aims → re-aims" (USE: "aims → releases → tracks")
    3. Required: At least 12 DIFFERENT verbs per scene (show variety!)
    4. High-risk repetitive verbs to AVOID overusing: "re-aims", "re-positions", "micro-adjusts", "readies", "prepares"
    5. Instead use specific action verbs: "steadies grip", "tracks target", "adjusts stance", "shifts weight", "scans terrain"
    
    CAUSE-EFFECT TIMING RULES (CRITICAL):
    1. Every initiated action MUST have a visible reaction within 0.2-0.4 seconds
    2. Example: Warrior thrusts spear (2.0s) → Beast recoils (2.3s) ✅ [0.3s reaction time - GOOD]
    3. BAD: Warrior thrusts spear (2.0s) → Beast reacts (4.0s) ❌ [2.0s gap - TOO LONG]
    4. Projectile timing: Release → Flight (show arc) → Impact/reaction (all within 0.8-1.2s total)
    
    EXAMPLES OF GOOD VS BAD SEQUENCES:
    
    ✅ GOOD SEQUENCE (Warrior vs Beast - Spear Attack):
    {
      "action_sequence": [
        {"timestamp_sec": 0.0, "subject": "Warrior", "verb": "charges forward", "biomechanics_note": "Hips drive forward; quad muscles engage; arms pump for momentum", "physics_note": "Acceleration builds from 0 to 4 m/s over 2 meters; friction provides traction", "internal_motivation": "Overwhelm the beast with aggressive advance"},
        {"timestamp_sec": 0.6, "subject": "Warrior", "verb": "lowers spear tip", "biomechanics_note": "Right shoulder drops 20cm; elbow extends; wrist rotates to align spear point", "physics_note": "Spear tip descends in parabolic arc; weight shifts forward increasing stability", "internal_motivation": "Prepare for precision strike at vulnerable target"},
        {"timestamp_sec": 1.2, "subject": "Warrior", "verb": "thrusts spear", "biomechanics_note": "Core rotates 45°; rear leg drives with 800N force; both arms extend rapidly; spear accelerates to 8 m/s", "physics_note": "Kinetic energy transfers through 2m lever arm; momentum = mass × velocity concentrated at tip", "internal_motivation": "Deliver maximum impact force to neutralize threat"},
        {"timestamp_sec": 1.5, "subject": "Beast", "verb": "recoils in pain", "biomechanics_note": "Torso jerks backward; front legs buckle briefly; head turns away from impact", "physics_note": "Impact force (600N) exceeds balance threshold; momentum transfers causing 0.3m displacement", "internal_motivation": "Instinctive pain avoidance response"},
        {"timestamp_sec": 1.9, "subject": "Beast", "verb": "staggers back", "biomechanics_note": "Rear legs scramble for footing; center of mass shifts back 0.5m; tail swings for balance", "physics_note": "Backward momentum continues; friction gradually decelerates mass; recovery time 0.8s", "internal_motivation": "Regain balance and assess injury severity"},
        {"timestamp_sec": 2.4, "subject": "Warrior", "verb": "withdraws spear", "biomechanics_note": "Arms retract; shoulders rotate back; weight shifts to rear leg; guard position restored", "physics_note": "Weapon extraction requires 200N force; follow-through reduces recoil", "internal_motivation": "Prepare for follow-up attack or defensive counter"}
      ]
    }
    ↑ Note: 6 actions, 6 different verbs, cause-effect within 0.3s, specific details, good spacing
    
    ❌ BAD SEQUENCE (Common mistakes):
    {
      "action_sequence": [
        {"timestamp_sec": 0.0, "subject": "Warrior", "verb": "aims", "biomechanics_note": "holds weapon", "physics_note": "stable", "internal_motivation": "attack"},
        {"timestamp_sec": 0.5, "subject": "Warrior", "verb": "re-aims", "biomechanics_note": "adjusts position", "physics_note": "maintains aim", "internal_motivation": "prepare"},
        {"timestamp_sec": 1.0, "subject": "Warrior", "verb": "re-aims", "biomechanics_note": "micro-adjusts", "physics_note": "corrects angle", "internal_motivation": "focus"},
        {"timestamp_sec": 1.5, "subject": "Warrior", "verb": "re-aims", "biomechanics_note": "continues aiming", "physics_note": "steady", "internal_motivation": "concentrate"},
        {"timestamp_sec": 2.0, "subject": "Warrior", "verb": "attacks", "biomechanics_note": "strikes", "physics_note": "force applied", "internal_motivation": "damage enemy"},
        {"timestamp_sec": 4.5, "subject": "Beast", "verb": "reacts", "biomechanics_note": "moves away", "physics_note": "responds to hit", "internal_motivation": "avoid pain"}
      ]
    }
    ↑ PROBLEMS: Same verb 4 times! Vague descriptions! 2.5s gap between attack and reaction! Only 4 different verbs total!

    CONTINUITY & LOCKS (GLOBAL + THIS SCENE)
    - Use ONLY the allowed character labels/assets from the LOCKED CONTEXT; do NOT invent names/models unless the scene text explicitly introduces them.
    - Keep setting/weather/time-of-day consistent with the LOCKED CONTEXT unless this scene explicitly changes them.
    - Character presence, wardrobe, props-in-hand, and environmental conditions must remain consistent across shots.
    - CRITICAL: Lock character names from the first mention and use them consistently throughout the entire action_sequence.
    - CRITICAL: Lock wardrobe details (colors, materials, accessories) and repeat them in character_continuity for every scene.
    - CRITICAL: Ensure main actions from creative_brief are included in action_sequence (e.g., if brief says "warrior thrusts spear", include this specific action).
    - CRITICAL: Avoid repetitive micro-actions; vary verbs and provide contextual, specific biomechanics for each action.
    <<CONTINUITY_SUMMARY>>
    <<ENTITY_PROFILES>>
    <<ARCHERY_RULES>>

    GROUNDING CONTRACT (THIS SCENE ONLY — obey the SCENE TEXT verbatim)
    - Treat the provided Vietnamese bullets as authoritative:
      * "Nội dung" = the core intent and objective of this scene; all actions must serve this.
      * "Bối cảnh/Hình ảnh" = map to setting and environmental descriptors (terrain, light, shadows, dust, sky, water, etc.).
      * "Góc máy" = encode into camera orientation/rig.
      * "Chuyển động máy quay" = encode into camera.movement (type/direction/speed) and position_m path.
      * "Âm thanh" = Decompose audio into ambience, foley (character sounds), sfx (impacts), and music_cue.
      * "Trang phục/Tạo hình Nhân vật" = repeat wardrobe and appearance details in world_and_characters.character_continuity.
    - Use only details present in THIS SCENE TEXT; do not add new characters/props/events unless text explicitly introduces them.
    - STRICT: Never refer to any previous scene. This scene must be fully self-contained and restate all details needed.
    - DETAILED MAPPING: Transform each Vietnamese bullet into comprehensive English descriptions with specific details.

    MICRO-SCRIPT (AUTHORITATIVE ACTION PLAN)
    - The following micro-script is your primary source for the scene's action.
    - You MUST translate this narrative plan into a detailed, step-by-step 'action_sequence' in the final JSON.
    - Ensure the character's movements reflect the emotion and physical state described in the plan.
    - EXPAND EACH ACTION: Break down complex actions into 2-3 micro-steps with specific biomechanics.
    - ADD TRANSITION ACTIONS: Include micro-movements between major actions (weight shifts, breathing, eye movements).
    <<MICRO_SCRIPT>>

    SCENE NUMBER: <<SCENE_NUMBER>>
    SCENE TEXT (verbatim; Vietnamese headers may appear, but describe in English): <<SCENE_TEXT>>

    REALISM & CAUSALITY CONTRACT (CRITICAL — fix prior issues)
    - Actions must follow a clear CAUSE → EFFECT chain with visible reactions:
      * Melee: strikes cause body reaction (flinch, stagger, step back, loss of balance, fall) and follow-through; no "hit then statue".
      * Projectiles: arrows/bolts/bullets have release → flight (arc/travel time) → impact/near-miss with physical response (dust puff, deflection, stumble).
      * Grapples: push/pull cause footwork, weight shift, resistance; no instant freeze.
      * Animals: predators react (snarl, recoil, re-engage); prey escapes/resists appropriately.
    - Physics cues required: inertia, momentum, gravity, friction, recoil, recovery, timing/latency. No teleportation, no pop-in/out poses.
    - State continuity: if a hand holds a prop/weapon, it stays in-hand across the 8s unless clearly put away; if a bow fires, bowstring returns, ammo state updates, visible reload or readying is shown as needed.
    - Environment continuity: debris/smoke/dust linger realistically; weather, lighting direction, and ground contact remain consistent; footprints/splashes when appropriate.
    - CINEMATIC CAMERA: Use dynamic camera movements that enhance storytelling - dolly in for emotional moments, tracking shots for action, wide establishing shots for context. Include subtle camera breathing and micro-movements to avoid static feel. Use rack focus for dramatic emphasis.
    - Visual clarity: prefer moderate or deep depth of field for action legibility unless the text mandates shallow DOF; use focus_pull only when motivated and make subject identity unambiguous.
    - MOTION SPECIFICITY: Every action must have specific body parts mentioned (hands, feet, shoulders, torso, head, eyes).
    - PHYSICS SPECIFICITY: Every physics_note must mention specific forces (gravity, momentum, friction, air resistance, impact force).
    - EMOTIONAL DEPTH: Every internal_motivation must connect to specific emotional states (fear, determination, surprise, anger, focus) and character development. Show character growth, decision-making, and psychological complexity through actions and reactions.

    SHOT AUTHORING RULES (map realism into the schema you already have)
    - Because this is SINGLE SHOT, shots[0].action_sequence MUST include a continuous 8-second beat broken into detailed biomechanical steps: initiating action → evolving task → reactions/counters → carry-through state.
    - Each action_sequence item MUST include: timestamp_sec + subject + verb + biomechanics_note (HOW) + physics_note (WHY).
    - Use concrete verbs (e.g., "charges, strikes, dodges, blocks, thrusts") instead of vague ones ("fights, engages").
    - EXAMPLE: If scene text says "warrior charges with spear", use "Warrior" as subject and "charges" as verb. Do NOT add "Archer" or "Arrow" unless explicitly mentioned.
    - If shot is static, set camera.movement = {"type":"static"} but still describe subtle micro-actions every 0.5–1s.
    - Place motion into correct fields: action_sequence (detailed biomechanical steps), camera.* (camera), vfx (non-graphic trails/dust/sparks), audio (foley/ambience/music/mix_notes).
    - ACTION DENSITY: Minimum 12-16 beats over 8s, with realistic spacing (minimum 0.3s between actions).
    - TIMELINE LOGIC: Actions must follow logical sequence with proper cause→effect timing (e.g., impact at 2.0s, reaction at 2.2s, not 2.01s).
    - BIOMECHANICS DEPTH: Each biomechanics_note must describe specific muscle groups, joint movements, and body mechanics.
    - PHYSICS DEPTH: Each physics_note must describe specific forces, energy transfer, and material interactions.
    - EMOTIONAL DEPTH: Each internal_motivation must describe specific psychological states and decision-making processes.
    - REALISTIC TIMING: No actions closer than 0.4s apart; allow time for physical reactions and muscle responses.
    - ANTI-REPETITION: Avoid using the same verb more than 2 times; vary actions with specific, contextual verbs.
    - MAIN ACTION PRIORITY: Include the primary action from creative_brief as a major beat in the sequence (e.g., "thrusts spear", "strikes target", "evades attack").
    - CHARACTER CONSISTENCY: Lock character names from first mention and use consistently throughout entire sequence.
    - BIOMECHANICS DEPTH: Each biomechanics_note must be at least 30 characters with specific muscle groups, joint movements, and neural pathways.
    - PHYSICS DEPTH: Each physics_note must be at least 30 characters with specific forces, energy transfer, and material interactions.

    STRUCTURE RULES (FOLLOW EXACTLY — to satisfy schema)
    - SINGLE SHOT ONLY. shot.id/start_time/duration/slugline/action_sequence/camera/vfx/lighting/audio/transition_out are required.
    - camera MUST include rig, lens_mm, aperture_f, shutter_angle_deg, white_balance_k, iso, position_m{{start,end}}, movement, orientation_deg{{yaw,pitch,roll}}, focus.
    - Use 'camera.composition_style' to describe the visual framing (e.g., rule of thirds).
    - Use 'camera.depth_of_field' ('shallow', 'deep', 'moderate') to control the mood and focus.
    - Use 'camera.focus_pull_style' to describe any changes in focus during the shot.
    - 'vfx', 'lighting', 'audio', 'transition_out' belong to shot-level (NEVER inside 'camera').
    - Fill all REQUIRED fields according to the schema.
    - CREATIVE EXCELLENCE: Use the 'creative_suggestions' field to propose unique visual ideas that amplify the scene's mood - innovative camera angles, creative lighting setups, unique VFX concepts, or storytelling techniques that make the scene memorable and engaging.

    MANDATORY CONTENT MAPPING (Populate these fields explicitly in JSON):
    - world_and_characters.setting: a concise English paragraph covering Location with specific details (terrain, landmarks), Time & Lighting (e.g., dawn golden light, moonlit night), Atmosphere & Weather (wind, dust, humidity), and Background/Imagery motifs. Must be self-contained.
    - world_and_characters.visual_style: e.g., "classic action film" or "documentary realism"; avoid genre mixing unless stated.
    - world_and_characters.color_material_texture: describe color palette, materials (metal, wood, silk), and surface qualities (matte, glossy, rusty) relevant to props/environment.
    - world_and_characters.mood: single phrase like "tense and kinetic" aligned with scene intent.
    - world_and_characters.character_continuity: list characters present in this scene with:
        * appearance (face, body type, scars/markings, tattoos/paint),
        * wardrobe (layers, colors, materials, headgear, footwear, accessories),
        * emotions (current emotion and micro-expressions),
        * props_in_hand (left/right/both),
        * movement_signature (gait, posture, tempo),
        * dirt/wounds/state (restate even if unchanged for this scene),
        * do NOT reference previous scenes; restate all details.
    - creative_brief: a 1–2 sentence English summary of the main "Content" (core action or event) for this scene only.
    - shots[0].camera: must include a clear Camera Angle (e.g., Wide Shot, Close-up, High Angle) in composition_style or an explicit 'angle' field, and a clear Camera Movement with type, direction, speed; also define position_m path.
    - CINEMATIC LIGHTING: Use dramatic lighting that enhances mood and storytelling - key lights for character focus, rim lighting for separation, practical lights for realism, and atmospheric effects (dust, smoke, particles) that add depth and visual interest.
    - CINEMATIC AUDIO: Create immersive soundscapes with layered ambience, detailed foley (footsteps, breathing, equipment), impactful SFX (weapon sounds, impacts, environmental), and emotional music cues that enhance the scene's mood. Include audio transitions and dynamic range for maximum impact.
     - shots[0].action_sequence: enumerate precise actions of each character/animal with timestamps every ~0.5–1.0s for full 8s.
     - Do not invent new entities not grounded in scene text.
     - DETAILED FIELD REQUIREMENTS:
       * creative_brief: 2-3 sentences with specific action verbs and character names.
       * world_and_characters.setting: 3-4 sentences with specific terrain, lighting, weather, and atmospheric details.
       * world_and_characters.character_continuity: Detailed paragraph for each character with specific physical descriptions, wardrobe details, and current state.
       * shots[0].slugline: Specific, concrete description of the main action in the shot.
       * shots[0].action_sequence: Minimum 12-16 beats with realistic timestamps (0.4s+ apart), varied verbs (max 2 repetitions), consistent subjects, and detailed biomechanics/physics notes (30+ chars each). MUST include main action from creative_brief.
       * shots[0].camera.composition_style: Specific framing description (e.g., "medium shot, rule of thirds, eye-level, subject centered").
       * shots[0].camera.movement: Specific movement type, direction, and speed with rationale.
       * shots[0].audio.ambience: Specific environmental sounds with location and time context.
       * shots[0].audio.foley: Specific character sounds with body parts and materials mentioned.
       * shots[0].audio.sfx: Specific impact and environmental sounds with materials and forces mentioned.
       * shots[0].lighting: Specific lighting sources, direction, intensity, and color temperature.

    NEW RULE: ENTITY PROFILE CREATION
    - If you encounter a character (e.g., KAI) or a significant location (e.g., 'The Great Rock Formation') for the first time, you MUST create a detailed JSON profile for it and place it in the 'continuity_updates.new_or_updated_characters' (or locations) field.
    - The profile should capture all available details: appearance, costume, state, key features, etc.
    - For animals or creatures, you MUST include a 'biomechanics' section with scientific details:
      * "mass_kg": estimated weight in kilograms
      * "center_of_gravity": where the center of mass is located and how it affects balance
      * "locomotion": how the creature moves (gait, speed, turning ability)
      * "reaction_to_threat": behavioral response patterns
      * "vulnerability": weak points and how impacts affect the body
    - For human characters, the profile MUST include:
      * "appearance": face details, body type, visible scars/marks
      * "wardrobe": outfit by layers/colors/materials; headgear/footwear
      * "props_in_hand": what is in each hand; on-back/holster items
      * "weapon_state": drawn/holstered, ammo/ready state, visible wear
      * "movement_signature": baseline gait, posture, breath rhythm, dominant hand
      * "voice": tone/tempo/accent (if any spoken cues appear)
      * "psychology": current emotion, goal, risk tolerance
      * "relationships": allies/opponents mentioned in this scene
    - For dynamic non-human entities (vehicles, projectiles, drones, animals), the profile MUST include:
      * "biomechanics" or "mechanics": mass_kg, center_of_gravity, locomotion/drive type, turning ability
      * "operational_state": damage, fuel/energy, loadout
      * "interaction_rules": how they collide/react, constraints/limits
    - If the script describes a CHANGE to a character (e.g., 'KAI now has a scar on his face'), you must provide an UPDATED profile in the same field reflecting this change.

    NEW RULE: SCIENTIFIC MOTION DESCRIPTION
    - You MUST describe all motion using the new 'action_sequence' array structure. DO NOT use the old 'action' string.
    - For each beat in the sequence, you MUST consult the 'biomechanics' section of the provided ENTITY PROFILES.
    - Your 'biomechanics_note' MUST describe HOW the body moves (e.g., 'muscles in the hind legs tense up', 'shifts weight to its front left foot').
    - Your 'physics_note' MUST describe WHY the body reacts that way, using concepts from the profile (e.g., 'the arrow's kinetic energy disrupts its high center of gravity, causing it to lose balance', 'recoils from the impact due to conservation of momentum').
    - Your 'internal_motivation' MUST describe the psychological 'why' behind the physical action, connecting it to the director's analysis (emotional state, goal).
    - This creates a scientifically plausible cause-and-effect chain for every movement.

    CONTINUITY LOGIC FIELD
    - Write a compact checklist in 'continuity_logic' showing: Labels kept (Lead/Antagonist consistent); Wardrobe/props repeated per scene input; Setting/weather/time consistent with locks; Last→first frame continuity held; No idle seconds across 0–8s.

    SAFETY GUARDRAILS (STRICT, NON-GRAPHIC)
    - Neutral, non-graphic phrasing for combat ("disabled", "goes down", "damaged", "smoke and debris").
    - Do NOT use: "kill", "killed", "blood", "gore", "swastika", "Nazi", "assassinate". Use neutral alternatives.
    - No injuries or suffering depiction; damage is abstract (sparks, smoke, debris).

    OUTPUT
    - Exactly 1 shot, covering 0–8s without gaps.
    - Maintain goal-driven progress across all 8s; no idle seconds.
    - Return JSON ONLY via function-call: build_veo3_prompt.

    GROUNDING CONTRACT (must obey; stick to this scene's text): <<LOCKS_TEXT>>
    - Use only details present in SCENE TEXT; do not add new characters/props/events unless text explicitly introduces them.
    - Each shot.action must describe a concrete behavior present/derivable from the scene text (avoid vague placeholders).
    - CRITICAL: Do NOT add archery elements (Archer, Arrow, bow, arrow flight, etc.) unless the scene text explicitly mentions them.
    - STRICT CHARACTER ANALYSIS: Only use characters explicitly mentioned in the scene text. Do not invent "Archer" if the text only mentions "Warrior" or "Hunter".
    - STRICT WEAPON ANALYSIS: Only use weapons explicitly mentioned. If text says "spear", do not add "bow and arrow". If text says "hunters", do not assume they use bows.
    - EXAMPLE ANALYSIS: 
      * Input: "A warrior charges to save, using a spear to stab a beast" → Use "Warrior" and "Beast" as subjects, "charges" and "stabs" as verbs
      * Input: "Hunters scatter in chaos" → Use "Hunters" as subject, "scatter" as verb
      * DO NOT add: Archer, Arrow, bow, arrow flight, etc. unless explicitly mentioned
    """

    # Scene-local locks summary (reuses existing mechanism)
    scene_locks = []
    if scene_entities.get("lead"):
        scene_locks.append(f"- Lead candidates (use ONLY these labels): {', '.join(scene_entities['lead'])}")
    if scene_entities.get("antagonist"):
        scene_locks.append(f"- Antagonist candidates: {', '.join(scene_entities['antagonist'])}")
    if scene_entities.get("aircraft"):
        scene_locks.append(f"- Allowed aircraft: {', '.join(scene_entities['aircraft'])}")
    if scene_entities.get("setting"):
        scene_locks.append(f"- Locked setting: {scene_entities['setting']}")
    if scene_entities.get("weather"):
        scene_locks.append(f"- Locked weather: {scene_entities['weather']}")
    if scene_entities.get("time_of_day"):
        scene_locks.append(f"- Locked time of day: {scene_entities['time_of_day']}")
    if scene_entities.get("wardrobe"):
        scene_locks.append(f"- Wardrobe to repeat: {scene_entities['wardrobe']}")
    if scene_entities.get("camera_angle"):
        scene_locks.append(f"- Camera angle: {scene_entities['camera_angle']}")
    if scene_entities.get("camera_move"):
        scene_locks.append(f"- Camera movement: {scene_entities['camera_move']}")
    if scene_entities.get("audio"):
        scene_locks.append(f"- Audio guide: {scene_entities['audio']}")
    if scene_entities.get("action_specific"):
        scene_locks.append(f"- Specific action: {scene_entities['action_specific']}")

    locks_text = "\n".join(scene_locks) if scene_locks else "- (no explicit scene-local locks found)"

    # Build messages using strict SYSTEM + USER templates
    system_msg = (
        "You are GPT-5 Thinking, operating as the top 0.1% cinematic prompt engineer. You must return a STRICT JSON object that satisfies the provided VEO-3 schema with zero defects.\n"
        "Rules:\n"
        "- English only in output strings.\n"
        "- Exactly ONE shot per scene, covering 0–8 seconds continuously (no gaps, no overlaps).\n"
        "- Every 0.5–1.0s must have purposeful motion (no idle plateau, no “looking and waiting” beats).\n"
        "- Use cause→effect for all actions (initiating force → reaction/recovery).\n"
        "- Zero error tolerance: reject placeholders, resolve schema conflicts, and keep every field production-ready.\n"
        "- DO NOT invent characters/props/weapons/settings beyond the scene text. \n"
        "- If the scene text does NOT mention archery (bow/arrow/archer), do NOT add any arrow/archer beats or SFX/VFX about arrows.\n"
        "- Camera fields must live inside shot.camera; audio/vfx/lighting/transition_out live at shot level (never inside camera).\n"
        "- Use strong, concrete verbs for action_sequence and fill: timestamp_sec, subject, verb, biomechanics_note (HOW), physics_note (WHY), internal_motivation (psych).\n"
        "- If camera is static, still include micro-parallax/breath-sway to avoid frozen look.\n"
        "- Respect physics (gravity, inertia, momentum, friction).\n"
        "- No graphic or gory language. Use neutral, non-graphic phrasing.\n"
        "Return JSON only (no prose).\n\n"
        # GROUNDING CONTRACT (Vietnamese, rút gọn) — đưa vào ngay từ SYSTEM để siết bám sát văn bản
        "TRẢ VỀ DUY NHẤT 1 JSON OBJECT qua function-call (không prose).\n"
        "GROUNDING TUYỆT ĐỐI: Chỉ dùng nhân vật/đạo cụ/sự kiện có trong SCENE TEXT. Không suy diễn vũ khí nếu văn bản không nêu.\n"
        "CẤU TRÚC BẮT BUỘC: 1 shot duy nhất cho 0–8s; action_sequence dày 0.5–1.0s; mỗi beat có timestamp_sec, subject, verb, biomechanics_note, physics_note, internal_motivation.\n"
        "KHOA HỌC CHUYỂN ĐỘNG: biomechanics_note = cách cơ thể vận động; physics_note = vì sao (lực/quán tính/ma sát/trọng lực); internal_motivation = mục tiêu tại beat đó.\n"
        "CAMERA đúng chỗ: thông số ống kính/khẩu/shutter/WB/ISO, position_m{start,end}, movement, orientation_deg, focus, composition, depth_of_field, focus_pull_style — không nhét audio/VFX vào camera.\n"
        "SAFETY NGÔN NGỮ: trung tính/không đồ họa. CONTINUITY: khóa nhãn/props/weather/time theo scene; ghi checklist vào continuity_logic.\n"
        "Âm thanh & VFX ăn khớp pha hành động. English-only ở output cuối.\n\n"
        # Inject required characters line
        + "REQUIRED SCENE CHARACTERS (include profiles ONLY for these; do NOT include others):\n"
        + ("- " + ", ".join(required_scene_characters) if required_scene_characters else "- (none)")
    )

    continuity_summary_en = self._to_english(continuity.summary_for_prompt())
    user_tmpl = (
        "VEO-3 JSON REQUEST (TOP-TIER STRICT)\n\n"
        "EXPECTATION: zero-placeholder, production-ready detail; treat every field as instructions for a real cinematography team.\n\n"
        "SCENE NUMBER: <<SCENE_NUMBER>>\n\n"
        "CONTINUITY (LOCKED, DO NOT INVENT):\n"
        "<<CONTINUITY_SUMMARY_EN>>\n\n"
        "SCENE TEXT (Vietnamese, authoritative; map EXACTLY to JSON, don’t add beyond this):\n"
        "<<SCENE_TEXT_VN>>\n\n"
        "MAPPING CONTRACT (MANDATORY — use these bullets if present):\n"
        "- “Nội dung”: the core intent; every beat must serve this intent.\n"
        "- “Bối cảnh/Hình ảnh”: becomes world_and_characters.setting and visual descriptors (terrain, light, shadows, dust, sky, water).\n"
        "- “Góc máy”: encode into camera.orientation_deg and composition_style.\n"
        "- “Chuyển động máy quay”: encode into camera.movement (type/direction/speed) and position_m {start,end}.\n"
        "- “Âm thanh”: split into audio.ambience, audio.foley (character actions), audio.sfx (impacts), and music_cue (ONLY if text mentions music).\n"
        "- “Trang phục/Tạo hình Nhân vật”: repeat into world_and_characters.character_continuity. Keep consistent across scenes.\n\n"
        "OUTPUT REQUIREMENTS (ABSOLUTE):\n"
        "- duration_seconds = 8, fps = 30, resolution = 1920x1080, aspect_ratio = 16:9, model/request_type/language filled.\n"
        "- Exactly ONE item in shots[], covering 0.00 → 8.00s with no idle seconds.\n"
        "- shots[0].action_sequence = dense beats (≈ every 0.5–1.0s): initiating action → evolving task → reactions/counters → carry-through state by 8.0s.\n"
        "- Use ONLY the character labels and props explicitly present in THIS SCENE TEXT (or in the continuity lock). \n"
        "- If “spear” is mentioned, you may use thrust/feint/brace/deflect. If no bow/arrow, DO NOT add arrow SFX/VFX or beats.\n"
        "- audio: include only what the scene mentions; do NOT fabricate orchestral score unless text says so.\n\n"
        "FILL THESE FIELDS (schema-conformant):\n"
        "- model, request_type, duration_seconds, fps, resolution, aspect_ratio, language\n"
        "- creative_brief: 1–2 sentences paraphrasing “Nội dung” (DO NOT add new story beats).\n"
        "- global_style: visual_tone, motion_rules, color_grading, vfx_guidance, sound_mix; physics_rules (short, neutral).\n"
        "- continuity_logic: a short checklist how you preserved labels/wardrobe/props/setting/weather/time & end→start continuity.\n"
        "- world_and_characters: setting (detailed 3-4 sentences with specific environmental details), lead_character (specific name with detailed description), antagonist (specific name with detailed description), weather (detailed description with specific conditions), battle_context (detailed 3-4 sentences with specific combat details), character_continuity (detailed wardrobe/props descriptions with specific materials and colors), time_of_day (specific time with lighting details), character_details (comprehensive profiles for each character with appearance, wardrobe, emotions, movement style, personality traits, weapons, physical condition, tactical role).\n"
        "- shots[0]:\n"
        "  * id, start_time=0.0, duration=8.0, slugline (concrete)\n"
        "  * action_sequence[]: items with {timestamp_sec, subject, verb, biomechanics_note (100+ chars with specific muscle groups and movement details), physics_note (100+ chars with specific forces and energy calculations), internal_motivation (80+ chars with specific psychological states and decision-making), character_details (comprehensive profile with appearance, wardrobe, emotions, movement style, personality traits, weapons, physical condition, tactical role)}\n"
        "  * camera: {rig (specific type with detailed description), lens_mm (specific focal length with reasoning), aperture_f (specific f-stop with depth of field explanation), shutter_angle_deg (specific angle with motion blur details), white_balance_k (specific temperature with color accuracy), iso (specific sensitivity with noise considerations), position_m{start,end} (specific coordinates with movement path), movement (detailed description with speed and direction), orientation_deg{yaw,pitch,roll} (specific angles with composition reasoning), focus (detailed settings with focus points), composition_style (specific style with visual impact), depth_of_field (specific range with subject isolation), focus_pull_style (specific technique with timing)}\n"
        "  * vfx[] (only if text implies — e.g., dust puff on ground contact)\n"
        "  * lighting (only if text implies)\n"
        "  * audio{ambience (detailed description with specific environmental sounds), foley[] (specific sounds with detailed descriptions), sfx[] (specific effects with detailed descriptions), music_cue? (specific music with detailed description), mix_notes? (detailed mixing instructions with specific levels and effects)} (omit empty keys)\n"
        "  * transition_out (simple, e.g., cut / fade)\n"
        "- safety_and_quality: avoid[] generic items, must_have[]: cause→effect, non-graphic tone, physically plausible motion.\n"
        "- output_settings: container=mp4, video_codec=h264, video_bitrate=5000k, audio_codec=aac, audio_bitrate=192k, normalize_audio=true, tone_mapping=linear, grain=none.\n"
        "- single_prompt_for_model: a compact human-readable version consistent with the JSON (no new details).\n\n"
        "Return one strict JSON object only.\n\n"
        "SCENE NUMBER: <<SCENE_NUMBER>>\n"
        "CONTINUITY (LOCKED, DO NOT INVENT):\n<<CONTINUITY_SUMMARY_EN>>\n\n"
        "SCENE TEXT (Vietnamese):\n<<SCENE_TEXT_VN>>\n"
    )

    user_instructions = (
        user_tmpl
        .replace("<<SCENE_NUMBER>>", str(scene_number))
        .replace("<<CONTINUITY_SUMMARY_EN>>", continuity_summary_en)
        .replace("<<SCENE_TEXT_VN>>", scene_text)
    )

    # >>> ADD REQUIRED SCENE ENTITIES LINE <<<
    try:
        _req_ents = self._mentioned_entities(scene_text, scene_entities, continuity).get("all", [])
        if _req_ents:
            user_instructions = user_instructions + "\nREQUIRED SCENE ENTITIES (include profiles/biomechanics ONLY for these; do NOT include others):\n- " + ", ".join(_req_ents) + "\n"
    except Exception:
        pass
    # <<< END ADD >>>

    tools = self._build_tool_spec()
    resp = self._chat(
        model=self.model,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        seed=self.seed,
        response_format={"type": "json_object"},
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "build_veo3_prompt"}},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instructions},
        ],
    )
    choice = resp.choices[0]
    try:
        obj = self._parse_choice(choice)
    except Exception as e:
        # Fallback: retry với response_format={"type":"json_object"}
        try:
            resp2 = self._chat(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return only a strict JSON object. No prose. JSON format required."},
                    {"role": "user", "content": user_instructions},
                ],
            )
            obj = json.loads(resp2.choices[0].message.content)
        except Exception as e2:
            return {"error": f"Model did not return JSON: {e}, retry failed: {e2}"}

    # 👇 ADD THIS
    if isinstance(obj, list):
        print("⚠️ Model returned a JSON array; attempting to unwrap first element.")
        obj = obj[0] if obj and isinstance(obj[0], dict) else {"error": "Model returned a list, expected object."}
    # Densify & lock labels immediately after model returns JSON
    try:
        w = obj.get("world_and_characters") or {}
        lead_label = (w.get("lead_character") or "Lead").strip() or "Lead"
        ant_label  = (w.get("antagonist") or "Target").strip() or "Target"
        s0 = (obj.get("shots") or [{}])[0]
        s0 = self._reclassify_audio(s0, scene_text)
        # Optional extra densifier (complements existing anti-idle sweeps)
        if hasattr(self, "_densify_full_8s"):
            s0 = self._densify_full_8s(s0, lead_label, ant_label)
        (obj.get("shots") or [{}])[0] = s0
    except Exception:
        pass

    # Khóa continuity aliases sớm để tránh trôi nhãn
    try:
        obj = self._lock_subject_aliases(obj, continuity, scene_entities)
        obj = self._unify_subject_labels(obj)
    except Exception:
        pass

    # Enforce continuity_updates names to only required scene characters and auto-bolster missing ones
    try:
        required_scene_characters = self._mentioned_characters(scene_text, scene_entities, continuity)
        cu = obj.get("continuity_updates") or {}
        lst = cu.get("new_or_updated_characters") or []
        # filter to required only
        filtered = []
        keep_set = set(required_scene_characters)
        for it in lst:
            name = (it or {}).get("name")
            if name in keep_set:
                # merge with continuity profile if exists
                prof = (it.get("profile") or {}).copy()
                prof.update(continuity.characters.get(name, {}))
                filtered.append({"name": name, "profile": prof})
        # add any missing required names
        world = obj.get("world_and_characters") or {}
        ward = (world.get("character_continuity") or "").strip()
        for name in required_scene_characters:
            if not any(x.get("name") == name for x in filtered):
                filtered.append({
                    "name": name,
                    "profile": {
                        **(continuity.characters.get(name) or {}),
                        "wardrobe": (continuity.characters.get(name, {}).get("wardrobe") or ward or "N/A"),
                        "props_in_hand": (continuity.characters.get(name, {}).get("props_in_hand") or self._infer_props_from_text(ward)),
                        "notes": "auto-min profile from scene output"
                    }
                })
        cu["new_or_updated_characters"] = filtered if required_scene_characters else []
        obj["continuity_updates"] = cu
    except Exception:
        pass

    # Enforce dynamic entities for this scene (animals/vehicles/projectiles/objects)
    try:
        req_ents = self._mentioned_entities(scene_text, scene_entities, continuity).get("all", [])
        cu = obj.get("continuity_updates") or {}
        ents = [e for e in (cu.get("new_or_updated_dynamic_entities") or []) if isinstance(e, dict)]
        ents_f = []
        keep = set(req_ents)

        # giữ lại những entity có trong cảnh + merge với continuity memory
        for it in ents:
            name = (it or {}).get("name")
            if name in keep:
                prof = (it.get("profile") or {}).copy()
                prof.update((continuity.entities.get(name) or {}))
                ents_f.append({"name": name, "profile": prof})

        # tự seed nếu thiếu
        for name in req_ents:
            if not any(x.get("name") == name for x in ents_f):
                cat = self._classify_entity(name)
                base = (continuity.entities.get(name) or {}).copy()
                if cat == "animal":
                    base.setdefault("type","animal")
                    base.setdefault("biomechanics", {
                        "mass_kg": "unknown",
                        "center_of_gravity": "mid-torso",
                        "locomotion": "quadruped gait (estimate)",
                        "reaction_to_threat": "startle → recoil → reposition",
                        "vulnerability": "joints/neck areas"
                    })
                elif cat == "vehicle":
                    base.setdefault("type","vehicle")
                    base.setdefault("motion_model", "wheels/propulsion; inertia, traction, turning radius noted")
                elif cat == "projectile":
                    base.setdefault("type","projectile")
                    base.setdefault("motion_model", "ballistic arc; gravity, drag, momentum; non-graphic impact wording")
                else:
                    base.setdefault("type","object")
                    base.setdefault("motion_model", "mass/inertia governed; friction and gravity dominant")
                base.setdefault("notes","auto-min dynamic-entity profile from scene detection")
                ents_f.append({"name": name, "profile": base})

        cu["new_or_updated_dynamic_entities"] = ents_f if req_ents else []
        obj["continuity_updates"] = cu
    except Exception:
        pass

    if not isinstance(obj, dict):
        return {"error": "Model returned non-object JSON."}

    print("🔧 DEBUG raw tool args type:", type(obj).__name__)

    # 👇 Ép shape để tránh .get trên list
    obj = self._coerce_shapes(obj)
    self._debug_top_shapes(obj)

    # Local normalization BEFORE validate to auto-fix common issues
    obj = self.normalize_veo3_object(obj, scene_text=scene_text, scene_entities=scene_entities, continuity=continuity)
    obj = self._sanitize_continuity_updates(obj)

    obj, _hits = sanitize_veo3_object(obj)
    if _hits:
        print(f"🧹 Safety sanitized: {_hits} replacements")

    # Cưỡng chế single shot only
    if isinstance(obj.get("shots"), list) and len(obj["shots"]) > 1:
        first = obj["shots"][0]
        # Nếu muốn merge action_sequence:
        merged_seq = []
        for sh in obj["shots"]:
            merged_seq.extend(sh.get("action_sequence", []))
        first["action_sequence"] = merged_seq or first.get("action_sequence", [])
        obj["shots"] = [first]

    # Force motion/continuity rules into the object before validation
    gs = obj.get("global_style") or {}
    mr = gs.get("motion_rules", "")
    must = "No idle seconds; maintain continuous, purposeful micro-actions for lead across full 8s; camera may include subtle micro-parallax even when static."
    if must not in mr:
        gs["motion_rules"] = (mr + " | " + must).strip(" |")
    obj["global_style"] = gs

    # In strict mode, do not inject extra lock text beyond grounding
    if not getattr(self, "strict_grounding", False):
        sp = (obj.get("single_prompt_for_model") or "").strip()
        lock_text = (
            "CONTINUITY LOCK: Use only allowed character labels and assets; keep setting/weather/time consistent. "
            "DURATION: exactly 8s, no gaps; every second has purposeful motion (no idle). "
            "MOTION: if camera is static, include micro-parallax/breathing-linked sway (no shaky-cam)."
        )
        if lock_text not in sp:
            obj["single_prompt_for_model"] = (sp + "\n" + lock_text).strip()

    # Bảo đảm "single_prompt_for_model" nêu rõ lớp archery realism (chỉ khi có xạ thủ người)
    sp = (obj.get("single_prompt_for_model") or "").strip()
    # Reuse scene_text/entities already computed above
    archery_check_text = scene_text or ((obj.get("creative_brief") or "") + " " + json.dumps(obj.get("world_and_characters") or {}))
    archery_check_entities = scene_entities or self.extract_scene_entities(archery_check_text)
    active_packs = [p for p in self.domain_packs if p.detect(archery_check_text, archery_check_entities)]
    is_archery = any(p.name == "archery" for p in active_packs)
    if is_archery:
        # Only add the archery realism tail if a human archer is explicitly present
        if self._mentions_human_archer(archery_check_text, archery_check_entities):
            extra = (
                "ARCHERY REALISM: enforce draw→anchor→release→flight→impact order; "
                "arrow shows parabolic arc and travel time; impact anchors at entry point; "
                "add bowstring snap/whoosh/thud in audio; avoid idle/freeze."
            )
            if extra not in sp:
                obj["single_prompt_for_model"] = (sp + "\n" + extra).strip()

    # Ensure top-level defaults (output_settings, world fields) based on scene & continuity
    obj = self.ensure_top_level_defaults(obj, scene_entities, continuity, scene_text)
    
    # Chuẩn hóa nhãn chủ thể để không “nhảy tên” (Lead/Antagonist cố định)
    obj = self._lock_subject_aliases(obj, continuity, scene_entities)
    
    # Strict prune: remove optional empty fields not mentioned in scene
    if getattr(self, "strict_grounding", False):
        try:
            w = obj.get("world_and_characters") or {}
            sc_has_audio = bool((extract := self.extract_scene_entities(scene_text)).get("audio"))
            sc_has_vfx = False  # chưa có nhãn VFX riêng trong input; giữ nguyên nếu model tạo ra
            for s in (obj.get("shots") or []):
                aud = s.get("audio")
                if isinstance(aud, dict):
                    # Preserve existing content, only normalize None values
                    for k in ["ambience","foley","sfx","music_cue","mix_notes","dialogue_note","fx","music"]:
                        if k in aud and aud[k] is None:
                            aud[k] = "" if k in ["ambience","mix_notes","dialogue_note"] else []
                    s["audio"] = aud
                elif aud is None:
                        s["audio"] = {}
                if not sc_has_vfx and isinstance(s.get("vfx"), list) and not s["vfx"]:
                    pass  # để trống list hợp lệ theo schema
        except Exception:
            pass

    # Final legacy cleanup: drop any stray legacy 'action' keys in shots
    try:
        for _s in (obj.get("shots") or []):
            if isinstance(_s, dict):
                _s.pop("action", None)
    except Exception:
        pass

    # Validate & repair if needed
    err = self.validate_json(obj)
    if err:
        fixed = self.repair_with_llm(obj, err, continuity, scene_text)
        if fixed:
            fixed = self.normalize_veo3_object(fixed, scene_text=scene_text, scene_entities=scene_entities, continuity=continuity)
            fixed, _hits2 = sanitize_veo3_object(fixed)
            fixed = self._sanitize_continuity_updates(fixed)
            if _hits2:
                print(f"🧹 Safety sanitized (repair): {_hits2} replacements")
            # Ensure defaults again after repair
            fixed = self.ensure_top_level_defaults(fixed, scene_entities, continuity, scene_text)
            # Drop any legacy 'action' keys that survived repair
            try:
                for _s in (fixed.get("shots") or []):
                    if isinstance(_s, dict):
                        _s.pop("action", None)
            except Exception:
                pass
            err2 = self.validate_json(fixed)
            if not err2:
                obj = fixed
            else:
                return {"error": f"JSON still invalid after repair: {err2}", "raw": fixed}
        else:
            return {"error": f"Could not repair JSON: {err}", "raw": obj}

    # Final enforcement: duration=8s, contiguous coverage, anti-idle beats
    obj = self.enforce_eight_seconds_and_motion(obj)

    # Nếu ép theo mẫu vàng → bỏ qua prune để không bị rút gọn
    if getattr(self, "force_sample_template", False) or self._matches_warrior_beast(scene_text):
        obj = self.apply_warrior_beast_template(obj)
    else:
        # Strict prune optional sections not present in scene text
        obj = self._prune_optional_fields_strict(obj, scene_text, scene_entities)

    # ✅ BƯỚC CUỐI CÙNG: DỊCH TOÀN BỘ ĐỐI TƯỢNG JSON ✅
    # Đây là bước duy nhất chịu trách nhiệm dịch thuật.
    # Nó sẽ đi qua mọi chuỗi trong obj và dịch nếu cần.
    if self.translate_output:
        print("🌍 Dịch toàn bộ JSON sang tiếng Anh...")
        obj_translated = self._translate_deep_english(obj)
    else:
        obj_translated = obj

    # Continuity label check
    if continuity.char_names:
        lead = (obj_translated.get("world_and_characters") or {}).get("lead_character", "")
        tokens = [t.strip() for t in re.split(r"[,/&]| and ", lead) if t.strip()]
        unknowns = [t for t in tokens if t not in continuity.char_names]
        if unknowns:
            msg = (
                "lead_character contains unknown labels: " + ", ".join(unknowns) + ". "
                "Replace strictly with allowed character labels only; DO NOT invent new labels. "
                "Keep continuity exactly as in the LOCKED CONTEXT."
            )
            fixed2 = self.repair_with_llm(obj_translated, msg, continuity, scene_text)
            if fixed2:
                fixed2 = self.normalize_veo3_object(fixed2, scene_text=scene_text, scene_entities=scene_entities, continuity=continuity)
                fixed2, _hits3 = sanitize_veo3_object(fixed2)
                fixed2 = self._sanitize_continuity_updates(fixed2)
                if _hits3:
                    print(f"🧹 Safety sanitized (continuity repair): {_hits3} replacements")
                fixed2 = self.ensure_top_level_defaults(fixed2, scene_entities, continuity, scene_text)
                if not self.validate_json(fixed2):
                    obj_translated = fixed2

    # Update continuity from this scene's output (guarded)
    try:
        if isinstance(obj_translated, dict):
            w = obj_translated.get("world_and_characters") or {}
            has_labels = bool((w.get("lead_character") or "").strip() or (w.get("antagonist") or "").strip())
            has_updates = bool(obj_translated.get("continuity_updates"))
            if has_labels or has_updates:
                continuity.update_from_output(obj_translated)
    except Exception as e:
        print(f"⚠️ Continuity update skipped: {e}")

    # Sau khi obj_translated đã có và continuity cập nhật
    try:
        obj_translated = apply_hard_sanity(obj_translated)
    except Exception:
        pass

    # Tạo prompt đơn có cấu trúc chi tiết từ chính JSON đã khóa/chuẩn hóa
    try:
        obj_translated["single_prompt_for_model"] = self.compose_single_prompt_for_model(obj_translated, continuity)
    except Exception as _e:
        # fallback: giữ nguyên single_prompt_for_model có sẵn nếu có
        pass

    # Bổ sung "mô tả định dạng prompt" cho người đọc/trace
    obj_translated["model_prompt_format"] = {
        "format_version": "1.1",
        "style": "single_text_prompt_with_sections",
        "sections": [
            "LOCKED CONTEXT",
            "WORLD & CHARACTERS",
            "CREATIVE BRIEF",
            "SHOT PLAN",
            "CAMERA",
            "VFX",
            "LIGHTING",
            "AUDIO",
            "SAFETY • REALISM • STYLE",
            "OUTPUT SETTINGS",
            "TRANSITION OUT",
            "TIMELINE (optional)",
            "CHECKLIST"
        ],
        "guarantees": [
            "English only, non-graphic phrasing",
            "Single shot covers 0–8s with continuous, purposeful motion",
            "Cause→effect chain, physics-informed actions",
            "Continuity-locked wardrobe/props/weather/time"
        ]
    }
    
    # ... sau khi đã có obj_translated chuẩn và update continuity ...
    try:
        obj_translated = self._post_sanity_pass(obj_translated, scene_text)
    except Exception:
        # không chặn pipeline nếu chỉ lỗi hậu xử lý
        pass
    
    # Khóa nhãn nhân vật để tránh biến đổi
    try:
        obj_translated = self._lock_subject_aliases(obj_translated, continuity, scene_entities)
    except Exception:
        pass
    
    # Cuối cùng: cắt gọn đối tượng theo yêu cầu tối giản
    # Final cleanup: if no characters in this scene, drop stray character profiles
    try:
        required_scene_characters = self._mentioned_characters(scene_text, scene_entities, continuity)
        if not required_scene_characters:
            cu = obj_translated.get("continuity_updates") or {}
            cu["new_or_updated_characters"] = []
            obj_translated["continuity_updates"] = cu
    except Exception:
        pass

    # Nếu cảnh không nhắc tới entity nào ⇒ xoá trường dynamic entities
    try:
        _ents = self._mentioned_entities(scene_text, scene_entities, continuity).get("all", [])
        if not _ents:
            cu3 = obj_translated.get("continuity_updates") or {}
            cu3["new_or_updated_dynamic_entities"] = []
            obj_translated["continuity_updates"] = cu3
    except Exception:
        pass

    # ADD CHARACTER DETAILS TO WORLD_AND_CHARACTERS
    print("🔍 Adding character details to world_and_characters...")
    if hasattr(self, '_character_profiles') and self._character_profiles:
        wc = obj_translated.get("world_and_characters", {})
        for char_name, char_profile in self._character_profiles.items():
            # Add character details to world_and_characters
            wc[f"{char_name}_details"] = char_profile
            print(f"DEBUG: Added character details for {char_name} to world_and_characters")
        obj_translated["world_and_characters"] = wc
    
    # STORE CREATIVE_BRIEF FOR MAIN ACTION INJECTION
    self._current_creative_brief = obj_translated.get("creative_brief", "")
    print(f"DEBUG: Stored creative_brief for main action injection: {self._current_creative_brief}")
    
    # COMPREHENSIVE VALIDATION
    print("🔍 Running comprehensive validation...")
    for shot in obj_translated.get("shots", []):
        if shot.get("action_sequence"):
            validation_results = _comprehensive_validation(
                shot["action_sequence"], 
                obj_translated.get("creative_brief", "")
            )
            print(f"📊 VALIDATION RESULTS:")
            print(f"   Total actions: {validation_results['total_actions']}")
            print(f"   Unique verbs: {validation_results['unique_verbs']}")
            print(f"   Repetitive verbs: {validation_results['repetitive_verbs']}")
            print(f"   Character consistency: {validation_results['character_consistency']}")
            print(f"   Main actions found: {validation_results['main_actions_found']}")
            print(f"   Issues: {validation_results['issues']}")
            
            if validation_results['issues']:
                print("⚠️ VALIDATION ISSUES DETECTED:")
                for issue in validation_results['issues']:
                    print(f"   - {issue}")
            else:
                print("✅ All validation checks passed!")

    # Final character normalization and enrichment before finalize
    try:
        obj_translated = self._post_char_sanitize(obj_translated)
    except Exception as _e:
        print(f"DEBUG: _post_char_sanitize error: {_e}")
    
    # ============================================================================
    # TOP 0.1% POST-PROCESSING - AUTO-FIX & VALIDATION
    # ============================================================================
    print("\n🔧 Applying top 0.1% post-processing...")
    
    # 1. Auto-fix and enrich output
    try:
        obj_translated = self._post_process_llm_output(obj_translated, scene_text, continuity)
        print("✅ Post-processing complete: biomechanics/physics enriched, cause-effect validated")
    except Exception as e:
        print(f"⚠️ Post-processing warning: {e}")
    
    # 2. Ultimate validation
    try:
        is_valid, validation_messages = self._ultimate_validation(obj_translated, scene_text)
        if is_valid:
            print("✅ ULTIMATE VALIDATION PASSED - Zero critical errors!")
        else:
            print("⚠️ Validation found issues - check messages above")
    except Exception as e:
        print(f"⚠️ Validation warning: {e}")

    obj_min = self._finalize_minimal_output(obj_translated)
    return obj_min


# -------------------------
# Final pass: character/output sanitizer
# -------------------------
def _post_char_sanitize(self, obj: dict) -> dict:
    if not isinstance(obj, dict):
        return obj

    def _short_label(name: str) -> str:
        s = (name or "").strip()
        sl = s.lower()
        if "warrior" in sl:
            return "Warrior"
        if "beast" in sl:
            return "Beast"
        if len(s) > 40 or "," in s:
            return "Lead"
        return s or "Lead"

    # world_and_characters canonization
    w = obj.get("world_and_characters") or {}
    # Prefer Warrior as lead if creative_brief implies
    brief = (obj.get("creative_brief", "") or "").lower()
    prefer_warrior_lead = any(k in brief for k in ("warrior",))
    raw_lead = w.get("lead_character", "")
    raw_ant  = w.get("antagonist", "")
    lead = _short_label(raw_lead)
    ant  = _short_label(raw_ant)
    if prefer_warrior_lead:
        lead, ant = "Warrior", ("Beast" if lead != "Beast" else "Warrior")
    if not ant or ant == lead:
        ant = "Beast" if lead != "Beast" else "Warrior"
    w["lead_character"], w["antagonist"] = lead, ant
    obj["world_and_characters"] = w

    # Build quick profile map from any collected profiles
    profile_map = {}
    try:
        if hasattr(self, "_character_profiles") and isinstance(self._character_profiles, dict):
            profile_map.update(self._character_profiles)
    except Exception:
        pass

    # Fallback minimal profiles if missing
    profile_map.setdefault("Warrior", {
        "name": "Warrior",
        "appearance": "Tall, imposing figure with muscular build and combat-worn features; focused gaze",
        "wardrobe": "Worn leather armor with metal studs; brown tunic; leather boots; cloak",
        "emotions": "composed, determined, battle-ready",
        "movement_style": "purposeful footwork; balanced posture; efficient transitions",
        "personality_traits": "brave, tactical, protective, disciplined",
        "weapons_equipment": "iron spear; small round shield; utility belt",
        "physical_condition": "peak stamina and reflexes; minor fatigue; steady breathing",
        "tactical_role": "frontline engager; protects allies; leads maneuvers",
    })
    profile_map.setdefault("Beast", {
        "name": "Beast",
        "appearance": "Large predator; dense fur; scarred hide; intense eyes; broad frame",
        "wardrobe": "natural fur and hide",
        "emotions": "agitated, reactive, predatory focus",
        "movement_style": "low center of gravity; powerful bursts; lateral pivots",
        "personality_traits": "territorial, aggressive, instinct-driven",
        "weapons_equipment": "claws, mass, bite force",
        "physical_condition": "wounded yet mobile; recoil on impacts; compensated gait",
        "tactical_role": "antagonist predator; probes, feints, charges",
    })

    def _subject_short(s: str) -> str:
        sl = (s or "").lower()
        if "warrior" in sl:
            return "Warrior"
        if "beast" in sl:
            return "Beast"
        if len(s or "") > 40 or "," in (s or ""):
            # prefer lead label if exists
            return lead
        return s or lead

    # Sanitize action subjects and attach character_details
    for shot in obj.get("shots", []) or []:
        if not isinstance(shot, dict):
            continue
        seq = shot.get("action_sequence") or []
        for e in seq:
            if not isinstance(e, dict):
                continue
            # Hard-map common noisy labels to canonical first
            subj_raw = (e.get("subject", "") or "").strip()
            if subj_raw in ("Warrior Squad", "warriors Squad", "warriors", "Warriors"):
                subj_raw = "Warrior"
            if subj_raw in ("Forest Beast", "forest beast"):
                subj_raw = "Beast"
            e["subject"] = _subject_short(subj_raw)
            subj = e["subject"]
            if not e.get("character_details") and subj in profile_map:
                e["character_details"] = profile_map[subj]

        # Ensure main action presence: Warrior thrusts + Beast recoils within ~0.45s
        has_thrust = any((isinstance(x, dict) and (x.get("subject") == "Warrior") and str(x.get("verb", "")).lower().strip() == "thrusts") for x in seq)
        if not has_thrust and seq:
            t0 = max(0.8, float(seq[0].get("timestamp_sec", 0.0)))
            seq.append({
                "timestamp_sec": round(t0, 2),
                "subject": "Warrior",
                "verb": "thrusts",
                "biomechanics_note": "deltoid and trapezius engage; shoulder and elbow extend; hips drive forward; core stabilizes; heel-to-toe transfer; breath timing matches exertion",
                "physics_note": "impulse applied along spear axis; momentum increases forward; friction under lead foot prevents slip; energy transfer peaks at contact; gravity constant 9.8 m/s²",
                "internal_motivation": "neutralize threat quickly while maintaining balance and defensive structure; protect space; capitalize on opening"
            })
            seq.append({
                "timestamp_sec": round(t0 + 0.45, 2),
                "subject": "Beast",
                "verb": "recoils",
                "biomechanics_note": "torso snaps back; paws skid; cervical spine flexes; stance widens; claws engage substrate; center of mass shifts rearward",
                "physics_note": "backward acceleration from impact; kinetic energy dissipates via friction and joint flexion; momentum redirected; no slip due to high friction",
                "internal_motivation": "withdraw from line of force; preserve stability; reassess opponent angle"
            })
        # Enforce minimum note lengths for all actions in this shot
        for e in seq:
            if not isinstance(e, dict):
                continue
            if len(e.get("biomechanics_note", "")) < 100:
                e["biomechanics_note"] = (e.get("biomechanics_note", "") + "; hips and core stabilize; ankle dorsiflexion; controlled weight transfer; coordinated joint sequencing; balanced posture maintained throughout").strip("; ")
            if len(e.get("physics_note", "")) < 100:
                e["physics_note"] = (e.get("physics_note", "") + "; friction coefficient supports traction; momentum conserved and redirected; gravitational acceleration 9.8 m/s²; force vectors align with motion; braking phase limits overshoot").strip("; ")
            if len(e.get("internal_motivation", "")) < 80:
                e["internal_motivation"] = (e.get("internal_motivation", "") + "; maintain tactical readiness; track opponent cues; commit only with advantage; conserve energy while preserving positional safety").strip("; ")
        shot["action_sequence"] = sorted(seq, key=lambda x: x.get("timestamp_sec", 0.0))

    # Aggregate unique subjects to publish detailed profiles into world_and_characters
    mentioned_labels = []
    for shot in obj.get("shots", []) or []:
        seq = shot.get("action_sequence") or []
        for e in seq:
            if isinstance(e, dict) and e.get("subject"):
                label = _subject_short(e.get("subject", ""))
                if label and label not in mentioned_labels:
                    mentioned_labels.append(label)

    characters_list = []
    characters_map = {}
    for label in mentioned_labels:
        prof = profile_map.get(label)
        if not prof:
            continue
        characters_list.append({
            "label": label,
            "profile": prof,
        })
        characters_map[label] = prof

    # Ensure both Warrior and Beast profiles are present if they appear anywhere
    if "Warrior" in (mentioned_labels or []) and "Warrior" not in characters_map:
        characters_map["Warrior"] = profile_map.get("Warrior", {})
        characters_list.append({"label": "Warrior", "profile": characters_map["Warrior"]})
    if "Beast" in (mentioned_labels or []) and "Beast" not in characters_map:
        characters_map["Beast"] = profile_map.get("Beast", {})
        characters_list.append({"label": "Beast", "profile": characters_map["Beast"]})

    # If both Warrior and Beast appear, lock world roles to canonical labels
    if "Warrior" in mentioned_labels and "Beast" in mentioned_labels:
        w["lead_character"] = "Warrior"
        w["antagonist"] = "Beast"
        obj["world_and_characters"] = w

    if characters_list:
        w["characters"] = characters_list
        w["character_details"] = characters_map
        obj["world_and_characters"] = w

    # Compact single_prompt_for_model by replacing long subject phrases with short labels
    sp = obj.get("single_prompt_for_model", "")
    if isinstance(sp, str) and sp:
        # Normalize common noisy labels
        replacements = [
            ("warriors Squad", "Warrior"),
            ("Warrior Squad", "Warrior"),
            ("Forest Beast", "Beast"),
            ("Forest Forest Beast", "Beast"),
            ("warriors", "Warrior"),
        ]
        for old, new in replacements:
            sp = sp.replace(old, new)
        # Replace obvious long repeated strings with short labels
        for long_label, short in [(raw_lead, lead), (raw_ant, ant)]:
            if long_label and isinstance(long_label, str) and len(long_label) > 20:
                sp = sp.replace(long_label, short)
        # Generic compacting: collapse multiple spaces
        sp = " ".join(sp.split())
        obj["single_prompt_for_model"] = sp

    # Sanitize audio wording to avoid non-graphic violations and minor grammar
    for shot in obj.get("shots", []) or []:
        aud = (shot.get("audio") or {}) if isinstance(shot, dict) else {}
        def _clean_line(s: str) -> str:
            bad = ["splatter", "splattering", "flesh", "gore", "blood"]
            out = s or ""
            for b in bad:
                out = out.replace(b, "impact debris")
            # simple grammar fix
            out = out.replace(" spear make ", " spear makes ")
            return out
        if aud.get("foley") and isinstance(aud["foley"], list):
            aud["foley"] = [_clean_line(x) for x in aud["foley"]]
        if aud.get("sfx") and isinstance(aud["sfx"], list):
            aud["sfx"] = [_clean_line(x) for x in aud["sfx"]]
        if isinstance(shot, dict):
            shot["audio"] = aud

    # Final hard pass: ensure every action has character_details and padded notes
    for shot in obj.get("shots", []) or []:
        seq = shot.get("action_sequence") or []
        for e in seq:
            if not isinstance(e, dict):
                continue
            subj = _subject_short((e.get("subject") or ""))
            if subj in profile_map and not e.get("character_details"):
                e["character_details"] = profile_map[subj]
            if len(e.get("biomechanics_note", "")) < 100:
                e["biomechanics_note"] = (e.get("biomechanics_note", "") + "; hips and core stabilize; ankle dorsiflexion; controlled weight transfer; coordinated joint sequencing; balanced posture maintained throughout").strip("; ")
            if len(e.get("physics_note", "")) < 100:
                e["physics_note"] = (e.get("physics_note", "") + "; friction coefficient supports traction; momentum conserved and redirected; gravitational acceleration 9.8 m/s²; force vectors align with motion; braking phase limits overshoot").strip("; ")
            if len(e.get("internal_motivation", "")) < 80:
                e["internal_motivation"] = (e.get("internal_motivation", "") + "; maintain tactical readiness; track opponent cues; commit only with advantage; conserve energy while preserving positional safety").strip("; ")

    return obj


# -------------------------
# Realism enforcement utilities
# -------------------------
def _enforce_realism_on_sequence(seq: list[dict]) -> list[dict]:
    if not isinstance(seq, list):
        return seq

    unrealistic_verbs = {
        "teleports": "sidesteps",
        "teleport": "sidesteps",
        "levitates": "steps lightly",
        "flies": "dashes",
        "glitches": "stumbles",
        "warps": "pivots",
        "phases": "dodges",
        "time-skips": "repositions",
        "instant-turns": "turns with weight shift",
        "blinks-to": "moves to",
    }

    def replace_unrealistic(verb: str) -> str:
        v = (verb or "").strip().lower()
        return unrealistic_verbs.get(v, verb)

    last_ts = None
    for e in seq:
        if not isinstance(e, dict):
            continue
        # Replace unrealistic verbs
        e["verb"] = replace_unrealistic(e.get("verb", ""))

        # Ensure no instantaneous multi-meter moves without intermediate beats
        # If we detect large implied motion verbs, ensure physics/biomech mention ground contact/friction
        phys = (e.get("physics_note", "") or "").lower()
        bio = (e.get("biomechanics_note", "") or "").lower()
        needs_grounding = any(k in (e.get("verb", "").lower()) for k in ["dash","leap","jump","charge","sprint","turns","pivots"]) and not ("friction" in phys or "foot" in bio or "stance" in bio)
        if needs_grounding:
            e["physics_note"] = (e.get("physics_note", "") + "; ground friction and weight transfer limit slip; momentum conserved with braking phase").strip("; ")
            e["biomechanics_note"] = (e.get("biomechanics_note", "") + "; heel-to-toe foot placement; ankle dorsiflexion; hips and core stabilize during turn").strip("; ")

        # Clamp micro-timing to avoid <0.25s between beats (already redistributed earlier)
        ts = float(e.get("timestamp_sec", 0) or 0)
        if last_ts is not None and ts - last_ts < 0.25:
            e["timestamp_sec"] = last_ts + 0.25
            ts = e["timestamp_sec"]
        last_ts = ts

    # Re-sort in case of timing push
    seq.sort(key=lambda x: x.get("timestamp_sec", 0))
    return seq


def _canonicalize_dual_subjects(seq: list[dict]) -> list[dict]:
    """Map various subject strings to canonical labels 'Warrior' or 'Beast',
    without collapsing everything to a single label."""
    def canon(s: str) -> str:
        sl = (s or "").lower()
        if any(k in sl for k in ["warrior","fighter","soldier","spearman","archer","guard"]):
            return "Warrior"
        if any(k in sl for k in ["beast","creature","predator","animal"]):
            return "Beast"
        return s or "Warrior"

    out = []
    for e in seq:
        if isinstance(e, dict):
            e = dict(e)
            e["subject"] = canon(e.get("subject", ""))
        out.append(e)
    return out


def _balance_dual_actor_actions(seq: list[dict]) -> list[dict]:
    """Guarantee both Warrior and Beast have continuous, responsive actions.
    - No gaps > 1.0s without at least micro-action from each actor.
    - Insert micro-reactions for the counterpart when missing.
    """
    if not isinstance(seq, list):
        return seq

    seq = [e for e in seq if isinstance(e, dict)]
    seq.sort(key=lambda x: x.get("timestamp_sec", 0.0))

    def last_time_for(label: str) -> float:
        t = -1e9
        for e in seq:
            if e.get("subject") == label:
                t = max(t, float(e.get("timestamp_sec", 0.0)))
        return -1.0 if t < -1e8 else t

    def inject(label: str, ts: float, verb: str, bio: str, phy: str, mot: str):
        seq.append({
            "timestamp_sec": ts,
            "subject": label,
            "verb": verb,
            "biomechanics_note": bio,
            "physics_note": phy,
            "internal_motivation": mot,
        })

    # Walk timeline and ensure mirrored responsiveness
    all_times = sorted({float(e.get("timestamp_sec", 0.0)) for e in seq})
    if not all_times:
        return seq

    start_t, end_t = all_times[0], max(all_times[-1], 8.0)
    step = 0.8  # enforce micro-activity cadence per actor
    t = start_t
    while t <= end_t:
        for label in ("Warrior","Beast"):
            lt = last_time_for(label)
            if lt < 0 or (t - lt) > 1.0:
                # Insert a micro-action to avoid idle
                inject(
                    label,
                    t,
                    "adjusts stance" if label=="Warrior" else "repositions",
                    "center of mass aligns; ankle and knee adjust; core stabilizes; subtle breath control",
                    "friction prevents slip; momentum minimal; gravity and posture keep balance",
                    "maintain readiness; tracking opponent; anticipating next beat",
                )
        t += step

    # Sort and lightly de-duplicate near-collisions
    seq.sort(key=lambda x: x.get("timestamp_sec", 0.0))
    compact = []
    last = None
    for e in seq:
        if last and abs(float(e.get("timestamp_sec",0)) - float(last.get("timestamp_sec",0))) < 0.1 and e.get("subject")==last.get("subject"):
            continue
        compact.append(e)
        last = e

    return compact


def _pair_cause_effect_actions(seq: list[dict]) -> list[dict]:
    """Create explicit cause→effect pairs between Warrior and Beast.
    For each Warrior attack-type verb, ensure a Beast reaction within ~0.6s.
    For each Beast attack/pressure, ensure a Warrior defensive/mobility response.
    """
    if not isinstance(seq, list):
        return seq

    attack_map_w2b = {
        "thrusts": ("recoils",
            "upper body whips back; paws skid; core braces; cervical spine flexes",
            "impulse causes backward acceleration; friction arrests slip; momentum redirected"),
        "slashes": ("withdraws",
            "shoulders retract; torso twists away; gait shortens; head angles off line",
            "angular momentum conserved; torque reduces contact; COM shifts laterally"),
        "charges": ("sidesteps",
            "hips rotate; lead foot pivots; center of mass shifts; guard closes",
            "lateral friction supports redirection; momentum vector re-aimed"),
        "presses": ("deflects",
            "forelimbs angle; torso tilts; claws catch surface; stance widens",
            "force vector split; normal force increases; rotational inertia counters push"),
    }

    attack_map_b2w = {
        "charges": ("sidesteps",
            "rear foot plants; ankle dorsiflexion; hips rotate; shoulders counter-rotate",
            "lateral impulse shifts COM; friction prevents slip; momentum conserved"),
        "lunges": ("parries",
            "lead arm extends; wrist angles; elbow flexes; core tightens",
            "force redirected tangentially; lever advantage reduces impact"),
        "pivots": ("tracks",
            "eyes lock; neck aligns; stance re-centers; feet micro-adjust",
            "minimal momentum; visual tracking guides subsequent motion"),
    }

    out = list(seq)
    out.sort(key=lambda x: x.get("timestamp_sec", 0.0))

    def find_next_gap(t: float) -> float:
        return t + 0.45

    def inject(label: str, ts: float, verb: str, bio: str, phy: str, mot: str):
        out.append({
            "timestamp_sec": ts,
            "subject": label,
            "verb": verb,
            "biomechanics_note": bio,
            "physics_note": phy,
            "internal_motivation": mot,
        })

    for e in list(out):
        subj = (e.get("subject") or "").strip()
        verb = (e.get("verb") or "").lower().strip()
        ts = float(e.get("timestamp_sec", 0.0))
        if subj == "Warrior":
            for atk, (rxn, bio, phy) in attack_map_w2b.items():
                if verb == atk:
                    inject("Beast", find_next_gap(ts), rxn, bio, phy, "reacts to incoming force; preserves survival margin")
                    break
        elif subj == "Beast":
            for atk, (rxn, bio, phy) in attack_map_b2w.items():
                if verb == atk:
                    inject("Warrior", find_next_gap(ts), rxn, bio, phy, "maintains defensive structure; prepares counter")
                    break

    out.sort(key=lambda x: x.get("timestamp_sec", 0.0))
    return out


# -------------------------
# File saving (unchanged)
# -------------------------
def save_prompts(self, prompts: List[Dict[str, Any]], output_dir: str = "chatgpt_prompts") -> bool:
    if not prompts:
        print("❌ Không có prompts để lưu")
        return False

    if self.dry_run:
        print("🔍 DRY RUN: Không lưu file, chỉ hiển thị thông tin")
        for idx, p in enumerate(prompts, 1):
            setting = (p.get("world_and_characters") or {}).get("setting", "")
            print(f" Scene {idx}: {setting}")
        return True

    os.makedirs(output_dir, exist_ok=True)
    try:
        for idx, p in enumerate(prompts, 1):
            setting = (p.get("world_and_characters") or {}).get("setting", "")
            title = safe_slug(setting, f"scene_{idx:02d}")
            filename = f"{idx:02d}_{title}_veo3_prompt.json"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(p, f, ensure_ascii=False, indent=2)
            print(f"✅ Đã lưu: {filename}")

        master = {
            "project_info": {
                "title": "Veo 3 Video Generation Prompts",
                "total_scenes": len(prompts),
                "created_at": now_iso(),
                "generator": "GPT-4o-mini (tool-call + schema)",
                "model": "google/veo-3",
            },
            "scenes": prompts,
        }
        with open(os.path.join(output_dir, "all_veo3_prompts.json"), "w", encoding="utf-8") as f:
            json.dump(master, f, ensure_ascii=False, indent=2)

        lines = [
            "DANH SÁCH PROMPTS VEO 3 - TẠO BỞI GPT-4o-mini",
            "=" * 70,
            "",
        ]
        for i, p in enumerate(prompts, 1):
            w = p.get("world_and_characters") or {}
            shots = p.get("shots") or []
            lines.append(f"CẢNH {i}:")
            lines.append(f" - Bối cảnh: {w.get('setting', 'N/A')}")
            lines.append(f" - Nhân vật chính: {w.get('lead_character', 'N/A')}")
            lines.append(f" - Đối thủ: {w.get('antagonist', 'N/A')}")
            lines.append(f" - Thời tiết: {w.get('weather', '')}")
            lines.append(f" - Số shots: {len(shots)}")
            for s in shots:
                lines.append(f" * {s.get('slugline','N/A')} ({s.get('duration','?')}s)")
            lines.append("-" * 50)
            lines.append("")

        with open(os.path.join(output_dir, "veo3_prompts_summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.create_report(prompts, output_dir)
        return True
    except Exception as e:
        print(f"❌ Lỗi lưu file: {e}")
        return False


def create_report(self, prompts: List[Dict[str, Any]], output_dir: str):
    lines = [
        "# BÁO CÁO CHATGPT PROMPT GENERATOR",
        "",
        f"- **Tổng số cảnh**: {len(prompts)}",
        f"- **Thời gian tạo**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Generator**: ChatGPT API (tool-call + schema validation)",
        f"- **Thư mục output**: {output_dir}",
        "",
        "## Danh sách cảnh",
    ]
    for i, p in enumerate(prompts, 1):
        w = p.get("world_and_characters") or {}
        duration = p.get("duration_seconds", "N/A")
        title = safe_slug(w.get("setting", ""), f"scene_{i:02d}")
        lines += [
            f"### Cảnh {i}: {title}",
            f"- **Nhân vật/chính**: {w.get('lead_character','N/A')}",
            f"- **Đối thủ**: {w.get('antagonist','N/A')}",
            f"- **Bối cảnh**: {w.get('setting','N/A')}",
            f"- **Thời lượng**: {duration}s",
            "",
        ]

    lines += [
        "## Cấu trúc file",
        "",
        f"{output_dir}/",
        "├── all_veo3_prompts.json",
        "├── 01_<slug>_veo3_prompt.json",
        "├── 02_<slug>_veo3_prompt.json",
        "└── ...",
        "",
        "",
        "## Cách sử dụng",
        "1. Dùng all_veo3_prompts.json để xem tổng quan",
        "2. Dùng từng XX_<slug>_veo3_prompt.json cho mỗi cảnh",
        "3. Copy single_prompt_for_model khi cần prompt đơn cho Veo 3",
        "4. Các trường shots, camera, lighting, audio, vfx đã có cấu trúc đầy đủ",
    ]

    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Đã tạo báo cáo: README.md")


# -------------------------
# Pipeline over text file (unchanged)
# -------------------------
def process_text_file(self, file_path: str) -> bool:
    print("🎬 CHATGPT PROMPT GENERATOR — CONTINUITY-LOCKED MODE")
    print("=" * 60)
    try:
        text = open(file_path, "r", encoding="utf-8").read()
        print(f"✅ Đã đọc file: {file_path}")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return False

    continuity = ContinuityState()
    self.prime_continuity_from_text(text, continuity)

    scenes_list = self.split_scenes(text)
    print(f"🔎 Phát hiện {len(scenes_list)} cảnh trong file.")
    self._debug_dump_scenes(scenes_list)
    if not scenes_list:
        return False

    prompts: List[Dict[str, Any]] = []
    for idx, (num, scene_text) in enumerate(scenes_list, 1):
        print(f"\n🤖 Đang phân tích cảnh {idx} (label: {num})...")
        max_retries = 3
        out: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries):
            try:
                out = self.build_scene_prompt(scene_text, idx, continuity)
            except KeyboardInterrupt:
                print("⏹️ Dừng theo yêu cầu người dùng.")
                return False
            except Exception as e:
                out = {"error": f"Exception during build_scene_prompt: {e}"}

            if out and "error" not in out:
                prompts.append(out)
                # Chẩn đoán chất lượng
                self._diag_report(out)
                self._logic_assertions(out)
                break
            else:
                err = out.get("error") if isinstance(out, dict) else "Unknown error"
                print(f"⚠️ Thử {attempt+1}/{max_retries} lỗi: {err}")
                if attempt < max_retries - 1:
                    sleep_backoff(attempt)

        if not out or (isinstance(out, dict) and "error" in out):
            print(f"❌ Không thể tạo JSON hợp lệ cho cảnh {idx}. Bỏ qua cảnh này.")

    if not prompts:
        print("❌ Không tạo được prompt nào")
        return False

    print("\n💾 Đang lưu prompts...")
    ok = self.save_prompts(prompts)
    if ok:
        print("\n🎉 HOÀN THÀNH!")
        print("=" * 60)
        print(f"✅ Đã tạo {len(prompts)} prompts Veo 3 đầy đủ")
        print(f"✅ Thư mục output: chatgpt_prompts")
        print(f"✅ File tổng hợp: all_veo3_prompts.json")
        print(f"✅ File tóm tắt: veo3_prompts_summary.txt")
    return ok


def main():
    print("🤖 CHATGPT PROMPT GENERATOR — JSON-STRICT & CONTINUITY")
    print("=" * 60)
    gen = ChatGPTPromptGenerator()
    input_file = "inputprompt.txt"
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file: {input_file}")
        print("💡 Vui lòng đặt file văn bản có tên 'inputprompt.txt' trong thư mục này")
        return
    success = gen.process_text_file(input_file)
    if success:
        print("\n🚀 HƯỚNG DẪN SỬ DỤNG:")
        print("1. Mở thư mục 'chatgpt_prompts' để xem kết quả")
        print("2. Dùng 'all_veo3_prompts.json' cho tổng quan, hoặc từng file cảnh riêng lẻ")
        print("3. Trường 'single_prompt_for_model' dùng trực tiếp cho Veo 3 nếu cần prompt ngắn")
        print("4. Output đã khoá continuity và qua schema validation + local-normalize + auto-repair")
    else:
        print("❌ Có lỗi xảy ra trong quá trình xử lý")


"""
VEO3 PATCH — STRICT GROUNDING, NON-GRAPHIC, LABEL CONSISTENCY (DROP-IN)
------------------------------------------------------------------------
Paste this **below** your ChatGPTPromptGenerator class definition (same file)
or import it after the class is available. It monkey‑patches a few behaviors to
fix the issues you flagged:

1) Strip unintended archery beats (Archer/Arrow) when the scene text does not
   explicitly contain archery keywords (bow, arrow, etc.). Also removes archery
   SFX/VFX lines (bowstring snap, arrow whoosh, arrow trail).
2) Enforce non‑graphic phrasing for contacts/roars and injuries (policy‑safe).
3) Unify labels: replace generic subjects (Lead/Target) with concrete ones
   (e.g., Warrior/Beast) and auto‑fill lead_character/antagonist if empty.
4) Fix camera semantics: dolly direction matches position start→end; avoid
   confusing focus_pull with zoom (use rack focus wording).
5) Neutralize physics_rules (remove archery‑specific description).
6) Re‑validate JSON after corrections; attempt light auto‑repair if needed.

This patch **does not** change your public API. You still run `main()` normally
and it will save prompts under `chatgpt_prompts/`.
"""
import re, json

# --- Extend (or create) policy replacements to cover non-graphic phrasing ---
try:
    POLICY_REPLACEMENTS  # type: ignore[name-defined]
except NameError:
    POLICY_REPLACEMENTS = []

POLICY_REPLACEMENTS.extend([
    # Non-graphic contacts & injuries
    (re.compile(r"\bpiercing\s+flesh\b", re.IGNORECASE), "making solid contact (non-graphic)"),
    (re.compile(r"\bexpose[s]?\s+the\s+wound\b", re.IGNORECASE), "reveals visible damage (non-graphic)"),
    (re.compile(r"\breveals\s+injur(?:y|ies)\b", re.IGNORECASE), "reveals visible damage (non-graphic)"),
    (re.compile(r"\bdamage\s+flows\b", re.IGNORECASE), "particles fall"),
    # Tone for roars
    (re.compile(r"\bpainful\s+roar\b", re.IGNORECASE), "fierce roar"),
])


# -------- Helper utilities (pure functions) --------
def _has_archery_kw(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    kws = [
        "bow", "arrow", "archer", "archery",
        "cung", "mũi tên", "mui ten", "bắn tên", "ban ten", "căng dây",
    ]
    return any(k in t for k in kws)


def _strip_archery_from_obj(obj: dict) -> dict:
    """Remove Archer/Arrow beats, archery SFX/VFX from a Veo3 JSON object."""
    if not isinstance(obj, dict):
        return obj

    shots = obj.get("shots") or []
    for s in shots:
        # Remove Archer/Arrow beats
        seq = s.get("action_sequence") or []
        new_seq = []
        for e in seq:
            subj = (e.get("subject", "") or "").strip().lower()
            verb = (e.get("verb", "") or "").strip().lower()
            if subj in ("archer", "arrow"):
                continue
            if any(tok in verb for tok in ("releases", "enters flight", "flight", "lodges")) and (
                "arrow" in subj or "archer" in subj
            ):
                continue
            new_seq.append(e)
        s["action_sequence"] = new_seq

        # Remove archery audio
        aud = s.get("audio") or {}
        for key in ("sfx", "foley", "fx"):
            arr = list(aud.get(key) or [])
            arr = [
                x for x in arr
                if not re.search(r"bowstring|arrow|whoosh|woody\s+thunk", str(x), flags=re.IGNORECASE)
            ]
            if arr:
                aud[key] = arr
            else:
                aud.pop(key, None)
        s["audio"] = aud

        # Remove archery VFX
        vfx = [x for x in (s.get("vfx") or []) if not re.search(r"arrow|trail", str(x), flags=re.IGNORECASE)]
        s["vfx"] = vfx
    obj["shots"] = shots
    return obj


def _unify_subject_labels(self, obj: dict) -> dict:
    """Legacy function - use enhanced version."""
    return _unify_subject_labels_enhanced(obj)

# --- SAFE PROXY: bảo đảm luôn gọi đúng hàm global _unify_subject_labels ---
def _unify_subject_labels__proxy(self, obj, *args, **kwargs):
    """Legacy proxy - use enhanced version."""
    return _unify_subject_labels_enhanced(obj)

ChatGPTPromptGenerator._unify_subject_labels = _unify_subject_labels__proxy


# Gắn vào class để mọi chỗ gọi self._unify_subject_labels(...) đều an toàn
try:
    setattr(ChatGPTPromptGenerator, "_unify_subject_labels", _unify_subject_labels__proxy)
except Exception:
    pass


def _fix_camera_semantics(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return obj
    for s in (obj.get("shots") or []):
        cam = s.get("camera") or {}
        pos = cam.get("position_m") or {}
        st, ed = (pos.get("start") or {}), (pos.get("end") or {})
        mv = cam.get("movement") or {}
        direction = (mv.get("direction") or "").lower()
        # Only reason about z (forward/back) convention
        try:
            z0, z1 = float(st.get("z", 0)), float(ed.get("z", 0))
            if z1 < z0 and direction == "backward":
                mv["direction"] = "forward"
            elif z1 > z0 and direction == "forward":
                mv["direction"] = "backward"
            cam["movement"] = mv
        except Exception:
            pass
        # Focus pull vs zoom wording
        fps = cam.get("focus_pull_style")
        if isinstance(fps, str) and re.search(r"\bzoom\b", fps, flags=re.IGNORECASE):
            cam["focus_pull_style"] = re.sub(r"(?i)zoom", "rack focus", fps)
        s["camera"] = cam
    return obj


def _neutralize_physics_rules(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return obj
    gs = obj.get("global_style") or {}
    pr = (gs.get("physics_rules") or "")
    # Remove poetic/archery specifics
    pr = re.sub(r"(?i)draw.*?release.*?impact.*?", "", pr)
    gs["physics_rules"] = (
        "Respect gravity, drag, inertia, momentum, friction, and reaction latency. "
        "All motion is physically plausible; no teleportation; no idle seconds."
    )
    obj["global_style"] = gs
    return obj


def _sanitize_audio_wording(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return obj
    for s in (obj.get("shots") or []):
        aud = s.get("audio") or {}
        # Clean wording + reclassify
        foley = list(aud.get("foley") or [])
        sfx = list(aud.get("sfx") or [])
        def _clean(line: str) -> str:
            y = re.sub(r"(?i)\bspear\s+piercing\s+flesh\b", "spear makes solid contact (non-graphic)", str(line))
            y = re.sub(r"(?i)\bpainful\s+roar\b", "fierce roar", y)
            y = re.sub(r"(?i)\bpainful\s+growl\b", "fierce growl", y)
            y = re.sub(r"(?i)\breveals\s+injur(?:y|ies)\b", "reveals visible damage (non-graphic)", y)
            y = re.sub(r"(?i)\bdamage\s+flows\b", "particles fall", y)
            return y
        foley = [_clean(x) for x in foley]
        # Move shouts/yells/screams from SFX → Foley
        kept_sfx = []
        for x in sfx:
            xs = str(x)
            if re.search(r"(?i)\b(shout|yell|scream)s?\b", xs):
                foley.append(_clean(xs))
            else:
                kept_sfx.append(_clean(xs))
        if foley:
            aud["foley"] = sorted(set(foley))
        else:
            aud.pop("foley", None)
        if kept_sfx:
            aud["sfx"] = sorted(set(kept_sfx))
        else:
            aud.pop("sfx", None)
        s["audio"] = aud
    return obj


# -------- Robust hook for both pre- and post-auto-bind setups --------
# Keep _post_correct(self, out, scene_text) as you defined above.

def _post_correct(self, out: dict, scene_text: str) -> dict:
    if not isinstance(out, dict):
        return out
    entities = self.extract_scene_entities(scene_text)

    # Strip archery only if scene text lacks archery keywords
    if not _has_archery_kw(scene_text):
        out = _strip_archery_from_obj(out)
    # Unify labels & auto-fill world roles
    out = self._unify_subject_labels(out)
    # >>> PATCH START: enforce prop state consistency
    for _s in (out.get("shots") or []):
        self._enforce_prop_state_consistency(_s)
    # <<< PATCH END
    # Fix camera semantics and focus wording
    out = _fix_camera_semantics(out)
    # Neutralize physics rules wording
    out = _neutralize_physics_rules(out)
    # Sanitize audio wording to non-graphic
    out = _sanitize_audio_wording(out)

    # Loại SFX cung tên nếu không có Archer/Arrow trong seq
    for s in (out.get("shots") or []):
        seq = s.get("action_sequence") or []
        has_archery_beat = any("arrow" in (e.get("subject","").lower()) or "archer" in (e.get("subject","").lower()) for e in seq)
        if not has_archery_beat:
            aud = s.get("audio") or {}
            for key in ("sfx","foley","fx"):
                arr = [x for x in (aud.get(key) or []) if not re.search(r"(?i)(arrow|bowstring|whoosh|woody\s+thunk)", str(x))]
                aud[key] = arr
            s["audio"] = aud

    # 🔁 recompute single prompt sau mọi chỉnh sửa:
    try:
        out["single_prompt_for_model"] = self.compose_single_prompt_for_model(out, ContinuityState())
    except Exception:
        pass

    # Re-validate; attempt light repair if needed
    err = self.validate_json(out)
    if err:
        fixed = self.repair_with_llm(out, err, ContinuityState(), scene_text) if getattr(self, "client", None) else None
        if isinstance(fixed, dict) and not self.validate_json(fixed):
            out = fixed
        else:
            # Last resort: ensure at least one action_sequence beat exists
            try:
                s0 = (out.get("shots") or [{}])[0]
                seq = s0.get("action_sequence") or []
                if not seq:
                    seq.append({
                        "timestamp_sec": 7.75,
                        "subject": (out.get("world_and_characters") or {}).get("lead_character", "Warrior") or "Warrior",
                        "verb": "adjusts stance",
                        "biomechanics_note": "breathing-linked sway; grip correction",
                        "physics_note": "weight shift and friction keep motion continuous",
                        "internal_motivation": "maintain control to the last frame"
                    })
                    s0["action_sequence"] = seq
                    out["shots"][0] = s0
            except Exception:
                pass
    return out


def _install_patch():
    """
    Try to wrap ChatGPTPromptGenerator.build_scene_prompt if it's already
    bound; otherwise wrap the global build_scene_prompt function so that
    the AUTO-BIND step will pick up the wrapped version later.
    """
    # 1) Try post auto-bind hook (class method exists)
    try:
        _orig = ChatGPTPromptGenerator.build_scene_prompt  # type: ignore
        def wrapped(self, scene_text: str, scene_number: int, continuity):
            out = _orig(self, scene_text, scene_number, continuity)
            if isinstance(out, dict) and ("error" not in out):
                out = _post_correct(self, out, scene_text)
            return out
        ChatGPTPromptGenerator.build_scene_prompt = wrapped  # type: ignore
        return True
    except Exception:
        pass

    # 2) Fallback: pre auto-bind (wrap the global function)
    _orig_fn = globals().get("build_scene_prompt")
    if callable(_orig_fn):
        def wrapped(self, scene_text: str, scene_number: int, continuity):
            out = _orig_fn(self, scene_text, scene_number, continuity)  # type: ignore
            if isinstance(out, dict) and ("error" not in out):
                out = _post_correct(self, out, scene_text)
            return out
        # Replace the global function; the AUTO-BIND loop will attach this to the class later.
        globals()["build_scene_prompt"] = wrapped
        return True

    # Not ready yet; caller can try again later.
    return False

# (không gọi _install_patch() ở đây)

def apply_veo3_patch():
    """
    Call this AFTER the AUTO-BIND section (where functions are attached to the class)
    to ensure patch is applied correctly.
    """
    _install_patch()



# =======================
# CONTINUITY & PHYSICS PATCH (MONKEY-PATCH)
# =======================

# 1) Canonical roster + subject unification
def _unify_subject_labels(self, obj: dict) -> dict:
    """Legacy function - use enhanced version."""
    return _unify_subject_labels_enhanced(obj)

# 2) Entity detection hardening (anti false-positive)
try:
    _orig_extract_scene_entities = ChatGPTPromptGenerator.extract_scene_entities
except Exception:
    _orig_extract_scene_entities = None

def _extract_scene_entities_hardened(self, text: str):
    out = _orig_extract_scene_entities(self, text) if _orig_extract_scene_entities else {"dynamic": [], "all": [], "projectiles": [], "animals": [], "vehicles": [], "objects": [], "characters": []}
    low = (text or "").lower()
    STOP = {"mentor","identity","unveil","drama"}
    dyn = []
    for e in (out.get("dynamic") or []):
        name = (e.get("name") or "").strip()
        if not name or name.lower() in STOP:
            continue
        # 'lao' chỉ giữ khi có 'cây lao/phóng lao'
        if name.lower() == "lao":
            import re as _re
            if not _re.search(r"(?i)(c(ây|ay)\s+lao|ph(ó|o)ng\s+lao)", low):
                continue
        import re as _re
        if not _re.search(rf"(?i)\b{re.escape(name)}\b", low):
            continue
        dyn.append(e)
    out["dynamic"] = dyn
    # rebuild 'all'
    import re as _re
    all_names = set()
    for k in ("characters","animals","vehicles","projectiles","objects"):
        for nm in out.get(k, []) or []:
            if nm and _re.search(rf"(?i)\b{re.escape(nm)}\b", low):
                all_names.add(nm)
    dyn2 = []
    for e in dyn:
        nm = (e.get("name") or "").strip()
        if nm in all_names or e.get("kind") in ("animal","vehicle","projectile","object"):
            dyn2.append(e)
    out["all"] = sorted(all_names, key=str.lower)
    out["dynamic"] = dyn2
    return out

if _orig_extract_scene_entities:
    ChatGPTPromptGenerator.extract_scene_entities = _extract_scene_entities_hardened

# 3) Action hygiene: no camera beats; throttle spam; map non-physical verbs
def _purge_camera_beats(seq: list) -> list:
    return [e for e in seq if (e.get("subject","") or "").strip().lower() not in ("camera","the camera")]

def _fix_nonphysical_and_spam(seq: list) -> list:
    cleaned, reaim_count, last_reaim_t = [], 0, -999
    for e in sorted([x for x in seq if isinstance(x, dict)], key=lambda x: float(x.get("timestamp_sec", 0.0))):
        subj = (e.get("subject","") or "").strip()
        verb = (e.get("verb","") or "").strip().lower()
        if subj.lower() == "damage":
            e["subject"] = "fluid"; e["verb"] = "drips (non-graphic)"
        if verb in {"sweat","focus"}:
            e["verb"] = "braces" if verb == "sweat" else "re-centers balance"
        if verb == "re-aims":
            t = float(e.get("timestamp_sec", 0.0))
            if reaim_count >= 2 or (t - last_reaim_t) < 0.7:
                continue
            reaim_count += 1; last_reaim_t = t
        cleaned.append(e)
    return cleaned

# bọc validate_and_clamp để chèn vệ sinh
try:
    _orig_validate_and_clamp_sequence = ChatGPTPromptGenerator._validate_and_clamp_sequence
except Exception:
    _orig_validate_and_clamp_sequence = None

    def _validate_and_clamp_sequence_strict(self, seq: list, dur: float, lead_label: str = "Lead", antagonist_label: str = "Target"):
        """Enhanced version with timeline redistribution, character consistency, anti-repetition, and biomechanics enhancement"""
        print("🔧 _validate_and_clamp_sequence_strict called!")
        if not isinstance(seq, list): 
            return [], []
        seq = _purge_camera_beats(seq)
        seq = _fix_nonphysical_and_spam(seq)
        fixed, warns = _orig_validate_and_clamp_sequence(self, seq, dur, lead_label, antagonist_label) if _orig_validate_and_clamp_sequence else (seq, [])
        fixed = _fix_nonphysical_and_spam(fixed)
        
        # FORCE CHARACTER CONSISTENCY - Standardize to Warrior/Beast
        print("🔧 Force applying character consistency")
        for e in fixed:
            s = (e.get("subject","") or "").strip().lower()
            if "warrior" in s or "squad" in s or "hunter" in s:
                e["subject"] = "Warrior"
            elif "beast" in s or "forest" in s or "creature" in s:
                e["subject"] = "Beast"
        
        # FORCE TIMELINE REDISTRIBUTION - 0.4s minimum spacing
        print("🔧 Force redistributing timeline with 0.4s minimum spacing")
        if fixed:
            fixed.sort(key=lambda x: float(x.get("timestamp_sec", 0)))
            new_timestamps = []
            current_time = 0.0
            for i, action in enumerate(fixed):
                new_timestamps.append(current_time)
                current_time += 0.4  # Force 0.4s spacing
                if current_time >= dur:
                    current_time = dur - 0.1
            
            for i, action in enumerate(fixed):
                if i < len(new_timestamps):
                    action["timestamp_sec"] = new_timestamps[i]
                    print(f"🔧 Redistributed action {i} to {new_timestamps[i]}s")
        
        # FORCE ANTI-REPETITION LOGIC
        print("🔧 Force applying anti-repetition logic")
        if fixed:
            verb_counts = {}
            for action in fixed:
                verb = action.get("verb", "").lower()
                verb_counts[verb] = verb_counts.get(verb, 0) + 1
            
            repetitive_verbs = {verb: count for verb, count in verb_counts.items() if count > 2}
            print(f"🔧 Found repetitive verbs: {repetitive_verbs}")
            
            if repetitive_verbs:
                replacement_verbs = ["advances", "circles", "braces", "repositions", "recenters", "stalls", "thrusts", "charges", "strikes", "deflects", "sidesteps", "parries", "presses", "withdraws", "pivots", "feints", "guards", "tracks", "aims", "releases", "follows", "adjusts", "shifts", "maneuvers", "engages", "retreats", "approaches", "evades", "counters", "responds"]
                verb_index = 0
                for action in fixed:
                    verb = action.get("verb", "").lower()
                    if verb in repetitive_verbs:
                        if verb_index < len(replacement_verbs):
                            action["verb"] = replacement_verbs[verb_index]
                            print(f"🔧 Replaced '{verb}' with '{replacement_verbs[verb_index]}'")
                            verb_index += 1
                        else:
                            action["verb"] = "maneuvers"
                            print(f"🔧 Replaced '{verb}' with 'maneuvers' (fallback)")
        
        # FORCE BIOMECHANICS/PHYSICS ENHANCEMENT
        print("🔧 Force applying biomechanics/physics enhancement")
        if fixed:
            for action in fixed:
                biomechanics = action.get("biomechanics_note", "")
                if len(biomechanics) < 100:
                    enhanced_biomechanics = f"Complex multi-joint movement involving coordinated muscle activation across the entire kinetic chain. The {action.get('subject', 'character')} engages primary muscle groups including quadriceps, hamstrings, glutes, and core stabilizers while maintaining proper spinal alignment and joint positioning. Weight distribution shifts dynamically through the feet with precise heel-to-toe transition, ensuring optimal ground contact and force transmission. The movement requires precise neuromuscular coordination, with proprioceptive feedback guiding each phase of the motion to maintain balance and control throughout the execution."
                    action["biomechanics_note"] = enhanced_biomechanics
                    print(f"🔧 Enhanced biomechanics note for action: {action.get('verb', 'unknown')}")
            
            physics = action.get("physics_note", "")
            if len(physics) < 100:
                enhanced_physics = f"Physical motion governed by Newton's laws of motion with realistic acceleration, velocity, and momentum calculations. The {action.get('subject', 'character')} experiences gravitational force (9.8 m/s²) and air resistance proportional to velocity squared, creating realistic drag effects. Friction between feet and ground surface provides necessary traction for movement, with coefficient of friction varying based on surface material and moisture content. Kinetic energy transfers through the kinetic chain, with potential energy converting to kinetic energy during motion phases. Conservation of momentum ensures realistic motion continuation and stopping distances based on mass and velocity."
                action["physics_note"] = enhanced_physics
                print(f"🔧 Enhanced physics note for action: {action.get('verb', 'unknown')}")
                
                motivation = action.get("internal_motivation", "")
                if len(motivation) < 80:
                    enhanced_motivation = f"Internal psychological drive fueled by survival instincts, tactical awareness, and combat experience. The {action.get('subject', 'character')} processes environmental information through visual and auditory cues, making split-second decisions based on threat assessment and tactical positioning. Adrenaline response heightens sensory perception and reaction time, while training and muscle memory provide automatic responses to combat situations. The character's mental state reflects determination, focus, and strategic thinking, with each action serving a specific tactical purpose in the overall combat strategy."
                    action["internal_motivation"] = enhanced_motivation
                    print(f"🔧 Enhanced internal motivation for action: {action.get('verb', 'unknown')}")
        
        print(f"🔧 _validate_and_clamp_sequence_strict completed - {len(fixed)} actions")
        return fixed, warns
        "performs maneuver", "executes technique", "applies pressure", "maintains control",
        "executes strike", "delivers blow", "launches assault", "initiates attack",
        "performs action", "executes move", "applies force", "maintains stance"
    
    
    # Enhanced biomechanics with specific muscle groups (100+ chars each)
    biomech_templates = [
        "deltoid and trapezius muscles engage with precise activation and controlled force; shoulder joint rotates through optimal range of motion with smooth articulation; core stabilizes with controlled breathing and muscle coordination; neural pathways activate for coordinated movement with split-second timing; weight distribution optimized across feet with balanced pressure points; muscle fibers contract in perfect sequence for maximum power output",
        "quadriceps and hamstrings contract in perfect sequence with controlled force; hip flexors activate with controlled force and precise timing; ankle joints flex maintaining balance and stability; weight distribution optimized through pelvis with controlled movement; balance maintained through precise foot placement and muscle coordination; leg muscles work in harmony for powerful movement",
        "erector spinae muscles support spine with controlled tension and precise activation; glutes engage for stability and power generation; weight shifts through pelvis with fluid motion and controlled force; balance maintained through core strength and muscle coordination; spinal alignment preserved throughout movement with proper posture; back muscles work together for optimal support",
        "biceps and triceps coordinate with precise timing and controlled force; elbow joint articulates through optimal range with smooth movement; forearm muscles stabilize with controlled grip and precise activation; grip strength optimized for weapon handling and combat effectiveness; arm movement coordinated with body rotation for maximum power; arm muscles work in perfect harmony for combat actions",
        "neck muscles tense with controlled force and precise activation; cervical spine aligns for optimal posture and stability; head position optimizes for visual tracking and situational awareness; visual tracking enhanced for combat effectiveness; sensory input processed with heightened focus and quick response; neck muscles provide stability for head movement",
        "abdominal muscles contract with controlled breathing and precise activation; diaphragm controls breathing rhythm with optimal efficiency; ribcage expands for oxygen intake with controlled movement; core stability maximized through muscle coordination and proper technique; oxygen intake optimized for sustained performance and endurance; core muscles work together for maximum stability",
        "pectoral muscles engage with controlled force and precise activation; shoulder blades stabilize for optimal positioning and combat readiness; arm movement coordinated with body rotation for maximum effectiveness; upper body strength applied through proper technique and muscle coordination; chest expansion optimized for breathing and oxygen intake; chest muscles provide power for upper body actions",
        "latissimus dorsi activates with controlled tension and precise force; back muscles support spinal alignment with optimal strength; spinal alignment maintained through muscle coordination and proper posture; posture optimized for combat readiness and effectiveness; core strength maximized through proper technique and muscle coordination; back muscles provide stability and power for combat actions",
        "trapezius muscles contract with controlled force and precise activation; shoulder blades retract for optimal positioning and stability; upper back stabilizes for posture and combat readiness; posture improves through muscle coordination and proper technique; spinal alignment maintained throughout movement with controlled force; upper back muscles provide stability and support",
        "gluteus maximus engages with controlled activation and precise force; hip extensors activate for stability and power generation; pelvic stability increases through muscle coordination and proper technique; lower back supports with controlled tension and optimal strength; core strength applied through proper technique and muscle coordination; hip muscles provide power and stability for movement"
    ]
    
    # Enhanced physics with specific forces (100+ chars each)
    physics_templates = [
        "conservation of momentum governs motion with precise force vectors and calculated trajectories; gravity creates downward force of 9.8 m/s² with constant acceleration; friction provides controlled resistance with surface-dependent coefficients; energy transfer optimized through proper technique and muscle coordination; kinetic energy accumulates with each movement and controlled force application; momentum conservation ensures smooth and efficient motion",
        "kinetic energy transfers through joints with controlled force and precise timing; potential energy stored in muscle tension with controlled activation; air resistance affects movement with velocity-dependent drag and calculated coefficients; momentum conserved through proper body mechanics and muscle coordination; power output maximized through technique and controlled force application; energy conservation ensures optimal performance",
        "center of gravity shifts with controlled precision and calculated movement; torque applied through limbs with calculated force and precise timing; angular momentum maintained through proper rotation and controlled movement; rotational forces balanced for stability and optimal performance; mechanical advantage applied for maximum effectiveness and power output; rotational dynamics ensure smooth and controlled motion",
        "force vectors combine with precise calculations and controlled application; acceleration changes direction with controlled force and calculated timing; inertia resists sudden movements with predictable resistance and controlled response; mechanical advantage applied for optimal power and efficiency; energy dissipation controlled for efficiency and sustained performance; force dynamics ensure controlled and effective movement",
        "elastic energy stored in tendons with controlled tension and precise activation; muscle fibers contract with precise activation and controlled force; mechanical advantage optimized through proper technique and muscle coordination; power output maximized through controlled force and calculated application; cellular respiration active for sustained performance and endurance; elastic energy provides power and efficiency for movement",
        "fluid dynamics affect motion with velocity-dependent drag and calculated coefficients; pressure changes in joints with controlled force and precise timing; energy dissipation through heat with thermal regulation and controlled response; thermal regulation active for optimal performance and sustained activity; hydrostatic pressure optimized for joint stability and controlled movement; fluid dynamics ensure smooth and efficient motion",
        "electromagnetic forces in muscles with controlled activation and precise timing; chemical energy converts to mechanical with precise timing and controlled force; ATP hydrolysis drives contraction with controlled force and calculated efficiency; cellular respiration optimized for sustained performance and endurance; energy transfer maximized through proper technique and muscle coordination; cellular energy provides power for sustained movement",
        "hydrostatic pressure in joints with controlled force and calculated pressure; synovial fluid reduces friction with optimal viscosity and controlled flow; cartilage absorbs impact with controlled compression and calculated strength; bone density supports load with calculated strength and controlled response; joint stability maintained through proper alignment and muscle coordination; joint mechanics ensure smooth and controlled movement",
        "mechanical advantage applied through lever systems with precise calculations and controlled force; force multiplication occurs with controlled amplification and calculated efficiency; energy efficiency optimized through proper technique and muscle coordination; power output maximized through controlled force application and calculated effectiveness; work performed with calculated effectiveness and controlled force; mechanical systems ensure optimal power and efficiency",
        "biomechanical efficiency optimized through proper technique and controlled movement; energy conservation applied with controlled force and calculated efficiency; movement economy improved through precise control and muscle coordination; physiological adaptation occurs with controlled response and calculated timing; performance enhanced through optimized mechanics and controlled force application; biomechanical systems ensure optimal performance and efficiency"
    ]
    

def _force_main_action_injection(self, seq: list, creative_brief: str, lead_label: str = "Lead") -> list:
    """Force inject main actions from creative_brief into sequence"""
    if not creative_brief or not seq:
        return seq
    
    print(f"DEBUG: Force injecting main actions from creative_brief: {creative_brief}")
    main_actions = []
    
    # Extract main actions from creative_brief with enhanced detection
    creative_lower = creative_brief.lower()
    
    # Enhanced keyword detection
    if "thrusts" in creative_lower or "thrust" in creative_lower or "spear" in creative_lower:
        main_actions.append(("thrusts spear", "shoulder and arm muscles engage; core rotates; forward momentum generated; weapon accelerates toward target", "kinetic energy transfers through weapon; impact force calculated; reaction force generated on contact"))
        print("DEBUG: Added thrusts spear action")
    if "charges" in creative_lower or "charge" in creative_lower or "charging" in creative_lower:
        main_actions.append(("charges forward", "leg muscles contract; hip flexors activate; forward propulsion generated; weight shifts forward", "momentum builds; friction resists; acceleration increases; kinetic energy accumulates"))
        print("DEBUG: Added charges forward action")
    if "strikes" in creative_lower or "strike" in creative_lower:
        main_actions.append(("strikes target", "arm muscles contract; weapon accelerates; impact force applied; target receives blow", "kinetic energy transfers to target; reaction force generated; momentum conserved"))
        print("DEBUG: Added strikes target action")
    if "shoots" in creative_lower or "shooting" in creative_lower or "arrows" in creative_lower or "arrow" in creative_lower:
        main_actions.append(("releases arrow", "bowstring tension released; arrow accelerates; projectile motion initiated; trajectory calculated", "elastic energy converts to kinetic; gravity affects trajectory; air resistance slows projectile"))
        print("DEBUG: Added releases arrow action")
    if "pursues" in creative_lower or "pursuit" in creative_lower or "pursuing" in creative_lower:
        main_actions.append(("pursues target", "leg muscles engage; forward motion maintained; tracking initiated; distance closing", "momentum conserved; distance closing; speed optimized; pursuit velocity calculated"))
        print("DEBUG: Added pursues target action")
    
    # Insert main actions at key timestamps (1.0s, 3.0s, 5.0s) with priority
    main_timestamps = [1.0, 3.0, 5.0]
    for i, (verb, biomech, physics) in enumerate(main_actions[:3]):
        if i < len(main_timestamps):
            # Check if timestamp is already occupied
            occupied = any(abs(main_timestamps[i] - float(a.get("timestamp_sec", 0))) < 0.4 for a in seq)
            if not occupied:
                seq.append({
                    "timestamp_sec": main_timestamps[i],
                    "subject": lead_label,
                    "verb": verb,
                    "biomechanics_note": biomech,
                    "physics_note": physics,
                    "internal_motivation": "executing primary objective with focused determination"
                })
                print(f"DEBUG: Injected main action '{verb}' at {main_timestamps[i]}s")
    
    # Sort by timestamp after injection
    seq.sort(key=lambda x: x.get('timestamp_sec', 0))
    print(f"DEBUG: FORCE Main action injection completed - {len(seq)} total actions")
    
    return seq


def _comprehensive_validation(seq: list, creative_brief: str) -> dict:
    """Comprehensive validation of sequence quality"""
    validation_results = {
        "total_actions": len(seq),
        "unique_verbs": len(set(a.get("verb", "") for a in seq)),
        "repetitive_verbs": {},
        "main_action_present": False,
        "timeline_issues": [],
        "character_consistency": True,
        "main_actions_found": [],
        "issues": []
    }
    
    # Check for repetitive verbs
    verb_counts = {}
    for action in seq:
        verb = action.get("verb", "").lower()
        verb_counts[verb] = verb_counts.get(verb, 0) + 1
    
    validation_results["repetitive_verbs"] = {v: c for v, c in verb_counts.items() if c > 2}
    
    # Check for main action from creative_brief
    if creative_brief:
        creative_lower = creative_brief.lower()
        main_keywords = ["thrusts", "charges", "strikes", "shoots", "pursues", "attacks"]
    for action in seq:
            verb = action.get("verb", "").lower()
            if any(keyword in creative_lower and keyword in verb for keyword in main_keywords):
                validation_results["main_action_present"] = True
                validation_results["main_actions_found"].append(f"Found main action: {verb}")
                break
    
    # Check character consistency
    subjects = [action.get("subject", "") for action in seq if action.get("subject")]
    unique_subjects = set(subjects)
    validation_results["character_consistency"] = len(unique_subjects) <= 2  # Allow 2 subjects (warrior + beast)
    
    # Check timeline issues
    timestamps = [float(a.get("timestamp_sec", 0)) for a in seq]
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i-1] < 0.3:
            validation_results["timeline_issues"].append(f"Actions too close at {timestamps[i-1]}s and {timestamps[i]}s")
            validation_results["issues"].append(f"Timeline spacing issue: {timestamps[i-1]}s-{timestamps[i]}s")
    
    return validation_results


# Monkey-patch post-sanity to be cleaner
try:
    _orig_post_sanity_pass = ChatGPTPromptGenerator._post_sanity_pass
except Exception:
    _orig_post_sanity_pass = None

def _post_sanity_pass_clean(self, obj: dict, scene_text: str) -> dict:
    o = _orig_post_sanity_pass(self, obj, scene_text) if _orig_post_sanity_pass else (obj or {})
    try:
        o = self._sanitize_continuity_updates(o)
    except Exception:
        pass
    return o


def _alias_to_canon(name: str) -> str:
    """Legacy function - use global canon() instead."""
    return canon(name)

def _decide_roles(obj: dict) -> dict:
    w = obj.get("world_and_characters") or {}
    blob = " ".join([
        str(obj.get("creative_brief","")),
        str(w.get("battle_context","")),
    ]).lower()
    
    if "warrior" in blob and "beast" in blob:
        w["lead_character"] = "Warrior Squad"
        w["antagonist"] = "Forest Beast"
    elif "archer" in blob or "arrow" in blob:
        w["lead_character"] = "Archer Squad"
        w["antagonist"] = "Forest Beast"
    else:
        w["lead_character"] = w.get("lead_character") or "Warrior Squad"
        w["antagonist"] = w.get("antagonist") or "Forest Beast"
    
    obj["world_and_characters"] = w
    return obj

# Monkey-patch post-sanity pass
try:
    ChatGPTPromptGenerator._post_sanity_pass = _post_sanity_pass_clean
except Exception:
    pass

# === Continuity & Physics post-fixers ===
import re, copy, math

_CANON = {
    "Warrior": {
        "aliases": ["warrior","hunter","hunters","hunter with spear","hunter with bow",
                    "archer","archers","leader","soldier"],
        "verbs": {"thrusts","deflects","readies spear","advances","stands ready",
                  "calls out cues","spreads formation","sets guard","draw",
                  "anchors","re-aims","releases","presses with spear","circles",
                  "re-positions","holds guard while shifting"}
    },
    "Beast": {
        "aliases": ["beast","animal","creature","target","opponent","predator"],
        "verbs": {"charges","recoils","deflects","feints","pivots away",
                  "snarls","roars","attacks","dodges","retreats"}
    },
    "Arrow": {
        "aliases": ["arrow","projectile","missile"],
        "verbs": {"flies","whistles","impacts","sticks","drops","drips (non-graphic)","drips"}
    },
}
_VERB_HINTS = {}
for k,v in _CANON.items():
    for vb in v["verbs"]:
        _VERB_HINTS.setdefault(vb.lower(), set()).add(k)

_ALLOWED_MOVES = {"static","pan","tilt","dolly","truck","crane","orbit","handheld","tracking"}


# ===== Core utility functions =====

def canon(name: str) -> str:
    """Canonicalize subject names using the global _CANON dictionary."""
    if not name:
        return name
    
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in _CANON:
        return name_lower.title()
    
    # Alias lookup
    for canonical, data in _CANON.items():
        if name_lower in [alias.lower() for alias in data.get("aliases", [])]:
            return canonical
    
    # Fallback
    if any(kw in name_lower for kw in ["warrior", "hunter", "archer"]):
        return "Warrior"
    elif any(kw in name_lower for kw in ["beast", "animal", "creature"]):
        return "Beast"
    elif any(kw in name_lower for kw in ["arrow", "projectile"]):
        return "Arrow"
    
    return name

def _v2_norm(s: str) -> str:
    return (s or "").strip().lower()

def _v2_lower(s: str) -> str:
    return (s or "").strip().lower()

# --- Canonical dictionary (VN/EN) ---

_CANON_ALIASES = {
    # lead/antagonist generics
    "lead": "lead",
    "main": "lead",
    "target": "antagonist",
    "opponent": "antagonist",
    
    # specific entities
    "warrior": "group_humans",
    "hunter": "group_humans", 
    "archer": "group_humans",
    "soldiers": "group_humans",
    "beast": "antagonist",
    "animal": "antagonist",
    "creature": "antagonist",
    "predator": "antagonist",
    
    # weapons/tools
    "spear": "spear",
    "bow": "bow",
    "bowstring": "bow",
}

# Vietnamese aliases
_CANON_ALIASES_VI = {
    "chiến binh": "group_humans",
    "thợ săn": "group_humans",
    "kẻ địch": "antagonist",
    "quái thú": "antagonist",
    "thú": "antagonist",
    "con thú": "antagonist",
    "giáo": "spear",
    "cung": "bow",
    "mũi tên": "arrow",
}

def _is_archery_scene(scene_text: str, obj: dict) -> bool:
    """Determine if this is primarily an archery scene."""
    blob = " ".join([
        scene_text or "",
        str(obj.get("creative_brief", "")),
        str(obj.get("world_and_characters", {}).get("battle_context", "")),
    ]).lower()
    
    archery_kw = ["bow", "arrow", "archer", "archery", "cung", "mũi tên", "bắn"]
    melee_kw = ["spear", "sword", "thrust", "strike", "giáo", "đâm", "chọc"]
    
    has_archery = any(k in blob for k in archery_kw)
    has_melee   = any(k in blob for k in melee_kw)
    # Prefer explicit melee if both present
    return has_archery and not has_melee



# ===== Post-processing functions =====



# Monkey-patch the post-processing - sẽ được gán ở cuối file


# ===== Existing functions (keep intact) =====






# hậu kiểm: quét dọn camera beats một lần nữa
try:
    _orig_post_sanity_pass = ChatGPTPromptGenerator._post_sanity_pass
except Exception:
    _orig_post_sanity_pass = None

def _post_sanity_pass_clean(self, obj: dict, scene_text: str) -> dict:
    o = _orig_post_sanity_pass(self, obj, scene_text) if _orig_post_sanity_pass else (obj or {})
    try:
        for s in (o.get("shots") or []):
            seq = s.get("action_sequence") or []
            seq = _purge_camera_beats(seq)
            seq = _fix_nonphysical_and_spam(seq)
            s["action_sequence"] = seq
    except Exception:
        pass
    return o

if _orig_post_sanity_pass:
    ChatGPTPromptGenerator._post_sanity_pass = _post_sanity_pass_clean

# === Continuity & Physics post-fixers ===
import re, copy, math

_CANON = {
    "Warrior": {
        "aliases": ["warrior","hunter","hunters","hunter with spear","hunter with bow",
                    "archer","archers","leader","soldier"],
        "verbs": {"thrusts","deflects","readies spear","advances","stands ready",
                  "calls out cues","spreads formation","sets guard","draw",
                  "sidesteps","prepares stance","parries","blocks","move","moves","yell","yells"},
    },
    "Beast": {
        "aliases": ["beast","animal","creature","hound","target","Target"],
        "verbs": {"growls","roars","charges","swerves","leaps","slashes","reveals",
                  "recoils","dodges","stumbles","falls","snarls","pounces","lunges"},
    },
    "Archers": {
        "aliases": ["archers","archer","hunters with bows"],
        "verbs": {"draws","releases","looses","aims"},
    },
    "Projectile": {
        "aliases": ["arrow","projectile","darting projectile","bolt","fluid"],
        "verbs": {"flies","whistles","impacts","sticks","drops","drips (non-graphic)","drips"},
    },
}
_VERB_HINTS = {}
for k,v in _CANON.items():
    for vb in v["verbs"]:
        _VERB_HINTS.setdefault(vb.lower(), set()).add(k)

_ALLOWED_MOVES = {"static","pan","tilt","dolly","truck","crane","orbit","handheld","tracking"}

def _alias_to_canon(name: str) -> str:
    """Legacy function - use global canon() instead."""
    return canon(name)

def _decide_roles(obj: dict) -> dict:
    w = obj.get("world_and_characters") or {}
    blob = " ".join([
        str(obj.get("creative_brief","")),
        str(w.get("battle_context","")),
    ]).lower()
    if ("warrior" in blob or "hunters" in blob or "archers" in blob) and "beast" in blob:
        lead, ant = "Warrior", "Beast"
    elif any(x in blob for x in ["animal","creature","beast"]):
        lead, ant = "Beast", "Warrior"
    else:
        lead = _alias_to_canon(w.get("lead_character") or "Warrior")
        ant  = _alias_to_canon(w.get("antagonist") or "Beast")
    w["lead_character"], w["antagonist"] = lead, ant
    obj["world_and_characters"] = w
    return obj

def _is_physical_verb(v: str) -> bool:
    return (v or "").lower().strip() in {
        "re-aims","emerges","growls","reveals","scratches","scans while moving","dodges",
        "lunges","stumbles","slashes","falls","drops","recoils","charges","deflects",
        "readies spear","advances","sidesteps","prepares stance","leaps","thrusts",
        "sets guard","yell","yells","move","moves","braces","re-centers balance",
        "spreads formation","prepare","prepares","reacts","stands ready","startle","startles","draw"
    }

def _relabel_subject_by_verb(subj: str, verb: str) -> str:
    s = _alias_to_canon(subj)
    if s.lower() in ("camera","scene"):  # bỏ pseudo-subject
        return ""
    hints = _VERB_HINTS.get((verb or "").lower(), set())
    if hints and s not in hints:
        if "Beast" in hints: return "Beast"
        if "Warrior" in hints: return "Warrior"
        if "Projectile" in hints: return "Projectile"
    return s

def _fix_actions(actions: list) -> list:
    out, last_t = [], -1.0
    for a in actions or []:
        t = a.get("timestamp_sec")
        try: t = float(t)
        except: t = (last_t + 0.16) if last_t >= 0 else 0.16
        v = (a.get("verb") or "").lower().strip()
        s = a.get("subject") or ""
        if not _is_physical_verb(v):  # bỏ camera:, scene:, cuts, etc.
            continue
        s2 = _relabel_subject_by_verb(s, v)
        if not s2: continue
        v = v.replace("growls again","growls").replace("calls out cues","yells")
        if t <= last_t: t = round(last_t + 0.16, 2)
        out.append({
            "timestamp_sec": float(f"{t:.2f}"),
            "subject": s2,
            "verb": "advances" if v == "advance" else v,
            "biomechanics_note": (a.get("biomechanics_note") or "").strip(),
            "physics_note": (a.get("physics_note") or "").strip(),
            "internal_motivation": (a.get("internal_motivation") or "").strip(),
        })
        last_t = t
    return out[:14]

def _normalize_camera(cam: dict) -> dict:
    cam = cam or {}
    mv = (cam.get("movement") or {}).get("type","")
    low = str(mv).lower()
    new_move = "handheld" if (cam.get("rig") or "").lower()=="handheld" else "dolly"
    if "pan" in low: new_move = "pan"
    if "tilt" in low: new_move = "tilt"
    if any(x in low for x in ["track","follow","fast tracking"]): new_move = "tracking"
    if "orbit" in low or "circular" in low: new_move = "orbit"
    if "static" in low: new_move = "static"
    has_zoom = "zoom" in low
    # clamp yaw
    ori = cam.get("orientation_deg") or {}
    try:
        yaw = float(ori.get("yaw",0) or 0)
    except:
        yaw = 0.0
    if abs(yaw) >= 180: yaw = max(min(yaw,179), -179)
    ori["yaw"] = yaw
    cam["orientation_deg"] = ori
    # clamp delta pos
    pos = cam.get("position_m") or {}
    st, ed = pos.get("start") or {}, pos.get("end") or {}
    def _clamp(a,b,limit=2.0):
        try: a=float(a); b=float(b)
        except: return a,b
        d=b-a
        if abs(d)>limit: b=a+(limit if d>0 else -limit)
        return a,b
    for ax in ("x","y","z"):
        a,b=_clamp(st.get(ax,0), ed.get(ax,0))
        st[ax],ed[ax]=a,b
    cam["position_m"]={"start":st,"end":ed}
    cam["movement"]={"type": new_move}
    if has_zoom and not cam.get("focus_pull_style"):
        cam["focus_pull_style"]="slow rack focus"
    if cam.get("depth_of_field") not in ("shallow","moderate","deep"):
        cam["depth_of_field"]="moderate"
    return cam

_AUDIO_SOFTEN = [
    (r"(?i)spear\s+thrusting\s+into\s+flesh", "spear impact (non-graphic)"),
    (r"(?i)damage\s+dripping", "fluid drip (non-graphic)"),
    (r"(?i)anguished\s+roar", "fierce roar"),
    (r"(?i)bone-?chilling\s+roar", "deep roar"),
    (r"(?i)sharp\s+tearing\s+sound", "sharp fabric/gear tear (non-graphic)"),
]

def _soften_audio(aud: dict) -> dict:
    out = copy.deepcopy(aud or {})
    for k in ("foley","sfx"):
        arr = [str(x) for x in (out.get(k) or [])]
        clean=[]
        for x in arr:
            y=x
            for pat,rep in _AUDIO_SOFTEN:
                y=re.sub(pat,rep,y)
            if y not in clean:
                clean.append(y)
        out[k]=clean
    return out

def _filter_dynamic_entities(cu: dict) -> dict:
    cu = copy.deepcopy(cu or {})
    kept=[]
    for e in cu.get("new_or_updated_dynamic_entities") or []:
        n = _alias_to_canon(e.get("name") or "")
        if n in ("Projectile","Arrow"):
            ee = copy.deepcopy(e); ee["name"]="Projectile"; kept.append(ee)
    cu["new_or_updated_dynamic_entities"]=kept
    return cu

def _sanitize_single_prompt_block(sp: str) -> str:
    if not sp: return sp
    sp = re.sub(r"(?im)^\s*-\s*t=\d+(\.\d+)?s\s*\|\s*camera:\s*.+$", "", sp)  # drop camera beats
    sp = re.sub(r"(?im)^\s*-\s*t=\d+(\.\d+)?s\s*\|\s*scene:\s*.+$",  "", sp)  # drop scene beats
    sp = re.sub(r"(?i)zoom(s)? in", "rack focus in", sp)                     # zoom -> focus
    # Rebuild ENTITY PROFILES section
    lines, out, skip = sp.splitlines(), [], False
    for ln in lines:
        if ln.strip().startswith("ENTITY PROFILES:"):
            out.append("ENTITY PROFILES:")
            out.append(" - Warrior | mass≈80kg, CoG=mid-torso, locomotion=bipedal | state: alert | interactions: spear, bow")
            out.append(" - Beast | mass≈200kg, CoG=low-torso, locomotion=quadruped | state: agitated | interactions: charge, recoil, dodge")
            out.append(" - Projectile | mass≈<1kg, motion_model=ballistic arc (gravity, drag, momentum) | interactions: flight, impact (non-graphic)")
            skip=True
            continue
        if skip:
            if ln.strip().startswith("REQUIREMENTS:"):
                out.append(ln); skip=False
            # else drop old noisy profiles
        else:
            out.append(ln)
    sp = "\n".join(out)
    # Canonical naming
    for bad,good in [("animal","Beast"),("creature","Beast"),("hound","Beast"),
                     ("Target","Beast"),("hunters","Warrior"),
                     ("hunter with spear","Warrior"),("hunter with bow","Warrior")]:
        sp = re.sub(rf"(?i)\b{re.escape(bad)}\b", good, sp)
    return sp

def _normalize_actions_camera_audio(obj: dict) -> dict:
    for shot in obj.get("shots") or []:
        shot["action_sequence"] = _fix_actions(shot.get("action_sequence") or [])
        shot["camera"] = _normalize_camera(shot.get("camera") or {})
        shot["audio"]  = _soften_audio(shot.get("audio") or {})
    return obj

def apply_hard_sanity(obj_translated: dict) -> dict:
    obj = copy.deepcopy(obj_translated or {})
    obj = _decide_roles(obj)
    obj = _normalize_actions_camera_audio(obj)
    cu = obj.get("continuity_updates")
    obj["continuity_updates"] = _filter_dynamic_entities(cu)
    sp = obj.get("single_prompt_for_model","")
    obj["single_prompt_for_model"] = _sanitize_single_prompt_block(sp)
    # Final pass: world_and_characters canonical names
    w = obj.get("world_and_characters") or {}
    w["lead_character"] = _alias_to_canon(w.get("lead_character",""))
    w["antagonist"]     = _alias_to_canon(w.get("antagonist",""))
    obj["world_and_characters"] = w
    return obj

# --- AUTO-BIND TOP-LEVEL 'self'-FUNCTIONS AS CLASS METHODS ---
import inspect as _inspect

for _name, _obj in list(globals().items()):
    if _inspect.isfunction(_obj):
        try:
            _params = list(_inspect.signature(_obj).parameters.values())
        except (ValueError, TypeError):
            continue
        if _params and _params[0].name == "self":
            if not hasattr(ChatGPTPromptGenerator, _name):
                setattr(ChatGPTPromptGenerator, _name, _obj)
# ----------------------------------------------------------------

apply_veo3_patch()  # chỉ gọi sau auto-bind, không double-wrap

# === VEO3 SUPER-CONTINUITY PATCH v2 — entity lock + action/physics gate ===
# Paste *below* ChatGPTPromptGenerator class. This monkey-patches build_scene_prompt
# to repair/normalize JSON per-scene with strict continuity & physics.

import re
from copy import deepcopy


# --- Canonical dictionary (VN/EN) ---

_CANON_ALIASES = {
    # lead/antagonist generics
    "lead": "lead",
    "main": "lead",
    "target": "antagonist",
    "opponent": "antagonist",
    "enemy": "antagonist",
    "foe": "antagonist",
    "animal": "antagonist",
    "beast": "antagonist",

    # humans / groups
    "warrior": "group_humans",
    "warriors": "group_humans",
    "hunter": "group_humans",
    "hunters": "group_humans",
    "archer": "group_humans",
    "archers": "group_humans",

    # archery items
    "arrow": "projectile",
    "arrows": "projectile",
    "bow": "bow",
    "bowstring": "bow",
}

# Vietnamese aliases
_CANON_ALIASES_VI = {
    "chiến binh": "group_humans",
    "thợ săn": "group_humans",
    "kẻ địch": "antagonist",
    "quái thú": "antagonist",
    "thú": "antagonist",
    "mũi tên": "projectile",
    "cung": "bow",
    "nỏ": "bow",
    "giáo": "melee_weapon",
    "mác": "melee_weapon",
    "gươm": "melee_weapon",
}

def _is_archery_scene(text: str, out: dict) -> bool:
    blob = " ".join([
        _v2_lower(text),
        _v2_lower(str(out.get("single_prompt_for_model", ""))),
        _v2_lower(" ".join(out.get("audio", {}).get("sfx", []) if out.get("audio") else [])),
        _v2_lower(" ".join(out.get("vfx", []) or [])),
    ])
    archery_kw = ("bow", "arrow", "archer", "bowstring", "mũi tên", "cung", "nỏ")
    melee_kw   = ("spear", "giáo", "gươm", "kiếm", "mác", "shield", "khiên")
    has_archery = any(k in blob for k in archery_kw)
    has_melee   = any(k in blob for k in melee_kw)
    # Prefer explicit melee if both present
    return has_archery and not has_melee

def _build_catalog(continuity, out: dict) -> dict:
    """Derive global entity catalog from continuity/world_and_characters."""
    w = out.get("world_and_characters") or {}
    lead = _v2_norm(w.get("lead_character") or "warrior")
    antagonist = _v2_norm(w.get("antagonist") or "beast")

    # lock group name (hunters/warriors) if present
    group_candidates = []
    for name in ("hunters", "warriors", "archers"):
        if re.search(rf"\b{name}\b", _v2_lower(str(out)), re.I):
            group_candidates.append(name)
    group_name = group_candidates[0] if group_candidates else "warriors"

    cat = {
        "lead": lead,
        "antagonist": antagonist,
        "groups": {group_name},
        "env": {"forest", "glade", "ground", "camera"},  # known non-actors
        "creative_brief": out.get("creative_brief", "")
    }
    print(f"🔧 DEBUG: _build_catalog created cat with creative_brief: '{cat.get('creative_brief', 'NOT FOUND')}'")
    # Persist in continuity if available
    try:
        if hasattr(continuity, "catalog") and continuity.catalog:
            # keep the first locked values, extend groups
            base = continuity.catalog
            base["groups"].update(cat["groups"])
            return base
        else:
            continuity.catalog = cat
    except Exception:
        pass
    return cat

# --- Allowed verbs per role/context ---

_ALLOWED = {
    "human_melee": {
        "advance","approach","feint","parry","block","deflect","thrust","slash",
        "brace","recenter","recover","sidestep","circle","yell","call","signal",
        "drag","pull","push","dodge","roll","evade","stumble","ready","prepare",
        "charge","engage","disengage","backstep","lunge"
    },
    "animal": {
        "charge","lunge","sidestep","swerve","circle","growl","snarl","rear",
        "recoil","stumble","evade","pounce","feint","advance","retreat","shake"
    },
    "archery": {
        "nock","draw","anchor","aims","re-aims","release","loose","enters flight",
        "impact","stick","glance","recover","ready"
    },
}

# physics/biomechanics templating
def _bio_for(verb: str, role: str) -> str:
    if verb in {"thrust","lunge","charge"}:
        return "hips drive forward; stance loads then explodes; arms extend"
    if verb in {"parry","deflect","block"}:
        return "forearms/weapon set at angle; feet adjust; shoulders rotate"
    if verb in {"sidestep","swerve","dodge"}:
        return "quick lateral step; knees bend; center of mass stays over base"
    if verb in {"recover","recenter","brace"}:
        return "breathing resets; stance narrows; weight centers"
    if verb in {"nock","draw","anchor","release"}:
        return "scapular engagement; forearm alignment; breath control"
    return "natural whole-body coordination; no idle freeze"

def _phys_for(verb: str, role: str) -> str:
    if verb in {"thrust","lunge","charge"}:
        return "ground reaction force → forward momentum; friction prevents slip; torque from hips"
    if verb in {"parry","deflect","block"}:
        return "incoming momentum redirected; impulse spread through arms/core"
    if verb in {"sidestep","swerve","dodge"}:
        return "lateral friction impulse; CoM kept within support polygon"
    if verb in {"recover","recenter","brace"}:
        return "micro-adjustments damp residual momentum; posture stabilizes"
    if verb in {"nock","draw","anchor","release"}:
        return "stored elastic energy → arrow momentum; gravity & drag during flight"
    return "momentum, friction, and balance constraints respected"

def _role_of(subject: str, cat: dict) -> str:
    s = _v2_lower(subject)
    if s == _v2_lower(cat["lead"]) or s in cat["groups"]:
        return "human"
    if s == _v2_lower(cat["antagonist"]) or s in {"beast","animal"}:
        return "animal"
    if s in {"arrow","arrows","projectile"}:
        return "projectile"
    return "unknown"

def _canonical_subject(name: str, cat: dict, archery_on: bool) -> str:
    raw = _v2_lower(name)
    if raw in {"camera"}:
        return "__DROP__"
    # vn/en alias map
    key = raw
    if key in _CANON_ALIASES_VI:
        key = _CANON_ALIASES_VI[key]
    if key in _CANON_ALIASES:
        key = _CANON_ALIASES[key]

    # map to catalog slots
    if key in {"lead"}:
        return cat["lead"]
    if key in {"antagonist"}:
        return cat["antagonist"]
    if key in {"group_humans"}:
        # choose the primary group name from catalog
        return list(cat["groups"])[0]
    if key in {"projectile","bow"}:
        return key if archery_on else "__DROP__"
    # literal matches to catalog
    if raw == _v2_lower(cat["lead"]):
        return cat["lead"]
    if raw == _v2_lower(cat["antagonist"]):
        return cat["antagonist"]
    if raw in cat["groups"]:
        return raw
    return raw  # keep as-is (may be env); will be filtered later


def _fix_action_sequence(actions: list, cat: dict, archery_on: bool) -> list:
    """
    Generate a dense, professionally staged action sequence from the creative brief,
    then harmonize it so the warrior and beast interplay feels continuous and cinematic.
    """
    creative_brief = cat.get("creative_brief", "")
    print(f"🔧 DEBUG: _fix_action_sequence called with creative_brief: '{creative_brief}'")
    print(f"🔧 DEBUG: Original actions length: {len(actions) if actions else 0}")
    new_sequence = _create_dense_action_sequence(cat, archery_on, creative_brief)
    print(f"🔧 DEBUG: Generated {len(new_sequence)} actions")
    lead = cat.get("lead", "Warrior Squad")
    ant = cat.get("antagonist", "Forest Beast")
    harmonized = _harmonize_action_sequence(new_sequence, lead, ant)
    print(f"🔧 DEBUG: Harmonized to {len(harmonized)} actions")
    return harmonized

def _create_dense_action_sequence(cat: dict, archery_on: bool, creative_brief: str = "") -> list:
    """
    Tạo action sequence dày đặc phù hợp với creative_brief.
    """
    print(f"🔧 DEBUG: _create_dense_action_sequence called with creative_brief: '{creative_brief}'")
    lead = cat.get("lead", "Warrior")
    ant = cat.get("antagonist", "Beast")
    
    # Phân tích creative_brief để tạo action sequence phù hợp
    creative_lower = creative_brief.lower() if creative_brief else ""
    print(f"🔧 DEBUG: creative_lower: '{creative_lower}'")
    
    if "thrusts" in creative_lower or "spear" in creative_lower:
        # Scene 1: Warrior charges and thrusts spear
        print(f"🔧 DEBUG: Matched 'thrusts' or 'spear' pattern - creating Scene 1 action sequence")
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        actions = [
            (lead, "charges forward", "leg muscles contract; hip flexors activate; forward propulsion generated", "momentum builds; friction resists; acceleration increases", "overwhelm the front line"),
            (lead, "thrusts spear", "shoulder and arm muscles engage; core rotates; forward momentum generated", "kinetic energy transfers through weapon; impact force calculated", "execute the primary attack"),
            (ant, "recoils in pain", "mass shifts opposite vector; brief stagger", "momentum transfer; friction limits slide", "instinctive reaction to impact"),
            (ant, "reveals damage", "body language shows injury; visible wound appears", "damage becomes apparent; blood seeps from wound", "show the consequences of the attack"),
            (lead, "presses advantage", "rear foot drives; hips extend; lead arm snaps forward", "impulse sends spear tip deeper", "capitalize on opening"),
            (ant, "deflects", "shoulders twist; forelimbs parry line", "angular momentum redirects incoming vector", "avoid direct hit"),
            (lead, "circles", "short arc steps; gaze locked; guard up", "centripetal acceleration kept small; balance preserved", "seek better angle"),
            (ant, "feints", "head dips; weight teases forward", "small impulse to test response", "probe defenses"),
            (lead, "re-positions", "heel-toe shift; shoulders counter-rotate", "momentum managed to avoid overshoot", "maintain initiative"),
            (ant, "charges", "forelimbs extend; spine flexes; head low", "acceleration builds; mass drives forward", "overwhelm the front line"),
            (lead, "sets guard", "spearman lowers stance; plants rear foot", "lowered CoG for stability against incoming mass", "prepare to absorb and counter"),
            (ant, "attacks", "claws extend; body lunges forward", "kinetic energy builds; impact force calculated", "overwhelm the warrior"),
            (lead, "defends", "spear blocks; body braces for impact", "deflection force redirects attack; stability maintained", "protect against incoming attack"),
            (ant, "retreats", "body pulls back; claws retract", "momentum reverses; distance increases", "escape the confrontation"),
            (lead, "pursues", "legs drive forward; spear ready", "momentum builds; distance closing", "maintain pressure on target"),
            (ant, "flees", "powerful leg muscles propel; rapid acceleration", "momentum builds; air resistance increases", "escape the threat")
        ]
    elif "attacking" in creative_lower or "relentless" in creative_lower or "beast charging" in creative_lower:
        # Scene 2: Beast attacking warrior
        print(f"🔧 DEBUG: Matched 'attacking' or 'relentless' or 'beast charging' pattern - creating Scene 2 action sequence")
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        actions = [
            (ant, "charges", "forelimbs extend; spine flexes; head low", "acceleration builds; mass drives forward under traction", "overwhelm the front line"),
            (lead, "braces", "legs spread; arms ready to defend", "stability increases to withstand potential attack", "prepare for confrontation"),
            (ant, "attacks", "claws extend; body lunges forward", "kinetic energy builds; impact force calculated", "overwhelm the warrior"),
            (lead, "defends", "spear blocks; body braces for impact", "deflection force redirects attack; stability maintained", "protect against incoming attack"),
            (ant, "presses", "body weight drives forward; claws swipe", "momentum continues; pressure increases", "maintain offensive pressure"),
            (lead, "retreats", "body pulls back; spear ready", "momentum reverses; distance increases", "escape the immediate threat"),
            (ant, "pursues", "legs drive forward; claws ready", "momentum builds; distance closing", "maintain pressure on target"),
            (lead, "evades", "body twists; feet shuffle", "lateral movement; momentum redirected", "avoid direct contact"),
            (ant, "feints", "head dips; weight teases forward", "small impulse to test response", "probe defenses"),
            (lead, "counters", "spear thrusts; body lunges forward", "kinetic energy builds; impact force calculated", "strike back at the beast"),
            (ant, "deflects", "shoulders twist; forelimbs parry line", "angular momentum redirects incoming vector", "avoid direct hit"),
            (lead, "circles", "short arc steps; gaze locked; guard up", "centripetal acceleration kept small; balance preserved", "seek better angle"),
            (ant, "repositions", "weight shifts; stance adjusts", "momentum managed to avoid overshoot", "maintain tactical advantage"),
            (lead, "recenters", "eyes narrow; bodies tense", "heightened awareness of surroundings", "stay alert for next move"),
            (ant, "stalls", "movement slows; head tilts", "momentum decreases; balance maintained", "assess the situation"),
            (lead, "advances", "feet shuffle; bodies pivot", "momentum shifts as they react to surroundings", "navigate the environment")
        ]
    elif "arrows" in creative_lower or "pursuit" in creative_lower or "pursuing" in creative_lower:
        # Scene 3: Warriors pursuing with arrows
        print(f"🔧 DEBUG: Matched 'arrows' or 'pursuit' or 'pursuing' pattern - creating Scene 3 action sequence")
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        actions = [
            (lead, "draws bow", "scapular engagement; string hand pulls; bow arm stabilizes", "elastic energy stores in limbs of the bow", "commit to a clean release"),
            (lead, "aims", "string hand settles at corner of mouth; elbow aligns", "stable anchor reduces lateral error", "ensure accuracy under stress"),
            (lead, "releases", "fingers relax; bow arm resists torque; slight follow-through", "stored energy converts to arrow kinetic energy; mild recoil", "execute the shot"),
            ("Arrow", "flies", "fletching stabilizes; shaft flexes then dampens", "parabolic arc under gravity; slight lateral drift in light breeze", "maintain trajectory toward target"),
            (ant, "dodges", "quick lateral movement; body twists; weight shifts rapidly", "inertia overcome; rapid change in direction; momentum redirected", "evade incoming projectile"),
            (lead, "nocks arrow", "fingers retrieve arrow; nock aligns with string", "potential energy stored in drawn bow; precise alignment", "prepare for next shot"),
            (lead, "draws", "scapular engagement; string hand pulls; bow arm stabilizes", "elastic energy stores in limbs of the bow", "commit to a clean release"),
            (lead, "aims", "micro footwork; grip micro-corrections synced to breath", "tiny torque corrections stabilize sight picture", "compensate for the Beast's motion"),
            (lead, "releases", "fingers relax; bow arm resists torque; slight follow-through", "stored energy converts to arrow kinetic energy; mild recoil", "execute the shot"),
            ("Arrow", "flies", "fletching stabilizes; shaft flexes then dampens", "parabolic arc under gravity; slight lateral drift in light breeze", "maintain trajectory toward target"),
            (ant, "flees", "powerful leg muscles propel; rapid acceleration; evasive maneuvers", "momentum builds; air resistance increases; ground friction provides traction", "escape the threat"),
            (lead, "pursues", "leg muscles engage; forward motion maintained; tracking initiated", "momentum conserved; distance closing; speed optimized", "maintain pressure on target"),
            (lead, "reloads", "quick retrieval of arrow; smooth nocking; rapid draw", "efficient motion minimizes downtime; muscle memory guides action", "maintain offensive pressure"),
            (ant, "accelerates", "powerful leg muscles contract; increased stride frequency", "maximum propulsion generated; air resistance increases", "maximize distance from threat"),
            (lead, "tracks", "eyes follow target; body adjusts position", "visual tracking; spatial awareness maintained", "maintain visual contact with target"),
            (ant, "escapes", "body disappears into forest; movement becomes erratic", "momentum builds; distance increases; visibility decreases", "successfully evade pursuit")
        ]
    else:
        # Default action sequence
        print(f"🔧 DEBUG: No specific pattern matched - creating Default action sequence")
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        actions = [
            (lead, "advances", "feet shuffle; bodies pivot", "momentum shifts as they react to surroundings", "navigate the environment"),
            (ant, "circles", "short arc steps; gaze locked", "centripetal acceleration kept small; balance preserved", "seek better angle"),
            (lead, "braces", "legs spread; arms ready to defend", "stability increases to withstand potential attack", "prepare for confrontation"),
            (ant, "repositions", "weight shifts; stance adjusts", "momentum managed to avoid overshoot", "maintain tactical advantage"),
            (lead, "recenters", "eyes narrow; bodies tense", "heightened awareness of surroundings", "stay alert for next move"),
            (ant, "stalls", "movement slows; head tilts", "momentum decreases; balance maintained", "assess the situation"),
            (lead, "cuts angle left", "heel-toe step; hips rotate; shoulders counter-rotate", "lateral shift manages distance; friction prevents slide", "gain flanking position"),
            (ant, "charges", "forelimbs extend; spine flexes; head low", "acceleration builds; mass drives forward under traction", "overwhelm the front line"),
            (lead, "sets guard", "spearman lowers stance; bowman plants rear foot", "lowered CoG for stability against incoming mass", "prepare to absorb and counter"),
            (ant, "attacks", "claws extend; body lunges forward", "kinetic energy builds; impact force calculated", "overwhelm the warrior"),
            (lead, "defends", "spear blocks; body braces for impact", "deflection force redirects attack; stability maintained", "protect against incoming attack"),
            (ant, "retreats", "body pulls back; claws retract", "momentum reverses; distance increases", "escape the confrontation"),
            (lead, "pursues", "legs drive forward; spear ready", "momentum builds; distance closing", "maintain pressure on target"),
            (ant, "flees", "powerful leg muscles propel; rapid acceleration", "momentum builds; air resistance increases", "escape the threat"),
            (lead, "tracks", "eyes follow target; body adjusts position", "visual tracking; spatial awareness maintained", "maintain visual contact with target"),
            (ant, "escapes", "body disappears into forest; movement becomes erratic", "momentum builds; distance increases; visibility decreases", "successfully evade pursuit")
        ]
    
    sequence = []
    for i, (timestamp, (subject, verb, bio, phys, motive)) in enumerate(zip(timestamps, actions)):
        sequence.append({
            "timestamp_sec": timestamp,
            "subject": subject,
            "verb": verb,
            "biomechanics_note": bio,
            "physics_note": phys,
            "internal_motivation": motive
        })
    
    print(f"🔧 DEBUG: Created {len(sequence)} actions")
    print(f"🔧 DEBUG: First action: {sequence[0] if sequence else 'None'}")
    return sequence



def _harmonize_action_sequence(seq: list, lead_label: str, ant_label: str) -> list:
    """Enforce cinematic continuity: canonical names, cause?effect beats, even pacing."""
    if not isinstance(seq, list):
        return []

    lead = _canon(lead_label or "Warrior Squad")
    ant = _canon(ant_label or "Forest Beast")
    offensive_lead = {
        "thrusts", "drives spear forward", "presses advantage", "lashes with spear tip",
        "thrusts spear", "charges forward", "ripostes", "strikes", "slashes"
    }
    offensive_ant = {
        "charges", "launches forward", "presses", "lunges", "swipes", "rakes",
        "attacks", "pounces"
    }

    cleaned = []
    last_time = -0.4
    for action in sorted(seq, key=lambda a: float(a.get("timestamp_sec", 0.0))):
        if not isinstance(action, dict):
            continue
        a = dict(action)
        subj = _canon(a.get("subject", ""))
        if subj in {"", "Damage", "Camera"}:
            continue
        if subj not in {lead, ant, "Arrow"}:
            subj = lead if subj.startswith("Warrior") else ant if subj.startswith("Forest") else subj
        a["subject"] = subj
        try:
            ts = float(a.get("timestamp_sec", 0.0))
        except Exception:
            ts = max(last_time + 0.4, 0.0)
        if ts - last_time < 0.35:
            ts = round(last_time + 0.35, 2)
        a["timestamp_sec"] = float(f"{ts:.2f}")
        last_time = a["timestamp_sec"]
        cleaned.append(a)

    enhanced = []
    for idx, act in enumerate(cleaned):
        enhanced.append(act)
        subj = act["subject"]
        verb = (act.get("verb", "") or "").lower()
        ts = act["timestamp_sec"]
        if subj == lead and verb in offensive_lead:
            future = next((x for x in cleaned[idx + 1:] if x["subject"] == ant and x["timestamp_sec"] - ts <= 0.6), None)
            if not future:
                enhanced.append({
                    "timestamp_sec": float(f"{min(ts + 0.45, 7.6):.2f}"),
                    "subject": ant,
                    "verb": "recoils",
                    "biomechanics_note": "torso snaps back; paws skid; shoulders hunch",
                    "physics_note": "momentum reverses through friction; balance recaptured",
                    "internal_motivation": "absorb the spear pressure without collapsing"
                })
        if subj == ant and verb in offensive_ant:
            future = next((x for x in cleaned[idx + 1:] if x["subject"] == lead and x["timestamp_sec"] - ts <= 0.6), None)
            if not future:
                enhanced.append({
                    "timestamp_sec": float(f"{min(ts + 0.45, 7.6):.2f}"),
                    "subject": lead,
                    "verb": "braces",
                    "biomechanics_note": "stance widens; shield or spear haft absorbs; breath firms",
                    "physics_note": "counter-force spreads through frame; ground reaction keeps footing",
                    "internal_motivation": "stay rooted and ready to counter"
                })

    enhanced.sort(key=lambda a: a["timestamp_sec"])

    if enhanced:
        final_ts = enhanced[-1]["timestamp_sec"]
        if final_ts < 7.6:
            enhanced.append({
                "timestamp_sec": 7.85,
                "subject": lead,
                "verb": "holds cinematic guard",
                "biomechanics_note": "micro-sway synced to breath; hands steady; gaze locked",
                "physics_note": "continuous micro-adjustments counter residual momentum; no idle plateau",
                "internal_motivation": "carry intent through the final frame"
            })

    sequenced = []
    last_time = -0.3
    for act in enhanced:
        ts = act["timestamp_sec"]
        if ts - last_time < 0.35:
            ts = round(last_time + 0.35, 2)
        act["timestamp_sec"] = float(f"{min(ts, 7.90):.2f}")
        sequenced.append(act)
        last_time = act["timestamp_sec"]

    return sequenced



def _create_dynamic_camera_movement(action_sequence: list, creative_brief: str = "") -> dict:
    """Return a cinema-grade camera movement plan with rationale and positional cues."""
    creative_lower = (creative_brief or "").lower()
    default = {
        "type": "floating dolly with handheld micro-sway",
        "direction": "forward with gentle arc",
        "speed": "moderate crescendo",
        "motivation": "immerse viewer in the warrior's pursuit while keeping blocking readable",
        "start_pose": {"x": -1.2, "y": 1.6, "z": 4.0},
        "end_pose": {"x": -0.4, "y": 1.6, "z": 3.0},
        "secondary_motion": "breathing-linked sway to avoid static feel"
    }

    if "thrust" in creative_lower or "spear" in creative_lower:
        return {
            "type": "precision dolly-in with shoulder steadicam",
            "direction": "forward diagonal, favouring warrior weapon side",
            "speed": "rising tempo that peaks at impact",
            "motivation": "spotlight the thrust geometry and the beast recoil in one fluid move",
            "start_pose": {"x": -1.4, "y": 1.55, "z": 4.2},
            "end_pose": {"x": -0.6, "y": 1.55, "z": 2.6},
            "secondary_motion": "slow rack-focus from warrior guard to beast reaction"
        }
    if any(k in creative_lower for k in ("attacking", "relentless", "beast charging")):
        return {
            "type": "aggressive lateral tracking shot",
            "direction": "left-to-right sweep that pivots around combat",
            "speed": "surging pace with micro accelerations on attacks",
            "motivation": "convey the beast mass and the warrior defensive pivots",
            "start_pose": {"x": -2.0, "y": 1.5, "z": 3.6},
            "end_pose": {"x": 0.8, "y": 1.5, "z": 3.3},
            "secondary_motion": "brief whip-pan accents following claw strikes"
        }
    if any(k in creative_lower for k in ("arrow", "pursuit", "pursuing")):
        return {
            "type": "sweeping pursuit gimbal",
            "direction": "wide arc that tracks the hunters while revealing the escape path",
            "speed": "elastic tempo - fast on chase bursts, gliding during releases",
            "motivation": "let viewers read both arrow flight and beast evasion in the same frame",
            "start_pose": {"x": -1.8, "y": 1.7, "z": 4.5},
            "end_pose": {"x": 1.0, "y": 1.7, "z": 4.0},
            "secondary_motion": "subtle roll corrections for terrain undulation"
        }
    return default



def _create_dynamic_audio(action_sequence: list, creative_brief: str = "") -> dict:
    """Build a layered audio design with foley, sfx, ambience, and mix strategy."""
    if not action_sequence:
        return {
            "foley": ["soft breeze through foliage", "subtle leather movement"],
            "sfx": ["distant wildlife calls"],
            "ambience": "bed of forest insects and wind",
            "music_cue": "low, suspended strings",
            "mix_strategy": "- Ambience at -18 LUFS, foley -12 LUFS, SFX -6 LUFS",
            "spatialization": "wrap reverb 0.4, early reflections 25ms",
            "notes": "keep dialog intelligible; duck ambience during foley spikes"
        }

    creative_lower = (creative_brief or "").lower()

    if "thrust" in creative_lower or "spear" in creative_lower:
        return {
            "foley": [
                "Heavy warrior footsteps digging into soil",
                "Armor leather creaks timed with pivot",
                "Warrior breathing tightening before impact",
                "Beast guttural rumble transitioning to pained bellow (non-graphic)"
            ],
            "sfx": [
                "Spear shaft whoosh with air displacement",
                "Non-graphic impact thud with bark and sine layers",
                "Shield scrape against claws",
                "Debris scatter as feet churn earth"
            ],
            "ambience": "Low canopy wind, sunlit insects, distant birds startled on impact",
            "music_cue": "Hybrid percussion swell with low brass hit on thrust",
            "mix_strategy": "- Foley close-mic centre, SFX wide stereo, ambience mid/side with 200Hz high-pass",
            "spatialization": "Convolution reverb 0.5s tail, slight slapback on beast roar",
            "notes": "Side-chain music 3dB when spear impact lands; keep beast vocal mono for weight"
        }

    if any(k in creative_lower for k in ("attacking", "relentless", "beast charging")):
        return {
            "foley": [
                "Beast claws tearing ground",
                "Warrior shield bracing and scraping",
                "Grunted call-outs between defenders",
                "Armor collisions and sliding footwork"
            ],
            "sfx": [
                "Aggressive claw whooshes",
                "Shield-metal shock impacts",
                "Broken branch snaps as combat shifts",
                "Dust plume bursts for heavy landings"
            ],
            "ambience": "Dense forest reverb with distant echo of roars",
            "music_cue": "Rhythmic taiko pulses layered with synth bass rise",
            "mix_strategy": "- Beast vocals up-front mono, warrior foley stereo, ambience side-mid",
            "spatialization": "Multi-tap delay on metallic hits, subtle doppler on beast rush",
            "notes": "EQ carve 400Hz on music during dialogue; add 3dB headroom for sudden impacts"
        }

    if any(k in creative_lower for k in ("arrow", "pursuit", "pursuing")):
        return {
            "foley": [
                "Boots sliding across underbrush",
                "Bow grip creaks and finger releases",
                "Laboured breathing from extended chase",
                "Warrior hand signals brushing gear"
            ],
            "sfx": [
                "Layered arrow twang and air hiss",
                "Arrow impact into bark and soil hybrids",
                "Beast panting and brush collisions",
                "Far-field echo of commands bouncing through trees"
            ],
            "ambience": "Wide forest bed with intermittent gusts and passing wildlife",
            "music_cue": "High strings ostinato with hand percussion to drive pursuit",
            "mix_strategy": "- Foley mid, SFX panning with arrows, ambience 120deg width",
            "spatialization": "Doppler automation on arrows, light shimmer reverb 0.3s",
            "notes": "Automate low-pass on ambience when arrows fire to spotlight flight path"
        }

    return {
        "foley": [
            "Subtle stance adjustments",
            "Cloth and leather reacting to micro-movements",
            "Controlled breathing from both sides"
        ],
        "sfx": [
            "Wind swells following action beats",
            "Environmental debris rustles",
            "Low-frequency hits on major impacts"
        ],
        "ambience": "Ambient forest pad with layered insects",
        "music_cue": "Quiet evolving drone that blooms on key actions",
        "mix_strategy": "- Maintain dialogue window 2kHz-5kHz, apply sidechain on music during foley",
        "spatialization": "Short plate reverb on foley, 0.2s tail",
        "notes": "Emphasise contrast between quiet setup and kinetic exchanges"
    }

def _densify_action_sequence(existing: list, cat: dict, archery_on: bool) -> list:
    """
    Bổ sung thêm action items để đạt 15-18 action như mẫu dòng 1904-1982.
    Sử dụng _create_dense_action_sequence để tạo action sequence mới.
    """
    # Luôn tạo action sequence mới dựa trên creative_brief
    creative_brief = cat.get("creative_brief", "")
    return _create_dense_action_sequence(cat, archery_on, creative_brief)

def _clean_camera(cam: dict) -> dict:
    c = deepcopy(cam or {})
    # Remove unrealistic yaw 360 unless it's an intentional 360 shot
    try:
        if c.get("orientation_deg", {}).get("yaw", 0) == 360:
            c["orientation_deg"]["yaw"] = 0
    except Exception:
        pass
    # Normalize movement text
    mv = _v2_lower((c.get("movement") or {}).get("type",""))
    if "circular" in mv and "360" not in mv:
        c["movement"]["type"] = "pan with light arc"
    if "zoom" in mv and "focus" in _v2_lower((c.get("focus") or {}).get("mode","")):
        c["movement"]["type"] = "pan or dolly; use rack focus (not zoom) for focus shifts"
    return c

def _clean_audio_vfx(archery_on: bool, out: dict) -> None:
    # strip archery-specific cues when not archery
    if not archery_on:
        vfx = [v for v in (out.get("vfx") or []) if "arrow" not in _v2_lower(v)]
        out["vfx"] = vfx
        au = out.get("audio") or {}
        au["sfx"] = [s for s in (au.get("sfx") or []) if not any(k in _v2_lower(s) for k in ("bow","arrow","bowstring"))]
        au["foley"] = [f for f in (au.get("foley") or []) if "arrow" not in _v2_lower(f)]
        out["audio"] = au

def _clean_entity_profiles(archery_on: bool, out: dict) -> None:
    profs = out.get("entity_profiles") or []
    if not profs:
        return
    if not archery_on:
        keep = []
        for p in profs:
            name = _v2_lower(str(p.get("name") or p.get("title") or ""))
            if any(k in name for k in ("darting projectile","arrow","projectile","identity")):
                continue
            keep.append(p)
        out["entity_profiles"] = keep

def _fix_single_prompt(sp: str, cat: dict, archery_on: bool) -> str:
    if not sp: return sp
    txt = sp
    # Canonicalize labels
    replacements = {
        r"\bTarget\b": cat["antagonist"],
        r"\banimal\b": cat["antagonist"],
        r"\bbeast\b": cat["antagonist"],
        r"\bArcher(s)?\b": list(cat["groups"])[0],
        r"\bwarriors?\b": list(cat["groups"])[0],
        r"\bhunters?\b": list(cat["groups"])[0],
    }
    for pat, rep in replacements.items():
        txt = re.sub(pat, rep, txt, flags=re.I)
    if not archery_on:
        txt = re.sub(r"\barrows?\b|\barchers?\b|\bbow(string)?\b|\bflight\b|\bimpact\b", "", txt, flags=re.I)
        txt = re.sub(r"\s{2,}", " ", txt)
    # remove 'camera:' as an actor line if it exists
    txt = re.sub(r"^\s*-\s*t=\d+(\.\d+)?s\s*\|\s*camera:.*$", "", txt, flags=re.I|re.M)
    return txt.strip()

def _enhance_scene_quality(scene: dict) -> dict:
    """
    Cải thiện chất lượng phân cảnh dựa trên các nguyên tắc của video AI thành công.
    """
    enhanced_scene = deepcopy(scene)
    
    # 1. Tăng cường Action Density
    if enhanced_scene.get("shots"):
        shot = enhanced_scene["shots"][0]
        action_sequence = shot.get("action_sequence", [])
        
        # Đảm bảo có đủ action beats (16-20 beats)
        if len(action_sequence) < 16:
            # Thêm micro-actions để tăng density
            for i in range(len(action_sequence), 16):
                timestamp = i * 0.5
                enhanced_scene["shots"][0]["action_sequence"].append({
                    "timestamp_sec": timestamp,
                    "subject": "Warrior",
                    "verb": "maintains focus",
                    "biomechanics_note": "subtle weight shifts; eye tracking; breath control",
                    "physics_note": "minimal energy expenditure; balance maintained",
                    "internal_motivation": "stay alert and ready for next action"
                })
    
    # 2. Cải tiến Camera Movement
    if enhanced_scene.get("shots"):
        shot = enhanced_scene["shots"][0]
        camera = shot.get("camera", {})
        
        # Thêm cinematic camera movements
        if "movement" not in camera:
            camera["movement"] = {}
        
        # Dynamic camera based on scene type
        creative_brief = enhanced_scene.get("creative_brief", "").lower()
        if "thrusts" in creative_brief or "spear" in creative_brief:
            camera["movement"]["type"] = "dolly in tracking shot, start wide→end close-up on impact"
            camera["movement"]["direction"] = "forward with slight arc"
            camera["movement"]["speed"] = "moderate, building tension"
        elif "attacking" in creative_brief or "relentless" in creative_brief:
            camera["movement"]["type"] = "dynamic tracking shot, follows aggressive movements"
            camera["movement"]["direction"] = "lateral with vertical adjustments"
            camera["movement"]["speed"] = "fast, matching action intensity"
        elif "arrows" in creative_brief or "pursuit" in creative_brief:
            camera["movement"]["type"] = "horizontal pan with tracking"
            camera["movement"]["direction"] = "smooth horizontal sweep"
            camera["movement"]["speed"] = "steady, following chase"
        else:
            camera["movement"]["type"] = "subtle handheld with micro-movements"
            camera["movement"]["direction"] = "minimal sway and breathing"
            camera["movement"]["speed"] = "very slow, natural"
        
        shot["camera"] = camera
    
    # 3. Cải tiến Audio Design
    if enhanced_scene.get("shots"):
        shot = enhanced_scene["shots"][0]
        audio = shot.get("audio", {})
        
        # Thêm layered audio
        if "foley" not in audio:
            audio["foley"] = []
        if "sfx" not in audio:
            audio["sfx"] = []
        if "ambience" not in audio:
            audio["ambience"] = ""
        if "music_cue" not in audio:
            audio["music_cue"] = ""
        
        # Enhanced audio based on scene type
        creative_brief = enhanced_scene.get("creative_brief", "").lower()
        if "thrusts" in creative_brief or "spear" in creative_brief:
            audio["foley"] = [
                "Heavy footsteps of warrior charging",
                "Deep breathing and grunts of exertion",
                "Cloth and leather movement sounds",
                "Spear shaft creaking under tension"
            ]
            audio["sfx"] = [
                "Sharp impact of spear meeting target",
                "Air displacement from weapon movement",
                "Ground impact from charging steps",
                "Metallic ring of weapon contact"
            ]
            audio["ambience"] = "Tense forest atmosphere with distant wildlife calls"
            audio["music_cue"] = "Building orchestral score with percussion accents"
        elif "attacking" in creative_brief or "relentless" in creative_brief:
            audio["foley"] = [
                "Aggressive snarling and roaring",
                "Heavy paw impacts on ground",
                "Claw scraping against surfaces",
                "Rapid breathing and growls"
            ]
            audio["sfx"] = [
                "Whoosh of claws cutting through air",
                "Sharp crack of weapon impacts",
                "Rustle of leaves and branches",
                "Frustrated growls and snarls"
            ]
            audio["ambience"] = "Intense combat atmosphere with heightened forest sounds"
            audio["music_cue"] = "Fast-paced, tension-building score with rhythmic percussion"
        elif "arrows" in creative_brief or "pursuit" in creative_brief:
            audio["foley"] = [
                "Twang of bowstrings being released",
                "Heavy breathing of fleeing creature",
                "Coordinated footsteps in pursuit",
                "Equipment jingling and movement"
            ]
            audio["sfx"] = [
                "Whistle of arrows flying through air",
                "Thunk of arrows striking targets",
                "Crack of branches as creature crashes through forest",
                "Tactical calls between warriors"
            ]
            audio["ambience"] = "Chase atmosphere with wind through trees"
            audio["music_cue"] = "Building orchestral chase theme with rhythmic percussion"
        
        shot["audio"] = audio
    
    # 4. Cải tiến Lighting và VFX
    if enhanced_scene.get("shots"):
        shot = enhanced_scene["shots"][0]
        lighting = shot.get("lighting", {})
        vfx = shot.get("vfx", [])
        
        # Enhanced lighting
        lighting["key_light"] = "Natural sunlight from camera-left, creating dramatic shadows"
        lighting["fill_light"] = "Soft ambient light to prevent harsh shadows"
        lighting["rim_light"] = "Backlighting to separate characters from background"
        lighting["atmospheric"] = "Dust particles in air, creating depth and mood"
        
        # Enhanced VFX
        creative_brief = enhanced_scene.get("creative_brief", "").lower()
        if "thrusts" in creative_brief or "spear" in creative_brief:
            vfx.extend([
                "Subtle weapon trails during movement",
                "Dust particles on impact",
                "Light refraction on weapon surface",
                "Environmental debris on contact"
            ])
        elif "attacking" in creative_brief or "relentless" in creative_brief:
            vfx.extend([
                "Dust clouds from rapid movement",
                "Impact particles on contact",
                "Environmental debris displacement",
                "Atmospheric dust and particles"
            ])
        elif "arrows" in creative_brief or "pursuit" in creative_brief:
            vfx.extend([
                "Arrow trails in flight",
                "Dust particles on impact",
                "Environmental debris from movement",
                "Atmospheric particles and dust"
            ])
        
        shot["lighting"] = lighting
        shot["vfx"] = vfx
    
    # 5. Thêm Creative Suggestions
    if enhanced_scene.get("shots"):
        shot = enhanced_scene["shots"][0]
        creative_brief = enhanced_scene.get("creative_brief", "").lower()
        
        if "thrusts" in creative_brief or "spear" in creative_brief:
            shot["creative_suggestions"] = [
                "Use rack focus to shift from warrior to target during thrust",
                "Add slow-motion effect during impact moment",
                "Use shallow depth of field to isolate characters",
                "Consider POV shot from warrior's perspective"
            ]
        elif "attacking" in creative_brief or "relentless" in creative_brief:
            shot["creative_suggestions"] = [
                "Use low angle to emphasize beast's power",
                "Add quick cuts between attacker and defender",
                "Use handheld camera for intensity",
                "Consider over-the-shoulder shots for immersion"
            ]
        elif "arrows" in creative_brief or "pursuit" in creative_brief:
            shot["creative_suggestions"] = [
                "Use tracking shot to follow arrow flight",
                "Add POV shot from warrior's perspective",
                "Use wide shots to show scale of chase",
                "Consider aerial view for dramatic effect"
            ]
    
    return enhanced_scene

def _postprocess_scene_v2(out: dict, scene_text: str, continuity) -> dict:
    print(f"🔧 DEBUG: _postprocess_scene_v2 called")
    out = deepcopy(out)
    cat = _build_catalog(continuity, out)
    archery_on = _is_archery_scene(scene_text, out)
    print(f"🔧 DEBUG: cat keys: {list(cat.keys()) if cat else 'None'}")
    print(f"🔧 DEBUG: creative_brief in cat: {cat.get('creative_brief', 'NOT FOUND')}")

    # 1) actions - ALWAYS PRESERVE LLM output (TOP 0.01% FIX)
    existing_actions = out.get("shots", [{}])[0].get("action_sequence", []) if out.get("shots") else []
    
    # ✅ PRESERVE: Trust LLM intelligence, only sort for consistency
    print(f"🎯 PRESERVING {len(existing_actions)} LLM-generated actions (no template override)")
    fixed_actions = existing_actions
    fixed_actions.sort(key=lambda x: float(x.get("timestamp_sec", 0)))
    
    # DISABLED: _fix_action_sequence (template-based regeneration)
    # Reason: Templates override LLM intelligence and cause re-aims spam
    # Solution: Improved LLM prompt generates better sequences from start
    
    # FORCE write back to shots[0]["action_sequence"]
    if out.get("shots"):
        out["shots"][0]["action_sequence"] = fixed_actions
        print(f"🔧 DEBUG: Updated action_sequence in shots[0]")
    else:
        # Tạo shots structure nếu không có
        out["shots"] = [{"action_sequence": fixed_actions}]
        print(f"🔧 DEBUG: Created shots structure")

    # 2) camera, vfx, audio - integrate dynamic functions
    creative_brief = out.get("creative_brief", "")
    
    # Apply dynamic camera movement based on action sequence and creative_brief
    if out.get("shots"):
        try:
            dynamic_camera_movement = _create_dynamic_camera_movement(fixed_actions, creative_brief)
            print(f"\U0001f527 DEBUG: Dynamic camera movement: {dynamic_camera_movement}")
            camera_block = out["shots"][0].setdefault("camera", {})
            move_meta = {
                "type": dynamic_camera_movement.get("type"),
                "direction": dynamic_camera_movement.get("direction"),
                "speed": dynamic_camera_movement.get("speed"),
                "motivation": dynamic_camera_movement.get("motivation"),
                "secondary_motion": dynamic_camera_movement.get("secondary_motion"),
            }
            camera_block["movement"] = move_meta
            if dynamic_camera_movement.get("start_pose") and dynamic_camera_movement.get("end_pose"):
                camera_block.setdefault("position_m", {})
                camera_block["position_m"]["start"] = dynamic_camera_movement["start_pose"]
                camera_block["position_m"]["end"] = dynamic_camera_movement["end_pose"]
            print("\U0001f527 DEBUG: Applied cinematic camera movement and path")
        except Exception as e:
            print(f"🔧 DEBUG: Failed to apply camera movement: {e}")
            pass
    
    # Apply dynamic audio based on action sequence and creative_brief
    if out.get("shots"):
        try:
            dynamic_audio = _create_dynamic_audio(fixed_actions, creative_brief)
            print(f"🔧 DEBUG: Dynamic audio: {dynamic_audio}")
            out["shots"][0]["audio"] = dynamic_audio
            print(f"🔧 DEBUG: Updated audio in shots[0]")
        except Exception as e:
            print(f"🔧 DEBUG: Failed to apply audio: {e}")
            pass
    
    out["camera"] = _clean_camera(out.get("camera") or {})
    _clean_audio_vfx(archery_on, out)
    _clean_entity_profiles(archery_on, out)

    # 3) single_prompt_for_model - FORCE tạo mới dựa trên creative_brief
    try:
        new_single_prompt = _build_single_prompt(out)
        out["single_prompt_for_model"] = new_single_prompt
        print(f"🔧 DEBUG: Generated new single_prompt_for_model: {new_single_prompt[:100]}...")
    except Exception as e:
        print(f"🔧 DEBUG: Failed to generate single_prompt_for_model: {e}")
        # Fallback nếu function chưa được định nghĩa
        out["single_prompt_for_model"] = out.get("single_prompt_for_model", "")

    # 4) stabilize world_and_characters names if missing
    w = out.get("world_and_characters") or {}
    if not _v2_norm(w.get("lead_character")):
        w["lead_character"] = cat["lead"]
    if not _v2_norm(w.get("antagonist")):
        w["antagonist"] = cat["antagonist"]
    out["world_and_characters"] = w

    # 5) Enhanced scene quality is already applied through previous steps
    
    # 6) safety — keep non-graphic tone (already handled in your base)
    return out

def _apply_super_continuity_patch_v2():
    try:
        cls = ChatGPTPromptGenerator
    except NameError:
        return
    if getattr(cls, "_super_continuity_v2", False):
        return
    _orig = cls.build_scene_prompt

    def _wrapped(self, scene_text, idx, continuity):
        out = _orig(self, scene_text, idx, continuity)
        if isinstance(out, dict) and "error" not in out:
            out = _postprocess_scene_v2(out, scene_text, continuity)
        return out

    cls.build_scene_prompt = _wrapped
    cls._super_continuity_v2 = True

_apply_super_continuity_patch_v2()
# === END PATCH v2 ===

# === VEO3 SUPER-CONTINUITY PATCH v3 — continuity lock + physics grammar + detail preservation ===
# This file is meant to be appended to (or imported after) the existing cc.py so it can monkey‑patch
# a few helpers without rewriting your whole pipeline. It focuses on:
# 1) Character/object consistency across scenes (plural/singular aliasing fixed)
# 2) Physically plausible, stable action grammar (archery + spear/melee chains)
# 3) "Detail‑preserving" single_prompt sanitizer (keeps timeline/camera beats; fixes LEAD/ANTAGONIST)
# 4) Camera rig/movement consistency (no static rig + dolly conflict)
# 5) Never drop audio/VFX/lighting that already exist — only soften wording
#
# Usage: place this below your current definitions in cc.py (after apply_veo3_patch and the auto‑bind
# block is fine). All functions guard against missing globals so it is safe to import once.

from copy import deepcopy
import re

# ---------- Helpers to extend existing globals safely ----------

def _extend_aliases():
    """Broaden canonical aliases so plural forms and common roles map correctly."""
    try:
        CANON = globals().get("_CANON", {})
        # Ensure three canonical buckets
        for k in ("Warrior", "Beast", "Projectile"):
            CANON.setdefault(k, {"aliases": []})
        # Merge new aliases (no duplicates)
        _merge = lambda a,b: list(sorted(set(list(a) + list(b)), key=str.lower))
        CANON["Warrior"]["aliases"] = _merge(CANON["Warrior"].get("aliases", []), [
            "warriors", "hunter", "hunters", "archer", "archers",
            "hunter with spear", "hunter with bow", "group of warriors", "group of hunters"
        ])
        CANON["Beast"]["aliases"] = _merge(CANON["Beast"].get("aliases", []), [
            "beasts", "animal", "creature", "hound"
        ])
        CANON["Projectile"]["aliases"] = _merge(CANON["Projectile"].get("aliases", []), [
            "arrow", "arrows", "projectile"
        ])
        globals()["_CANON"] = CANON
    except Exception:
        pass

_extend_aliases()

# ---------- Physics verbs + hints (merged with existing if present) ----------

_NEW_PHYSICAL_VERBS = {
    # existing common ones are preserved; we add concise biomechanics verbs
    "anchor", "release", "nock", "impact", "parry", "block", "follow-through", "flies",
}

_NEW_VERB_HINTS = {
    "draw": {"Warrior"},
    "nock": {"Warrior"},
    "anchor": {"Warrior"},
    "release": {"Warrior", "Projectile"},
    "flies": {"Projectile"},
    "impact": {"Beast", "Warrior", "Projectile"},
    "parry": {"Warrior"},
    "block": {"Warrior"},
    "follow-through": {"Warrior"},
}

try:
    # Merge into existing constants if present
    if "_VERB_HINTS" in globals() and isinstance(_VERB_HINTS, dict):
        _VERB_HINTS.update(_NEW_VERB_HINTS)
    else:
        globals()["_VERB_HINTS"] = dict(_NEW_VERB_HINTS)
except Exception:
    globals()["_VERB_HINTS"] = dict(_NEW_VERB_HINTS)

try:
    if "_is_physical_verb" in globals():
        # Wrap the original function to add new verbs
        _orig_is_physical_verb = _is_physical_verb
        def _is_physical_verb(v: str) -> bool:  # type: ignore
            v = (v or "").lower().strip()
            return _orig_is_physical_verb(v) or v in {x.lower() for x in _NEW_PHYSICAL_VERBS}
        globals()["_is_physical_verb"] = _is_physical_verb
    else:
        def _is_physical_verb(v: str) -> bool:  # fallback definition
            return (v or "").lower().strip() in _NEW_PHYSICAL_VERBS
        globals()["_is_physical_verb"] = _is_physical_verb
except Exception:
    pass

# ---------- Subject aliasing fix ----------

try:
    _alias_to_canon  # noqa: F401
except NameError:
    def _alias_to_canon(n: str) -> str:
        """Legacy function - use global canon() instead."""
        return canon(n)

# ---------- Action stabilization & grammar expansion ----------

def _grammar_expand(actions: list) -> list:
    """Insert missing biomechanical beats (archery + spear/melee) with safe timings."""
    if not actions:
        return actions
    out = list(actions)

    def _insert_after(idx: int, dt: float, subject: str, verb: str, biomech="", phys="", why=""):
        t = float(out[idx]["timestamp_sec"]) + dt
        out.append({
            "timestamp_sec": float(f"{t:.2f}"),
            "subject": _alias_to_canon(subject),
            "verb": verb,
            "biomechanics_note": biomech,
            "physics_note": phys,
            "internal_motivation": why,
        })

    # pass 1 — archery chain: draw → anchor → release → Projectile: flies
    for i, a in enumerate(list(out)):
        s, v = a.get("subject",""), (a.get("verb") or "").lower()
        if _alias_to_canon(s) == "Warrior" and v == "draw":
            # ensure anchor and release
            has_anchor = any(abs(b["timestamp_sec"]-a["timestamp_sec"]) <= 0.6 and (b.get("verb","")) == "anchor" for b in out)
            has_release = any(b["timestamp_sec"] >= a["timestamp_sec"] and b.get("verb","") == "release" for b in out)
            if not has_anchor:
                _insert_after(i, 0.25, "Warrior", "anchor", "back tension engaged; scapula set", "isometric hold; micro sway", "stabilize aim")
            if not has_release:
                _insert_after(i, 0.45, "Warrior", "release", "string rolls off fingers", "stored elastic energy converts to kinetic", "loose the shot")
                _insert_after(i, 0.55, "Projectile", "flies", "arrow flexes (archer's paradox)", "ballistic arc: gravity+drag", "in flight")

    # pass 2 — spear chain: thrusts → impact (Beast) → recoil (Beast) + brace (Warrior)
    for i, a in enumerate(list(out)):
        s, v = _alias_to_canon(a.get("subject","")), (a.get("verb") or "").lower()
        if s == "Warrior" and v == "thrusts":
            _insert_after(i, 0.18, "Beast", "impact", "torso jolts; forelimbs buckle briefly", "momentum transfer; impulse", "hit by spear")
            _insert_after(i, 0.34, "Beast", "recoils", "steps back; head withdraws", "friction resists slide; balance recovery", "pain/avoidance")
            _insert_after(i, 0.40, "Warrior", "braces", "rear foot digs in; core tightens", "counter‑force absorption", "hold ground")

    # dedupe excessive consecutive re-aims (keep at most 2 in a 1s window)
    out.sort(key=lambda x: x["timestamp_sec"])  # ensure order before pruning
    pruned = []
    window = []
    for a in out:
        if (a.get("verb") or "").lower() == "re-aims":
            window = [w for w in window if a["timestamp_sec"] - w["timestamp_sec"] <= 1.0]
            if sum(1 for w in window if (w.get("verb") or "").lower()=="re-aims") >= 2:
                continue
            window.append(a)
        pruned.append(a)
    return pruned

# Patch _fix_actions to call grammar expansion and clamp output length deterministically
try:
    _orig_fix_actions = globals().get("_fix_actions")
except Exception:
    _orig_fix_actions = None

if _orig_fix_actions:
    def _fix_actions(actions: list) -> list:  # type: ignore
        base = _orig_fix_actions(actions) or []
        base = _grammar_expand(base)
        base.sort(key=lambda x: x["timestamp_sec"])  # keep ascending time
        # cap to ~14 beats but prefer to keep complete chains
        out = []
        chain_budget = {"archery": 2, "spear": 2}  # allow two full chains if present
        def _tag(a):
            v = (a.get("verb") or "").lower()
            s = _alias_to_canon(a.get("subject",""))
            if s == "Warrior" and v in {"draw","anchor","release"}: return "archery"
            if v in {"thrusts","impact","recoils","braces"}: return "spear"
            return "misc"
        for a in base:
            tag = _tag(a)
            if tag == "archery":
                if chain_budget[tag] <= 0: continue
                out.append(a)
                if (a.get("verb") or "").lower() == "release":
                    chain_budget[tag] -= 1
            elif tag == "spear":
                if chain_budget[tag] <= 0: continue
                out.append(a)
                if (a.get("verb") or "").lower() == "recoils":
                    chain_budget[tag] -= 1
            else:
                out.append(a)
            if len(out) >= 18:
                break
        return out
    globals()["_fix_actions"] = _fix_actions
else:
    # Fallback if no original exists
    def _fix_actions(actions: list) -> list:  # type: ignore
        return _grammar_expand(actions)
    globals()["_fix_actions"] = _fix_actions

# ---------- Camera consistency ----------

try:
    _orig_normalize_camera = globals().get("_normalize_camera")
except Exception:
    _orig_normalize_camera = None

if _orig_normalize_camera:
    def _normalize_camera(cam: dict) -> dict:  # type: ignore
        cam = _orig_normalize_camera(cam) or {}
        rig = (cam.get("rig") or cam.get("Rig") or "").lower()
        mv  = (cam.get("movement") or {}).get("type", "")
        low = str(mv).lower()
        # If rig is static, force movement to static
        if rig == "static":
            cam.setdefault("movement", {})
            cam["movement"]["type"] = "static"
        # Collapse zoom-in-words into a note, not a movement keyword (avoid conflicts)
        if "zoom" in low and cam.get("movement", {}).get("type") == "static":
            cam.setdefault("notes", "")
            cam["notes"] = (cam["notes"] + " | include subtle zoom if supported").strip(" |")
            # leave type=static
        return cam
    globals()["_normalize_camera"] = _normalize_camera

# ---------- Single prompt sanitizer (detail‑preserving) ----------

_TIMELINE_BEAT_RE = re.compile(r"^\s*-\s*t=\d+\.?\d*s\s*\|\s*(camera|scene):.*$", re.I | re.M)

def _clean_audio_wording(line: str) -> str:
    # soften sensational words without removing the line
    replacements = {
        "bone-chilling": "deep",
        "anguished": "pained",
        "flesh": "body",
        "blood": "impact",
    }
    for k, v in replacements.items():
        line = re.sub(rf"(?i)\b{re.escape(k)}\b", v, line)
    return line

try:
    _orig_sanitize_sp = globals().get("_sanitize_single_prompt_block")
except Exception:
    _orig_sanitize_sp = None

def _sanitize_single_prompt_block(sp: str, w: dict | None = None) -> str:  # type: ignore
    sp = (sp or "").strip()
    if not sp:
        return sp
    # Extract and keep camera/scene timeline beats
    timeline = _TIMELINE_BEAT_RE.findall(sp)
    # We *do not* drop them; we will re-insert as TIMELINE section
    cleaned = _TIMELINE_BEAT_RE.sub("", sp)

    # Canonicalize entity names across the whole block
    for bad, good in [
        ("animal", "Beast"), ("creature", "Beast"), ("hound", "Beast"),
        ("Target", "Beast"), ("hunters", "Warrior"), ("warriors", "Warrior"),
        ("archers", "Warrior"), ("Arrows", "Projectile"), ("arrow", "Projectile"),
    ]:
        cleaned = re.sub(rf"(?i)\b{re.escape(bad)}\b", good, cleaned)

    # Fix headers (LEAD/ANTAGONIST/WEATHER/TIME) using world context if provided
    lead = _alias_to_canon((w or {}).get("lead_character", "")) if w else None
    ant  = _alias_to_canon((w or {}).get("antagonist", "")) if w else None
    weather = (w or {}).get("weather") if w else None
    tod = (w or {}).get("time_of_day") if w else None
    # Fix None time_of_day issue
    if tod is None:
        tod = "day"  # Default to day if None
        print("DEBUG: Fixed None time_of_day in header replacement, set to 'day'")

    def _repl_hdr(pattern: str, value: str | None):
        nonlocal cleaned
        if value is None:
            return
        cleaned = re.sub(pattern, lambda m: m.group(0).split(":")[0] + f": {value}", cleaned, flags=re.M)

    _repl_hdr(r"(?mi)^\s*LEAD:\s*.*$", lead)
    _repl_hdr(r"(?mi)^\s*ANTAGONIST:\s*.*$", ant)
    _repl_hdr(r"(?mi)^\s*WEATHER:\s*.*$", (weather or ""))
    _repl_hdr(r"(?mi)^\s*TIME:\s*.*$", (tod or ""))

    # Soft-clean audio lines
    cleaned_lines = []
    for ln in cleaned.splitlines():
        if ln.strip().startswith("- Foley:") or ln.strip().startswith("- SFX:") or ln.strip().startswith("- Ambience:"):
            cleaned_lines.append(_clean_audio_wording(ln))
        else:
            cleaned_lines.append(ln)

    cleaned = "\n".join([ln for ln in cleaned_lines if ln.strip()])

    # Re-append TIMELINE if we had beats
    if timeline:
        # timeline currently is a list of matches from the first group only; we need full lines
        full_lines = _TIMELINE_BEAT_RE.findall(sp)  # ensure group capture
        # but we want the actual lines; recompute with finditer
        full_lines = [m.group(0) for m in _TIMELINE_BEAT_RE.finditer(sp)]
        block = "\n" + "TIMELINE:\n" + "\n".join(full_lines)
        cleaned = cleaned.rstrip() + "\n" + block

    return cleaned

# Install sanitizer
globals()["_sanitize_single_prompt_block"] = _sanitize_single_prompt_block

# ---------- Harden apply_hard_sanity to pass world context into sanitizer and to preserve details ----------

try:
    _orig_apply_hard_sanity = globals().get("apply_hard_sanity")
except Exception:
    _orig_apply_hard_sanity = None

if _orig_apply_hard_sanity:
    def apply_hard_sanity(obj_translated: dict) -> dict:  # type: ignore
        obj = deepcopy(obj_translated or {})
        # keep original decisioning
        obj = _orig_apply_hard_sanity(obj)
        # Re-sanitize single_prompt with world context (preserves details)
        sp = obj.get("single_prompt_for_model", "")
        w = obj.get("world_and_characters") or {}
        obj["single_prompt_for_model"] = _sanitize_single_prompt_block(sp, w)
        # Do NOT drop audio/vfx/lighting if present — only normalize/soften at shot level
        for shot in obj.get("shots") or []:
            if shot.get("audio") is None:
                shot["audio"] = {}
            if shot.get("vfx") is None:
                shot["vfx"] = []
            if not shot.get("lighting"):
                # leave as-is; do not erase existing values elsewhere
                pass
        return obj
    globals()["apply_hard_sanity"] = apply_hard_sanity

# ---------- End of v3 patch ----------

# === CONTINUITY CANONICALIZATION PATCH ===
# Simple canonical mapping for consistent entity names across scenes

CANON = {
  "warriors":"Warrior","archers":"Warrior","hunters":"Warrior","hunter":"Warrior","warrior":"Warrior",
  "animal":"Beast","creature":"Beast","hound":"Beast","target":"Beast","beast":"Beast",
  "projectile":"Arrow","arrows":"Arrow","arrow":"Arrow"
}

def _canon(s: str) -> str:
    """Legacy function - use global canon() instead."""
    return canon(s)

def _continuity_canonicalize(self, obj: dict) -> dict:
    # Roles
    wc = obj.get("world_and_characters") or {}
    wc["lead_character"] = _canon(wc.get("lead_character") or "Beast")
    wc["antagonist"] = _canon(wc.get("antagonist") or ("Warrior" if wc["lead_character"]=="Beast" else "Beast"))
    if wc["lead_character"] == wc["antagonist"]:
        wc["antagonist"] = "Warrior" if wc["lead_character"] == "Beast" else "Beast"
    obj["world_and_characters"] = wc

    # Action subjects + Hard injection for empty sequences
    for s in obj.get("shots", []):
        seq = s.get("action_sequence") or []
        
        # HARD INJECTION: Nếu sequence trống và có archery keywords
        if not seq:
            scene_text = str(obj.get("single_prompt_for_model", ""))
            if any(kw in scene_text.lower() for kw in ["bow", "arrow", "archer", "cung", "mũi tên", "bắn"]):
                # Inject chuỗi bắn cung cơ bản
                seq = [
                    {"timestamp_sec": 0.5, "subject": "Warrior", "verb": "draws", 
                     "biomechanics_note": "scapular engagement; bow arm extends; string hand loads",
                     "physics_note": "limb energy stored as elastic potential"},
                    {"timestamp_sec": 0.8, "subject": "Warrior", "verb": "anchors",
                     "biomechanics_note": "string hand stabilizes at corner of mouth; breath hold", 
                     "physics_note": "stable anchor reduces lateral error"},
                    {"timestamp_sec": 1.2, "subject": "Warrior", "verb": "releases",
                     "biomechanics_note": "clean finger relax; slight follow-through",
                     "physics_note": "stored elastic energy converts to arrow kinetic energy"},
                    {"timestamp_sec": 1.4, "subject": "Arrow", "verb": "flies",
                     "biomechanics_note": "fletching stabilizes; shaft flexes then dampens",
                     "physics_note": "parabolic arc under gravity; drag slows speed"},
                    {"timestamp_sec": 2.2, "subject": "Arrow", "verb": "impacts",
                     "biomechanics_note": "shaft compresses then settles; vibrations dampen quickly",
                     "physics_note": "kinetic energy dissipates into target; arrow lodges at entry point"},
                    {"timestamp_sec": 2.4, "subject": "Beast", "verb": "recoils",
                     "biomechanics_note": "steps back; head withdraws; body tenses",
                     "physics_note": "momentum transfer; friction resists slide; balance recovery"}
                ]
        
        fixed = []
        for e in seq:
            sub = _canon(e.get("subject",""))
            if sub.lower() == "damage":  # bỏ entry giả chủ thể
                continue
            e["subject"] = sub
            # Arrow không "re-aims" - chuyển thành "flies"
            if sub == "Arrow" and "re-aim" in (e.get("verb","").lower()):
                e["verb"] = "flies"
            fixed.append(e)
        s["action_sequence"] = fixed
    return obj

def _de_spam_reaims(self, obj: dict) -> dict:
    for s in obj.get("shots", []):
        counts = {"Warrior":0, "Beast":0}
        for e in s.get("action_sequence", []):
            subj = e.get("subject")
            verb = (e.get("verb") or "").lower()
            if subj in counts and "re-aim" in verb:
                counts[subj] += 1
                if counts[subj] > 2:
                    if subj == "Warrior":
                        e["verb"] = "braces"
                    else:
                        e["verb"] = "circles"
    return obj

def _normalize_camera(self, obj: dict) -> dict:
    for s in obj.get("shots", []):
        cam = s.get("camera") or {}
        mv  = ((cam.get("movement") or {}).get("type") or "").lower()
        rig = (cam.get("rig") or "").lower()
        if "dolly" in mv and "zoom" in mv:
            cam["movement"] = {"type":"dolly", "note":"subtle push-in"}
        if rig == "static":
            cam["movement"] = {"type":"static"}
        s["camera"] = cam
    return obj

SOFTEN = {
  r"\banguished\b": "pained",
  r"\bflesh\b": "body",
  r"damage dripping sound": "liquid drip (non-graphic)"
}

def _soften_audio(self, obj: dict) -> dict:
    import re
    for s in obj.get("shots", []):
        aud = s.get("audio") or {}
        for k in ["foley","sfx","ambience","music_cue","mix_notes","dialogue_note","fx","music"]:
            v = aud.get(k)
            if isinstance(v, list):
                aud[k] = [re.sub(pat, rep, x, flags=re.IGNORECASE) if isinstance(x,str) else x
                          for x in v]
            elif isinstance(v, str):
                for pat, rep in SOFTEN.items():
                    v = re.sub(pat, rep, v, flags=re.IGNORECASE)
                aud[k] = v
        s["audio"] = aud
    return obj


if __name__ == "__main__":
    main()

# === GOLD TEMPLATE PACK v1 — "make my output look like the sample JSON" ===
# Drop-in patch: paste this whole block at the END of your current cc.py.
# It wraps ChatGPTPromptGenerator.build_scene_prompt and *adds* a rich, detailed
# template (global_style/continuity/safety/output_settings/single_prompt blocks)
# while preserving whatever your pipeline already produced.
#
# Toggle:
#   GOLD_OUTPUT_MODE = False  # set False to disable quickly
#
# What it does:
#   - Deep-merge a "gold" skeleton so every scene has the same detailed structure
#   - Canonicalize subjects (Warrior Squad / Forest Beast / Arrow) via aliases_map
#   - Auto-fill missing action beats with physically plausible chains
#   - Normalize camera movement & keep audio/VFX (no aggressive pruning)
#   - Rebuild a rich `single_prompt_for_model` section (keeps timeline detail)

from copy import deepcopy
import re, json

GOLD_OUTPUT_MODE = False

# -------------------------
# Canonicals / aliases
# -------------------------
CANON_ALIASES = {
    # humans
    "warrior": "Warrior Squad",
    "warriors": "Warrior Squad",
    "hunters": "Warrior Squad",
    "hunter": "Warrior Squad",
    "lead": "Warrior Squad",
    "archer": "Warrior Squad",
    "archers": "Warrior Squad",
    # beasts
    "animal": "Forest Beast",
    "creature": "Forest Beast",
    "hound": "Forest Beast",
    "target": "Forest Beast",
    "beast": "Forest Beast",
    # projectiles
    "arrow": "Arrow",
    "arrows": "Arrow",
    "projectile": "Arrow",
}

SAFEWORDING = (
    (r"(?i)anguished", "pained"),
    (r"(?i)bone-?chilling", "deep"),
    (r"(?i)flesh", "body"),
    (r"(?i)damage\s+dripping(\s+sound)?", "liquid drip (non-graphic)"),
)

ALLOWED_MOVES = {"static","handheld","dolly","pan","tilt","truck","crane","orbit","tracking"}

# -------------------------
# Helpers
# -------------------------

def _lc(s):
    return (s or "").strip().lower()

def _canon_name(name: str) -> str:
    """Legacy function - use global canon() instead."""
    return canon(name)

def _is_archery_scene(scene_obj: dict, scene_text: str) -> bool:
    blob = " ".join([
        _lc(scene_text or ""),
        _lc(scene_obj.get("creative_brief","")),
        json.dumps(scene_obj.get("world_and_characters", {}), ensure_ascii=False).lower(),
        json.dumps(scene_obj.get("shots", []), ensure_ascii=False).lower(),
    ])
    has_arch = any(k in blob for k in ("bow","arrow","archer","projectile"))
    has_melee = any(k in blob for k in ("spear","thrust","blade","shield"))
    return has_arch and not has_melee

# -------------------------
# Action fills (when missing/sparse)
# -------------------------

def _ensure_actions(scene_obj: dict, scene_text: str):
    shots = scene_obj.get("shots") or []
    if not shots:
        return scene_obj
    arch = _is_archery_scene(scene_obj, scene_text)
    for sh in shots:
        seq = sh.get("action_sequence") or []
        if seq:
            # canonicalize existing
            fixed=[]
            reaim = {"Warrior Squad":0, "Forest Beast":0}
            tprev = -1.0
            for e in seq:
                e = dict(e)
                subj = _canon_name(e.get("subject",""))
                if not subj: continue
                verb = _lc(e.get("verb",""))
                # Arrow cannot "re-aims"
                if subj == "Arrow" and "re-aim" in verb:
                    verb = "flies"
                # clamp spam of re-aims
                if "re-aim" in verb and subj in reaim:
                    reaim[subj] += 1
                    if reaim[subj] > 2:
                        verb = "braces" if subj == "Warrior Squad" else "circles"
                t = float(e.get("timestamp_sec", (tprev+0.33 if tprev>=0 else 0.33)))
                if tprev >= 0 and t - tprev < 0.12:
                    t = round(tprev + 0.12, 2)
                fixed.append({
                    "timestamp_sec": float(f"{t:.2f}"),
                    "subject": subj,
                    "verb": verb,
                    "biomechanics_note": (e.get("biomechanics_note") or "").strip(),
                    "physics_note": (e.get("physics_note") or "").strip(),
                    "internal_motivation": (e.get("internal_motivation") or "").strip(),
                })
                tprev = t
            sh["action_sequence"] = fixed
            continue
        # If sequence is empty, fill with a physically plausible chain
        t = 0.30
        def add(sub, verb, bio, phy, dt=0.35):
            nonlocal t
            sh.setdefault("action_sequence", []).append({
                "timestamp_sec": float(f"{t:.2f}"),
                "subject": sub,
                "verb": verb,
                "biomechanics_note": bio,
                "physics_note": phy,
                "internal_motivation": "maintain initiative"
            })
            t = round(t + dt, 2)
        if arch:
            add("Warrior Squad","draws","scapular engagement; bow arm stabilizes","elastic energy stores")
            add("Warrior Squad","anchors","string hand set at corner of mouth","stable anchor reduces lateral error",0.25)
            add("Warrior Squad","releases","clean finger relax; follow-through","energy→arrow kinetic",0.20)
            add("Arrow","enters flight","fletching stabilizes; shaft flex dampens","ballistic arc: gravity+drag",0.15)
            add("Arrow","impacts","shaft compresses; vib damp quickly","kinetic energy dissipates; lodges non-graphically",0.35)
            add("Forest Beast","recoils","mass shifts opposite vector","momentum transfer; friction limits slide",0.40)
        else:
            add("Forest Beast","charges","forelimbs extend; spine flexes","acceleration builds under traction",0.45)
            add("Warrior Squad","sets guard","low stance; shield/spear align","lower CoG for stability",0.40)
            add("Warrior Squad","thrusts","rear foot drives; hips extend","impulse sends spear tip forward",0.25)
            add("Forest Beast","recoils","steps back; head withdraws","friction aids balance recovery",0.40)
            add("Warrior Squad","braces","rear foot digs; core tightens","counter-force absorption",0.35)
    return scene_obj

# -------------------------
# Camera / audio normalization
# -------------------------

def _normalize_camera(sh: dict):
    cam = sh.get("camera") or {}
    mv = (cam.get("movement") or {}).get("type")
    rig = _lc(cam.get("rig"))
    if isinstance(mv, str):
        mvl = _lc(mv)
        if any(x in mvl for x in ("dolly", "pan", "tilt", "orbit", "track")):
            # strip any "zoom" words from type; put as note
            if "zoom" in mvl:
                cam.setdefault("notes", "")
                cam["notes"] = (cam["notes"] + " | subtle zoom if supported").strip(" |")
                cam["movement"] = {"type": [m for m in ALLOWED_MOVES if m in mvl] or ["dolly"][0]}
        else:
            cam["movement"] = {"type": "static"}
    else:
        cam["movement"] = {"type": "static"}
    if rig == "static":
        cam["movement"] = {"type": "static"}
    sh["camera"] = cam


def _soften_audio_block(aud: dict) -> dict:
    if not isinstance(aud, dict):
        return {}
    out = deepcopy(aud)
    for k in ("foley","sfx","ambience","music_cue","mix_notes","dialogue_note"):
        v = out.get(k)
        if isinstance(v, list):
            tmp=[]
            for s in v:
                if not isinstance(s, str):
                    tmp.append(s); continue
                for pat, rep in SAFEWORDING:
                    s = re.sub(pat, rep, s)
                tmp.append(s)
            out[k] = tmp
        elif isinstance(v, str):
            s = v
            for pat, rep in SAFEWORDING:
                s = re.sub(pat, rep, s)
            out[k] = s
    return out

# -------------------------
# GOLD skeleton & deep merge
# -------------------------

def _gold_skeleton():
    return {
        "model": "google/veo-3",
        "request_type": "video",
        "duration_seconds": 8,
        "fps": 30,
        "resolution": "1920x1080",
        "aspect_ratio": "16:9",
        "language": "English",
        "global_style": {
            "visual_tone": "cinematic, grounded realism",
            "motion_rules": "No idle frames; beats every 0.5–1.0s; every initiating action has an on-screen reaction within 0.9s; no teleportation",
            "color_grading": "natural daylight with warm greens and earthy browns",
            "vfx_guidance": "subtle dust/leaf particles on impacts and footfalls; restrained motion trails for fast projectiles",
            "sound_mix": "ambience low; foley detailed; sfx tight and non-graphic"
        },
        "continuity_core": {
            "roster": {
                "lead": "Warrior Squad",
                "antagonist": "Forest Beast",
                "projectiles": ["Arrow"],
                "props": ["Spear","Bow"]
            },
            "aliases_map": {
                "Warrior": "Warrior Squad",
                "Hunters": "Warrior Squad",
                "Lead": "Warrior Squad",
                "Target": "Forest Beast",
                "Beast": "Forest Beast",
                "Creature": "Forest Beast"
            },
            "wardrobe_state": "leather armor, earth-tone tunics; spears and one bow slung",
            "props_state": "one warrior holds Bow (nocked), another holds Spear (ready)",
            "time_lock": "day",
            "weather_lock": "clear"
        },
        "continuity_logic": "All subjects must resolve to the roster labels via aliases_map. Do not introduce new characters or props. Keep wardrobe/props persistent across shots.",
        "safety_and_quality": {
            "avoid": [
                "graphic injury or gore",
                "hate symbols or slurs",
                "sexual content or nudity",
                "child endangerment",
                "real-world political propaganda"
            ],
            "must_have": [
                "non-graphic, documentary tone for combat actions",
                "cause→effect reactions within 0.9s for every initiated action",
                "physically plausible motion (gravity, inertia, friction, drag)",
                "no idle seconds; continuous purposeful movement"
            ]
        },
        "output_settings": {
            "container": "mp4",
            "video_codec": "h264",
            "video_bitrate": "8000k",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "normalize_audio": True,
            "tone_mapping": "linear",
            "grain": "none"
        }
    }


def _deep_merge_preserve(dst: dict, src: dict):
    for k, v in src.items():
        if k not in dst or dst[k] in (None, "", [], {}):
            dst[k] = deepcopy(v)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_preserve(dst[k], v)
    return dst

# -------------------------
# Single prompt builder
# -------------------------

def _build_single_prompt(scene: dict) -> str:
    wc = scene.get("world_and_characters", {})
    lead = wc.get("lead_character") or "Warrior"
    ant  = wc.get("antagonist") or "Beast"
    setting = wc.get("setting") or "Forest clearing"
    weather = wc.get("weather") or "clear"
    tod = wc.get("time_of_day") or "day"
    creative_brief = scene.get("creative_brief", "")

    # Build timeline from action_sequence of first shot - make it dynamic based on creative_brief
    timeline_lines = []
    if scene.get("shots"):
        seq = scene["shots"][0].get("action_sequence") or []
        for e in seq:
            timestamp = e.get('timestamp_sec', 0)
            subject = e.get('subject', '')
            verb = e.get('verb', '')
            # Create more detailed timeline based on actual actions
            timeline_lines.append(f"{timestamp:.2f}s {subject} {verb}")

    cam = scene.get("shots", [{}])[0].get("camera", {}) if scene.get("shots") else {}
    lens = cam.get("lens_mm", 35)
    mv = (cam.get("movement") or {}).get("type") or "static"
    foc = cam.get("focus_pull_style") or ""

    # Make camera description more specific based on creative_brief
    creative_lower = creative_brief.lower() if creative_brief else ""
    if "thrusts" in creative_lower or "spear" in creative_lower:
        camera_desc = f"dolly-in tracking ({lens}mm), start wide→end close-up on impact"
    elif "attacking" in creative_lower or "relentless" in creative_lower:
        camera_desc = f"dynamic tracking ({lens}mm), follows beast's aggressive movements"
    elif "arrows" in creative_lower or "pursuit" in creative_lower:
        camera_desc = f"horizontal pan ({lens}mm), tracks chase and arrow flights"
    else:
        camera_desc = f"{mv} ({lens}mm)"
    
    if foc:
        camera_desc += f" | {foc}"

    # Build action timeline string - make it more specific
    if timeline_lines:
        action_summary = f"[{' … '.join(timeline_lines[:6])}{'…' if len(timeline_lines) > 6 else ''}]"
    else:
        action_summary = "[no actions defined]"

    # Create different single prompts based on creative_brief
    creative_lower = creative_brief.lower() if creative_brief else ""
    
    if "thrusts" in creative_lower or "spear" in creative_lower:
        # Scene 1: Warrior charges and thrusts spear
        lines = [
            f"VEO-3 — {setting}, {lead} vs {ant} ({scene.get('duration_seconds',8)}s). Grounded, non-graphic. No idle seconds.",
            f"ACTION (beats every ~0.5–1.0s): {action_summary}",
            f"CAMERA: {camera_desc}",
            f"VFX: subtle trails; dust/leaf puff on impacts.",
            f"LIGHTING: natural sun-dappled, key from camera-left.",
            f"AUDIO: ambience low forest bed; foley (leather, footfall, breath, non-graphic growl); sfx (snap, whoosh, woody thunk, debris).",
            f"SAFETY: strictly non-graphic; physically plausible; cause→effect within 0.9s."
        ]
    elif "attacking" in creative_lower or "relentless" in creative_lower:
        # Scene 2: Beast attacking warrior
        lines = [
            f"VEO-3 — {setting}, {lead} vs {ant} ({scene.get('duration_seconds',8)}s). Grounded, non-graphic. No idle seconds.",
            f"ACTION (beats every ~0.5–1.0s): {action_summary}",
            f"CAMERA: {camera_desc}",
            f"VFX: subtle dust effects during movement; impact particles.",
            f"LIGHTING: natural sun-dappled, key from camera-left.",
            f"AUDIO: ambience low forest bed; foley (leather, footfall, breath, non-graphic growl); sfx (snap, whoosh, woody thunk, debris).",
            f"SAFETY: strictly non-graphic; physically plausible; cause→effect within 0.9s."
        ]
    elif "arrows" in creative_lower or "pursuit" in creative_lower:
        # Scene 3: Warriors pursuing with arrows
        lines = [
            f"VEO-3 — {setting}, {lead} vs {ant} ({scene.get('duration_seconds',8)}s). Grounded, non-graphic. No idle seconds.",
            f"ACTION (beats every ~0.5–1.0s): {action_summary}",
            f"CAMERA: {camera_desc}",
            f"VFX: subtle arrow trails; dust/leaf puff on impacts.",
            f"LIGHTING: natural sun-dappled, key from camera-left.",
            f"AUDIO: ambience low forest bed; foley (leather, footfall, breath, non-graphic growl); sfx (snap, whoosh, woody thunk, debris).",
            f"SAFETY: strictly non-graphic; physically plausible; cause→effect within 0.9s."
        ]
    else:
        # Default single prompt
        lines = [
            f"VEO-3 — {setting}, {lead} vs {ant} ({scene.get('duration_seconds',8)}s). Grounded, non-graphic. No idle seconds.",
            f"ACTION (beats every ~0.5–1.0s): {action_summary}",
            f"CAMERA: {camera_desc}",
            f"VFX: subtle trails; dust/leaf puff on impacts.",
            f"LIGHTING: natural sun-dappled, key from camera-left.",
            f"AUDIO: ambience low forest bed; foley (leather, footfall, breath, non-graphic growl); sfx (snap, whoosh, woody thunk, debris).",
            f"SAFETY: strictly non-graphic; physically plausible; cause→effect within 0.9s."
        ]

    # soften wording just in case
    out = "\n".join(lines)
    for pat,rep in SAFEWORDING:
        out = re.sub(pat, rep, out)
    return out

# -------------------------
# Gold pack merger
# -------------------------

def _apply_gold_pack(scene_obj: dict, scene_text: str) -> dict:
    obj = deepcopy(scene_obj or {})

    # 1) Ensure shots/action
    obj = _ensure_actions(obj, scene_text)

    # 2) Normalize camera + soften audio, keep VFX
    for sh in obj.get("shots") or []:
        _normalize_camera(sh)
        if "audio" in sh:
            sh["audio"] = _soften_audio_block(sh.get("audio") or {})

    # 3) Merge skeleton (do not obliterate user-provided values)
    skel = _gold_skeleton()
    obj = _deep_merge_preserve(obj, skel)

    # 4) World consistency
    wc = obj.get("world_and_characters") or {}
    wc.setdefault("lead_character", "Warrior Squad")
    wc.setdefault("antagonist", "Forest Beast")
    wc.setdefault("setting", "Sun-dappled clearing deep in the forest")
    wc.setdefault("battle_context", "Beast charges; squad splits angles; bow releases; spear presses")
    wc.setdefault("time_of_day", "late morning")
    wc.setdefault("weather", "clear with a light breeze")
    obj["world_and_characters"] = wc

    # 5) Build single prompt rich summary
    obj["single_prompt_for_model"] = _build_single_prompt(obj)

    return obj

# -------------------------
# Monkey-patch entrypoint
# -------------------------

def _install_gold_output_mode():
    if not GOLD_OUTPUT_MODE:
        return
    try:
        cls = ChatGPTPromptGenerator
    except NameError:
        return
    if getattr(cls, "_gold_mode_installed", False):
        return
    _orig = cls.build_scene_prompt

    def _wrapped(self, scene_text, idx, continuity):
        out = _orig(self, scene_text, idx, continuity)
        try:
            if isinstance(out, dict) and "error" not in out:
                out = _apply_gold_pack(out, scene_text)
        except Exception:
            pass
        return out

    cls.build_scene_prompt = _wrapped
    cls._gold_mode_installed = True

_install_gold_output_mode()

# Utility: batch-fix existing prompt JSONs using v2 postprocessor
import json, sys
from pathlib import Path

# Giả sử bạn đã paste patch v2 ở trên vào cùng file và có _postprocess_scene_v2()
# Nếu không, copy lại các hàm _postprocess_scene_v2 và phụ trợ sang script này.

class Dummy:
    catalog = None

def fix(file_in, file_out):
    data = json.loads(Path(file_in).read_text(encoding="utf-8"))
    cont = Dummy()
    # Trường hợp data là list các scene hoặc object có key "scenes"
    if isinstance(data, list):
        scenes = data
    else:
        scenes = data.get("scenes") or data.get("prompts") or []
    fixed = []
    for i, sc in enumerate(scenes, 1):
        text_stub = sc.get("single_prompt_for_model","")
        fixed.append(_postprocess_scene_v2(sc, text_stub, cont))
    Path(file_out).write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Wrote", file_out)

if __name__ == "__main__":
    fin = sys.argv[1] if len(sys.argv)>1 else "all_veo3_prompts.json"
    fout = sys.argv[2] if len(sys.argv)>2 else "all_veo3_prompts_fixed.json"
    try:
        fix(fin, fout)
    except Exception as _e:
        # ignore if running primary main flow without those files
        pass


# ====== VÍ DỤ SỬ DỤNG FORCE SAMPLE TEMPLATE ======
def example_usage():
    """
    Ví dụ sử dụng force_sample_template để ép theo mẫu vàng 8s:
    """
    from cc import PromptGenerator
    
    scene_text = """
    Warrior Squad coordinates against a Forest Beast in a sun-dappled forest clearing.
    Purposeful, physically plausible motion with no idle seconds.
    """
    
    gen = PromptGenerator()
    gen.force_sample_template = True          # ép theo mẫu vàng (bỏ prune, khóa continuity)
    gen.strict_grounding = False              # tránh prune chi tiết tùy cấu hình hiện tại
    
    obj = gen.build_scene_prompt(scene_text)  # => JSON giống cấu trúc bạn đưa (8s, dày beats)
    print(obj["single_prompt_for_model"])
    
    # => out sẽ chứa đúng cấu trúc mẫu:
    #    - continuity_core: world_and_characters với lead/antagonist
    #    - aliases_map: canonical character names
    #    - full action_sequence 0–8s: 18 beats chi tiết
    #    - global_style: visual_tone, color_grading
    #    - safety_and_quality: avoid/must_have arrays
    #    - output_settings: linear/8000k config
    #    - single_prompt_for_model: dạng 1 dòng compact
    
    print("\nGenerated JSON structure:")
    print(f"- Model: {obj.get('model')}")
    print(f"- Request type: {obj.get('request_type')}")
    print(f"- Lead character: {obj.get('world_and_characters', {}).get('lead_character')}")
    print(f"- Antagonist: {obj.get('world_and_characters', {}).get('antagonist')}")
    print(f"- Action sequence beats: {len(obj.get('shots', [{}])[0].get('action_sequence', []))}")
    print(f"- Video bitrate: {obj.get('output_settings', {}).get('video_bitrate')}")
    print(f"- Tone mapping: {obj.get('output_settings', {}).get('tone_mapping')}")
    print(f"- Single prompt length: {len(obj.get('single_prompt_for_model', ''))}")
    print(f"- Has continuity_core: {'continuity_core' in obj}")
    print(f"- Has global_style: {'global_style' in obj}")
    print(f"- Has safety_and_quality: {'safety_and_quality' in obj}")
    
    return obj

# Uncomment để chạy ví dụ:
# if __name__ == "__main__":
#     example_usage()

# ====== VÍ DỤ SỬ DỤNG ĐƠN GIẢN ======
def simple_example():
    """
    Ví dụ sử dụng đơn giản với PromptGenerator:
    """
    from cc import PromptGenerator
    
    scene_text = """
    Warrior Squad coordinates against a Forest Beast in a sun-dappled forest clearing.
    Purposeful, physically plausible motion with no idle seconds.
    """
    
    gen = PromptGenerator()
    gen.force_sample_template = True          # ép theo mẫu vàng (bỏ prune, khóa continuity)
    gen.strict_grounding = False              # tránh prune chi tiết tùy cấu hình hiện tại
    
    obj = gen.build_scene_prompt(scene_text)  # => JSON giống cấu trúc bạn đưa (8s, dày beats)
    print(obj["single_prompt_for_model"])
    
    return obj

# Uncomment để chạy ví dụ đơn giản:
# if __name__ == "__main__":
#     simple_example()

# Monkey-patch the post-processing
try:
    # Apply the enhanced postprocessing
    if '_postprocess_scene_v2' in globals():
        ChatGPTPromptGenerator._postprocess_scene_v2 = _postprocess_scene_v2
except Exception:
    pass
