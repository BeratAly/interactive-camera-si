"""Command router (§23, §48): natural-language commands in TR + EN.

Fast-path: deterministic local commands are answered directly from REAL
vision/state data — instant, offline, and impossible to hallucinate (§77).
Anything else falls through to the LLM provider (if configured).

No shell execution here; system actions (phase 10) go through the future
whitelisted CommandManager with permission checks (§24/§43).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# word-boundary-insensitive normalization (accents / punctuation tolerant)
_TR = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "ğ": "g", "ü": "u",
                     "ö": "o", "ç": "c", "â": "a"})


def _norm(text: str) -> str:
    t = text.lower().translate(_TR)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _strip_trigger(original: str) -> str:
    """Remove the leading 'remember/hatirla ...' trigger word(s) from the
    user's ORIGINAL wording so stored memories keep accents/casing."""
    parts = original.strip().split(None, 2)
    if len(parts) >= 3 and parts[0].lower().translate(_TR) in (
            "remember", "hatirla", "hatırla"):
        rest = parts[2] if parts[1].lower() in ("that", "ki", "şu", "su") \
            else " ".join(parts[1:])
        return rest.lstrip(":- ").strip() or original.strip()
    if len(parts) == 2 and parts[0].lower().translate(_TR) in (
            "remember", "hatirla", "hatırla"):
        return parts[1].strip()
    return original.strip()


@dataclass(frozen=True)
class CommandResult:
    action: str          # e.g. "WHAT_DO_YOU_SEE"
    answer: Optional[str] = None   # set => answered locally, skip LLM
    clear_conversation: bool = False
    remember: Optional[str] = None  # explicit memory write (§17)


def route_command(user_text: str, vision_ctx: dict, events: list[str],
                  state_name: str, memories: list[str]) -> Optional[CommandResult]:
    """Return a CommandResult if handled locally, else None (-> LLM)."""
    t = _norm(user_text)
    people = int(vision_ctx.get("people", 0))
    camera_online = bool(vision_ctx.get("camera_online", False))
    faces = vision_ctx.get("faces", [])

    def seen_phrase() -> str:
        if not camera_online:
            return "Kamera aktif degil, simdi bir sey goremiyorum. "\
                   "/ Camera is not active, so I cannot see anything right now."
        if people == 0:
            return "Su an gorus alanimda kimse yok. / No person in view currently."
        conf = faces[0]["confidence"] if faces else 0.0
        ident = faces[0]["identity"] if faces else "UNKNOWN"
        who = "" if ident == "UNKNOWN" else f" ({ident})"
        return (f"Gorusumde {people} kisi var{who}, yuksek olasilikla "
                f"(%{conf*100:.0f}). / I can see {people} person(s){who}, "
                f"high confidence ({conf*100:.0f}%).")

    # --- what do you see? / ne görüyorsun? -------------------------------
    if _has(t, r"\bne goruyorsun\b", r"\bneyi goruyorsun\b",
            r"\bwhat (do you see|can you see)\b", r"\bgorebiliyor musun\b",
            r"\bwhat s going on\b"):
        return CommandResult("WHAT_DO_YOU_SEE", seen_phrase())

    # --- how many people? / kaç kişi var? ---------------------------------
    if _has(t, r"\bkac kisi\b", r"\bkaç kisi\b", r"\bhow many (people|persons)\b",
            r"\bkaç insan\b", r"\bkac insan\b"):
        if not camera_online:
            ans = ("Kamera kapali. / Camera is off, count unavailable.")
        elif people == 0:
            ans = "Sifir. Kamerada kimse gorunmuyor. / Zero persons in view."
        else:
            ans = f"{people} kisi. / {people} person(s)."
        return CommandResult("PEOPLE_COUNT", ans)

    # --- who am I? / ben kimim? (phase 6 fills identity) -------------------
    if _has(t, r"\bben kimim\b", r"\bwho am i\b"):
        known = [f for f in faces if f.get("identity", "UNKNOWN") != "UNKNOWN"]
        if known:
            f = known[0]
            ans = (f"Kimligin: {f['identity']}, guven %{f['confidence']*100:.1f}. "
                   f"/ You appear to be {f['identity']}.")
        elif people:
            ans = ("Yuzunuz algilaniyor ama tanimli bir profil eslesmedi: UNKNOWN. "\
                   "/ A face is detected but matches no stored profile.")
        else:
            ans = ("Kamerada yuz gorunamiyor. / I cannot see a face to identify.")
        return CommandResult("WHO_AM_I", ans)

    # --- FPS / sistem durumu ----------------------------------------------
    if _has(t, r"\bfps\b", r"\bkare hizi\b"):
        fps = vision_ctx.get("capture_fps")
        dfps = vision_ctx.get("detect_fps")
        ans = (f"FPS: yakalama {fps:.1f}, algilama {dfps:.1f}. "
               f"/ capture {fps:.1f}, detection {dfps:.1f}.") \
            if isinstance(fps, float) else "Vision motoru cevrimdisi. / Vision offline."
        return CommandResult("FPS_QUERY", ans)

    if _has(t, r"\bai durumunu?\b", r"\bsistem durumu\b", r"\bstatus\b",
            r"\bnasilisin\b", r"\bhow are you\b", r"\bcheck ?system\b"):
        ai = "ONLINE" if state_name != "OFFLINE" else "OFFLINE"
        cam = "ONLINE" if camera_online else "OFFLINE"
        ans = (f"Sistem: {state_name} | Kamera: {cam} | AI Core: {ai}. "
               f"/ System {state_name}, camera {cam}, AI core {ai}.")
        return CommandResult("STATUS", ans)

    # --- recent events / son olaylar ---------------------------------------
    if _has(t, r"\bson olay\b", r"\bson olaylar\b", r"\bne oldu\b",
            r"\brecent (events?|what happened)\b", r"\bevent log\b",
            r"\bhappened recently\b"):
        if events:
            ans = "Son olaylar: / Recent events: " + "; ".join(events[-5:])
        else:
            ans = "Kayitli olay yok. / No recorded events yet."
        return CommandResult("EVENT_HISTORY", ans)

    # --- conversation control ----------------------------------------------
    if _has(t, r"\bkonusmayi temizle\b", r"\btemizle\b", r"\bclear (the )?(chat|conversation)\b",
            r"\breset chat\b"):
        return CommandResult("CLEAR_CONVERSATION",
                             "Konusma temizlendi. / Conversation cleared.",
                             clear_conversation=True)

    # --- explicit memory (§17) ----------------------------------------------
    m = re.search(r"(?:hatirla|remember that|remember)\s*[:\-]?\s*(.+)$", t)
    if m and len(m.group(1)) > 2:
        fact = _strip_trigger(user_text)
        return CommandResult("REMEMBER",
                             "Kaydettim. / I'll remember that.",
                             remember=fact[:200])
    if _has(t, r"\bneleri hatirliyorsun\b", r"\bne hatirliyorsun\b",
            r"\bwhat do you (remember|know about me)\b"):
        ans = ("Hatirladiklarim: " + "; ".join(memories)) if memories else \
            ("Seni hakkinda kaydedilmis bir sey yok. / No stored memories about you.")
        return CommandResult("LIST_MEMORIES", ans)

    # --- greeting / help -----------------------------------------------------
    if _has(t, r"^(selam|merhaba|hey|hello|hi)\b", r"\bbana ne yapabilirsin\b",
            r"\byardım et\b", r"\byardim et\b", r"\bwhat can you do\b", r"\bhelp\b"):
        ans = ("Goruyorum, dinliyorum ve hatirliyorum. Kameraya bakabilirim: "
               "\"Ne goruyorsun?\", \"Kac kisi var?\", \"Ben kimim?\", \"FPS kac?\" "
               "/ I observe, reason and remember. Try: What do you see? / "
               "How many people? / Who am I? / FPS?")
        return CommandResult("HELP", ans)

    return None  # not a local command -> let the LLM handle it
