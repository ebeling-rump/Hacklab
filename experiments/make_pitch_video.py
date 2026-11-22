"""Build the ~3-minute pitch video from title cards, a chart and real simulation clips.

Narration uses the Windows built-in voice (System.Speech, offline); replace it with a team member's
voice for the submission if possible. Output: results/videos/pitch_video.mp4

    .venv\\Scripts\\python experiments/make_pitch_video.py
"""
import subprocess
import sys
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "scratch" / "pitch_video"
OUT = ROOT / "results" / "videos" / "pitch_video.mp4"
VID = ROOT / "results" / "videos"
W, H, FPS = 1280, 720, 24
DARK, LIGHT, ORANGE, BLUE, BODY = "#0F1B2D", "#F5F2EB", "#F08A24", "#3B8ED8", "#C3CEDD"


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def say(text, path):
    """Offline TTS with the Windows voice 'Microsoft Zira Desktop'."""
    txt = path.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "try { $s.SelectVoice('Microsoft Zira Desktop') } catch {}; $s.Rate = 1; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()")
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def card(path, eyebrow, title, lines=(), big=None):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(DARK)
    fig.text(0.07, 0.86, eyebrow, color=ORANGE, fontsize=15, fontweight="bold")
    fig.text(0.07, 0.80, title, color=LIGHT, fontsize=34, fontweight="bold", va="top", wrap=True)
    y = 0.55
    if big:
        for i, (num, lab) in enumerate(big):
            x = 0.07 + i * 0.30
            fig.text(x, 0.50, num, color=ORANGE, fontsize=52, fontweight="bold")
            fig.text(x, 0.43, lab, color=BODY, fontsize=15, va="top", wrap=True)
        y = 0.20
    for ln in lines:
        fig.text(0.07, y, ln, color=BODY, fontsize=20, va="top")
        y -= 0.085
    fig.savefig(path, facecolor=DARK)
    plt.close(fig)


def chart(path):
    ns = [1, 3, 5, 10, 20]
    rows = {"Physics + online system ID": ([31.5, 28.7, 26.5, 23.4, 21.9], BODY, "--"),
            "TabPFN v2 (local) + experience": ([31.5, 27.1, 23.4, 18.9, 16.0], BLUE, "-"),
            "TabPFN v3.5 (API) + experience": ([30.2, 23.1, 19.3, 14.5, 13.0], ORANGE, "-")}
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    for lab, (y, c, ls) in rows.items():
        ax.plot(range(5), y, ls, color=c, lw=4, marker="o", ms=9, label=lab)
    ax.set_xticks(range(5), [str(n) for n in ns])
    ax.set_xlabel("own pushes on the new (unseen, wheeled) object", color=BODY, fontsize=15)
    ax.set_ylabel("position prediction error [mm]", color=BODY, fontsize=15)
    ax.tick_params(colors=BODY, labelsize=14)
    for s in ax.spines.values():
        s.set_color("#3A4A63")
    ax.grid(alpha=0.15)
    ax.legend(facecolor=DARK, edgecolor="#3A4A63", labelcolor=LIGHT, fontsize=14)
    fig.suptitle("With experience from 40 other objects, TabPFN is most accurate from push one",
                 color=LIGHT, fontsize=20, fontweight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.885, "8 unseen objects, paired per object. v3.5 beats v2 and physics on 8/8 objects at every budget.",
             color=BODY, fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(path, facecolor=DARK)
    plt.close(fig)


SCALE = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0F1B2D,"
         f"fps={FPS},format=yuv420p")


def segment(i, visual, narration, clip_speed=None, clip_start=0.0):
    wav = WORK / f"n{i:02d}.wav"
    dur = say(narration, wav) + 0.9
    out = WORK / f"s{i:02d}.mp4"
    ff = ffmpeg()
    if str(visual).endswith(".png"):
        vin = ["-loop", "1", "-framerate", str(FPS), "-i", str(visual)]
        vf = f"[0:v]{SCALE}[v]"
    else:
        src = Path(visual)
        vin = ["-ss", str(clip_start), "-i", str(src)]
        speed = clip_speed or 1.0
        vf = f"[0:v]setpts=PTS/{speed},tpad=stop_mode=clone:stop_duration=60,{SCALE}[v]"
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", *vin, "-i", str(wav), "-filter_complex",
           f"{vf};[1:a]apad,aresample=44100[a]", "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-c:a", "aac", "-b:a", "128k", str(out)]
    subprocess.run(cmd, check=True)
    print(f"segment {i}: {dur:.1f}s", flush=True)
    return out, dur


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    c = lambda n: WORK / n  # noqa: E731
    card(c("title.png"), "TABPFN FOR SPATIAL AWARENESS  ·  BERLIN ROBOTICS & AI HACKATHON 2026",
         "Learning to Move the Unknown",
         ["A robot learns how an unfamiliar object moves from a handful of pushes,",
          "then steers it through a doorway. Simulation in MuJoCo, TabPFN at its core."])
    card(c("approach.png"), "OUR APPROACH", "Three ingredients, one planner",
         ["1  Physics prior: a classical pushing model that estimates friction and mass online",
          "2  TabPFN: learns what physics gets wrong from a few pushes, in context, with uncertainty",
          "3  Experience: pushes on 40 other objects, given to TabPFN as context",
          "Planner: imagine ~100 pushes, look two ahead around walls, push, observe, refit, repeat"])
    card(c("rigid.png"), "FINDING 1  ·  HONEST BASELINE", "For ordinary rigid boxes, physics is hard to beat",
         ["Physics + online system identification: best predictor on rigid boxes.",
          "TabPFN on top of physics: a close second. Every model-based planner solves",
          "all standard tasks; a reactive controller fails 10% and keeps hitting walls."])
    chart(c("chart.png"))
    card(c("api.png"), "FINDING 3  ·  NEWER FOUNDATION MODEL", "Same code, newer TabPFN, better robot",
         big=[("-38%", "error vs physics after 10 pushes\n(TabPFN v3.5 via Prior Labs API)"),
              ("8 / 8", "unseen objects where v3.5 beats\nour local v2 at every budget"),
              ("6 / 6", "A1 crates through the door\n(physics: 5 / 6)")])
    card(c("lessons.png"), "WHAT WE LEARNED", "Better predictions are not automatically better control",
         ["Planners exploit model errors; experience from other objects closed that gap.",
          "With 2-10 pushes, Gaussian processes were overconfident; TabPFN stayed calibrated.",
          "Limits: simulation only, 2-14 s per decision vs 0.15 s for physics,",
          "and the planner does not model wall contact yet."])
    card(c("vision.png"), "THE BIGGER PICTURE", "Physics as prior. Foundation model in context.\nExperience as context.",
         ["The same recipe fits opening unknown doors, carrying unknown loads, walking on",
          "unknown ground, and it improves for free as the foundation model improves.",
          "",
          "github.com/ebeling-rump/Hacklab  ·  branch claude/fewshot-pushing"])

    L = VID / "A1_doorway_Lcrate_tabpfn_api_vs_physics.mp4"
    C6 = VID / "C6H_s1_physics_sysid_vs_tabpfn_api_transfer_hybrid.mp4"
    plan = [
        (c("title.png"), "Learning to move the unknown. A robot learns how an unfamiliar object moves "
                         "from a handful of pushes, with TabPFN at its core.", None, 0),
        (L, "A warehouse robot finds a crate blocking a doorway. It doesn't know how heavy it is, how "
            "slippery, or where the weight sits. Push it in the middle, and it may swing sideways and jam "
            "in the door. Physics models need those parameters. Learned models need thousands of trials. "
            "A robot meeting a new object has neither.", 1.0, 0),
        (c("approach.png"), "We give the robot three things. A physics prior that estimates friction and "
                            "mass as it goes. TabPFN, which learns what physics gets wrong from just a few "
                            "pushes, in context, and says how sure it is. And experience: pushes on forty "
                            "other objects, given to TabPFN as context. A planner imagines about a hundred "
                            "pushes, looks two steps ahead around walls, pushes, observes, and learns.", None, 0),
        (c("rigid.png"), "We tested this on eight challenges in MuJoCo, with fixed seeds and held-out shapes. "
                         "First, the honest result: for ordinary rigid boxes, physics with online system "
                         "identification is hard to beat. TabPFN comes a close second.", None, 0),
        (c("chart.png"), "But when objects get messy, on caster wheels, with fast pushes that make them "
                         "slide, TabPFN with experience from other objects is the most accurate model from "
                         "the very first push, even on shapes it has never seen. A Gaussian process given "
                         "the same information is worse at every budget.", None, 0),
        (c("api.png"), "Then we switched to Prior Labs' newer TabPFN through the API, without changing "
                       "anything else. It beat our local version on all eight test objects, with about forty "
                       "percent less error than physics after ten pushes.", None, 0),
        (L, "Here is the Unitree A1, pushing an L-shaped crate it has never seen through a door narrower "
            "than the crate is long. The blue ghost is what it expects; the orange crate is what really "
            "happens. Top: TabPFN, done in eight pushes. Bottom: physics, fifteen. Across six crates, "
            "TabPFN got all six through, physics five. A demo, not a statistic.", 2.6, 0),
        (C6, "In a cluttered room with an L-shaped object on wheels, both succeed; TabPFN parks it twice "
             "as precisely, but touches walls more often.", 2.0, 0),
        (c("lessons.png"), "What we learned: better predictions are not automatically better control, because "
                           "planners exploit model errors. Experience fixes that. TabPFN's uncertainty stays "
                           "honest with very few samples. And it is slower than physics, and simulation only.",
         None, 0),
        (c("vision.png"), "Physics as prior, a foundation model that adapts in context, and experience as "
                          "context. Robots that learn physical skills from experience, not re-programming. "
                          "Thank you.", None, 0),
    ]
    segs, total = [], 0.0
    for i, (vis, text, speed, start) in enumerate(plan):
        out, d = segment(i, vis, text, speed, start)
        segs.append(out)
        total += d
    lst = WORK / "list.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
    subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(OUT)], check=True)
    print(f"total {total:.1f}s -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
