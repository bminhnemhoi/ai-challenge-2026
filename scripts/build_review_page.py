"""Build a one-file HTML review page for a whole round.

Human eyes beat every model at "is this the scene?".  The scoring rules make
that worth a lot: moving a hit from rank 6-20 to rank 1 is +0.4 on that query,
and the operator only has to recognise the shot, not find it.

The page shows the top candidates per query as thumbnails straight from the
Hugging Face CDN, so nothing has to be downloaded.  Clicking a thumbnail copies
a ready-made command that pins that video to the top of the query's CSV.

    python scripts/build_review_page.py --queries round_p1/queries --out round_p1/review.html
    (then just open the file in a browser)
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    detect_task,
    ranked_hits,
    read_en_override,
    read_query_text,
    split_events,
    split_qa,
)
from src.core.colours import colours_in_query  # noqa: E402
from src.core.submission import MAX_ROWS, allocate_trake_rows  # noqa: E402

CDN = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

#: confidence cut for the BTC detections — the same one their baseline uses
OBJ_CONF = 0.4
#: classes present in more than this share of keyframes carry no information
#: (measured: "Person" 39%, "Clothing" 46%, "Human face" 36%), so showing them
#: would just add noise to every caption
#: words too common in a Vietnamese query to be worth marking in a title
_STOP = {"khoảnh", "khắc", "đầu", "tiên", "video", "đoạn", "thấy", "trong", "một",
         "được", "những", "người", "này", "của", "với", "trên", "dưới", "sau",
         "trước", "cùng", "lúc", "hình", "ảnh", "clip", "cảnh", "tìm", "các",
         "sự", "kiện", "hoàn", "toàn", "bắt", "rồi", "cái", "chiếc"}

OBJ_SKIP = {"Clothing", "Human face", "Human body", "Human head", "Human arm",
            "Human leg", "Human hand", "Human nose", "Human hair", "Human eye",
            "Human mouth", "Human ear", "Footwear", "Sports equipment"}


def load_video_info(data_dir: Path, video_ids) -> dict:
    """YouTube id and title per video, from the organisers' media-info archive.

    Every one of the 873 videos carries a ``watch_url``, which is what makes
    reviewing possible at all: the operator can watch the actual moment instead
    of guessing from a 158-pixel thumbnail. Nothing has to be downloaded — the
    player is embedded and seeks straight to the timestamp of the keyframe.
    """
    import zipfile

    z = data_dir / "media-info-aic25-b1.zip"
    want = set(video_ids)
    out: dict = {}
    if not z.exists():
        return out
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"):
                continue
            vid = Path(n).stem
            if vid not in want:
                continue
            try:
                j = json.loads(zf.read(n))
            except Exception:  # noqa: BLE001
                continue
            url = str(j.get("watch_url") or "")
            m = re.search(r"(?:v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})", url)
            if m:
                out[vid] = {"y": m.group(1), "t": str(j.get("title") or "")[:90]}
    return out


def load_object_labels(data_dir: Path, keys):
    """(video_id, frame_stem) -> short 'Bicycle x3, Person x2' caption.

    Object counts are shown, never scored.  Measured on the ground truth, an
    automatic count bonus was inert — 57% of all keyframes contain at least one
    person, so "at least one person" is a constant rather than a filter — and
    the ground truth has only two counting queries, too few to validate any
    rule.  But half the round-1 queries name a count ("three cyclists", "four
    children"), and a human scanning thumbnails can use an exact count
    instantly.  So the signal goes to the operator, not to the ranker.
    """
    import zipfile

    keys = set(keys)
    out: dict = {}
    if not keys:
        return out

    def absorb(key, payload):
        try:
            j = json.loads(payload)
        except Exception:
            return
        c = Counter()
        for cls, sc in zip(
            j.get("detection_class_entities", []), j.get("detection_scores", [])
        ):
            if float(sc) > OBJ_CONF and cls not in OBJ_SKIP:
                c[cls] += 1
        out[key] = ", ".join(f"{cls}×{n_}" if n_ > 1 else cls for cls, n_ in c.most_common(4))

    # Either layout: download_data.py unpacks the archive, and older runs of it
    # deleted the zip afterwards, so a clone can legitimately have one and not
    # the other. Reading only the zip meant those clones lost every caption.
    unpacked = data_dir / "objects"
    if unpacked.is_dir():
        for video_id, stem in keys:
            f = unpacked / video_id / f"{stem}.json"
            if f.exists():
                absorb((video_id, stem), f.read_bytes())
        if out:
            return out

    z = data_dir / "objects-aic25-b1.zip"
    if not z.exists():
        return out
    # objects/<video_id>/<stem>.json without exception, so each frame is one
    # lookup into the central directory rather than a walk of 178,195 names
    with zipfile.ZipFile(z) as zf:
        for video_id, stem in keys:
            try:
                absorb((video_id, stem), zf.read(f"objects/{video_id}/{stem}.json"))
            except KeyError:
                continue
    return out

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>AIC 2026 — soát kết quả</title>
<style>
 :root {{ color-scheme: dark; }}
 * {{ box-sizing: border-box; }}
 body {{ font: 14px/1.5 system-ui, sans-serif; background:#12141a; color:#e6e8ee; margin:0; padding:0 24px 60px; }}
 h1 {{ font-size:19px; margin:0; }}
 .q {{ background:#1a1d26; border:1px solid #262b38; border-radius:10px; padding:16px; margin-bottom:18px; scroll-margin-top:130px; }}
 .q.done {{ border-color:#2f6b45; }}
 .qh {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px; }}
 .qid {{ font-weight:700; font-size:15px; }}
 .tag {{ font-size:11px; padding:2px 8px; border-radius:99px; background:#2a3040; color:#9fb4ff; }}
 .tag.warn {{ background:#4a2b1a; color:#ffc08a; }}
 .tag.ok {{ background:#1e4634; color:#7ee2a8; }}
 .qtext {{ color:#aeb6c6; margin-bottom:12px; white-space:pre-wrap; font-size:13px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(158px,1fr)); gap:8px; }}
 figure {{ margin:0; cursor:pointer; border:2px solid #23283400; border-radius:7px; overflow:hidden;
           background:#0d0f14; position:relative; transition:transform .08s; }}
 figure:hover {{ border-color:#5b7cff; transform:translateY(-2px); }}
 figure.pick {{ border-color:#4ade80; box-shadow:0 0 0 3px #4ade8033; }}
 figure.pick::after {{ content:"✓ ĐÃ CHỌN"; position:absolute; top:6px; right:6px; background:#4ade80;
                       color:#06210f; font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; }}
 img {{ width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#0d0f14; }}
 figcaption {{ font-size:11px; padding:4px 6px; color:#8b94a6; display:flex; justify-content:space-between; gap:4px; }}
 .objs {{ font-size:10px; padding:0 6px 5px; color:#6f7a8d; line-height:1.3; }}
 .ocr {{ font-size:10px; padding:0 6px 6px; color:#d8b4fe; line-height:1.35;
         max-height:34px; overflow:hidden; }}
 .ocr b {{ color:#f0abfc; }}
 .col {{ font-size:10px; padding:0 6px 5px; color:#8b94a6; }}
 .col b {{ color:#4ade80; }}
 .col i {{ color:#f87171; font-style:normal; }}
 .rank {{ color:#5b7cff; font-weight:700; }}
 .zoom {{ position:absolute; top:5px; left:5px; background:#0d0f14cc; border:1px solid #3a4250;
          color:#cfd6e4; font-size:11px; line-height:1; padding:4px 6px; border-radius:5px;
          cursor:zoom-in; opacity:0; transition:opacity .1s; }}
 figure:hover .zoom {{ opacity:1; }}
 /* one TRAKE candidate = one whole video chain, E1..EN side by side */
 .chain {{ border:2px solid #23283400; border-radius:8px; background:#0d0f14; padding:8px;
           margin-bottom:8px; cursor:pointer; transition:transform .08s; }}
 .chain:hover {{ border-color:#5b7cff; transform:translateY(-2px); }}
 .chain.pick {{ border-color:#4ade80; box-shadow:0 0 0 3px #4ade8033; }}
 .chain .ch {{ display:flex; align-items:center; gap:9px; margin-bottom:7px; font-size:12px; color:#9aa3b2; }}
 .chain .ch b {{ color:#e6e8ee; font-size:13px; }}
 .chain .grid {{ grid-template-columns:repeat(auto-fill,minmax(178px,1fr)); }}
 .ev {{ font-size:10px; color:#9fb4ff; padding:3px 6px 0; }}
 .vtitle {{ font-size:11.5px; color:#c8b273; max-width:52ch; overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }}
 .vtitle b {{ color:#ffd97a; }}
 #lb {{ position:fixed; inset:0; background:#000000ee; z-index:99; display:none;
        align-items:center; justify-content:center; flex-direction:column; cursor:zoom-out; }}
 #lb img {{ max-width:96vw; max-height:88vh; width:auto; aspect-ratio:auto; object-fit:contain; }}
 #lbc {{ color:#cfd6e4; font:13px ui-monospace, monospace; padding:10px; }}
 .ansbox {{ margin:10px 0 12px; padding:10px; background:#0d0f14; border:1px solid #3a4250;
            border-left:3px solid #ffc08a; border-radius:7px; }}
 .ansbox label {{ display:block; color:#ffc08a; font-size:12px; margin-bottom:6px; }}
 .ansbox input {{ width:min(560px,90%); }}
 .ansbox.ok {{ border-left-color:#4ade80; }}
 .ansbox.ok label {{ color:#7ee2a8; }}
 #bar {{ position:sticky; top:0; background:#12141aee; backdrop-filter:blur(8px); padding:12px 0;
         z-index:9; border-bottom:1px solid #262b38; margin-bottom:18px; }}
 .row {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
 button {{ background:#2a3040; color:#e6e8ee; border:1px solid #3a4250; border-radius:6px;
           padding:7px 13px; cursor:pointer; font:inherit; }}
 button:hover {{ background:#39415a; }}
 button.pri {{ background:#2f6b45; border-color:#3d8a59; font-weight:600; }}
 button.pri:hover {{ background:#3d8a59; }}
 #cmd {{ white-space:pre-wrap; word-break:break-all; color:#7ee2a8; background:#0d0f14;
         border:1px solid #262b38; border-radius:6px; padding:10px; margin-top:10px;
         font:12px/1.6 ui-monospace, monospace; max-height:230px; overflow:auto; display:none; }}
 #prog {{ color:#9aa3b2; font-size:13px; }}
 .jump a {{ color:#8b94a6; text-decoration:none; font-size:11px; padding:2px 5px; border-radius:4px; }}
 .jump a.d {{ color:#4ade80; }}
 .jump a.w {{ color:#ffc08a; }}
 .hint {{ color:#6f7a8d; font-size:12px; }}
 input[type=text] {{ background:#0d0f14; border:1px solid #3a4250; color:#e6e8ee; border-radius:5px;
                     padding:5px 8px; font:inherit; width:min(420px,60vw); }}
 figure, .chain {{ -webkit-user-drag: element; }}
 figure.drag, .chain.drag {{ opacity:.35; }}
 figure.over, .chain.over {{ outline:3px dashed #5b7cff; outline-offset:-3px; }}
 .grab {{ position:absolute; bottom:26px; right:5px; background:#0d0f14cc; border:1px solid #3a4250;
          color:#cfd6e4; font-size:11px; line-height:1; padding:4px 6px; border-radius:5px;
          cursor:grab; opacity:0; transition:opacity .1s; }}
 figure:hover .grab {{ opacity:1; }}
 .rank.n1 {{ color:#4ade80; }}
 #dl {{ color:#7ee2a8; font-size:13px; }}
 .play {{ position:absolute; top:5px; right:5px; background:#b91c1ccc; border:1px solid #ef4444;
          color:#fff; font-size:11px; line-height:1; padding:4px 7px; border-radius:5px;
          cursor:pointer; opacity:0; transition:opacity .1s; }}
 figure:hover .play {{ opacity:1; }}
 figure.pick .play {{ right:auto; left:5px; top:28px; }}
 /* the video inspector */
 #vi {{ position:fixed; inset:0; background:#080a0edd; z-index:120; display:none;
        align-items:center; justify-content:center; padding:18px; }}
 #vibox {{ background:#12141a; border:1px solid #2a3040; border-radius:12px; width:min(1180px,98vw);
           max-height:96vh; overflow:auto; padding:16px; }}
 #vihead {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }}
 #vihead b {{ font-size:15px; }}
 #vihead .sub {{ color:#8b94a6; font-size:12px; }}
 #viframe {{ position:relative; width:100%; aspect-ratio:16/9; background:#000; border-radius:8px;
             overflow:hidden; }}
 #viframe iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
 .virow {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:11px 0; }}
 .virow label {{ color:#9aa3b2; font-size:12px; }}
 .virow input[type=number] {{ width:120px; background:#0d0f14; border:1px solid #3a4250;
                              color:#e6e8ee; border-radius:5px; padding:5px 8px; font:inherit; }}
 #vinow {{ color:#7ee2a8; font:13px ui-monospace, monospace; }}
 #vistrip {{ display:flex; gap:6px; overflow-x:auto; padding:6px 0 10px; scroll-behavior:smooth; }}
 #vistrip figure {{ flex:0 0 132px; }}
 #vistrip figure.cur {{ border-color:#ffc08a; box-shadow:0 0 0 3px #ffc08a33; }}
 #vistrip img {{ aspect-ratio:16/9; }}
 #vistrip figcaption {{ font-size:10px; }}
 .evtabs {{ display:flex; gap:6px; flex-wrap:wrap; }}
 .evtabs button.on {{ background:#2f6b45; border-color:#3d8a59; }}
 .vihint {{ color:#6f7a8d; font-size:12px; line-height:1.6; }}
 #viq {{ color:#aeb6c6; font-size:12.5px; background:#0d0f14; border-left:3px solid #5b7cff;
         border-radius:6px; padding:8px 10px; margin:10px 0; white-space:pre-wrap;
         max-height:120px; overflow:auto; }}
 /* a query that needs the video: never make the operator hunt for the button */
 .q.scrutiny figure .play, .q.scrutiny .chain .play {{ opacity:.92; }}
 .q.scrutiny {{ border-left:3px solid #ffc08a; }}
 .q.scrutiny.done {{ border-left-color:#4ade80; }}
 body.onlyscrutiny .q:not(.scrutiny) {{ display:none; }}
 figure.manual {{ border-color:#a855f7; }}
 figure.manual .objs {{ color:#c4a3f0; }}
 .del {{ position:absolute; bottom:26px; left:5px; background:#7f1d1dcc; border:1px solid #b91c1c;
         color:#fff; font-size:11px; line-height:1; padding:3px 7px; border-radius:5px;
         cursor:pointer; opacity:0; transition:opacity .1s; }}
 figure:hover .del {{ opacity:1; }}
 #vidone {{ color:#7ee2a8; font-size:12.5px; }}
 button.watch {{ background:#7f1d1d; border-color:#b91c1c; }}
 button.watch:hover {{ background:#b91c1c; }}
</style>
<div id="bar">
  <div class="row">
    <h1>AIC 2026 — soát kết quả</h1>
    <span id="prog">0 / {nq} câu đã chọn</span>
    <button class="pri" onclick="exportZip()">⬇ Tải submission.zip</button>
    <button class="watch" onclick="toggleScrutiny()" id="scbtn">Chỉ hiện câu cần soi video</button>
    <button onclick="show()">Lệnh sửa (cách cũ)</button>
    <button onclick="copyCmd()">Copy lệnh</button>
    <button onclick="clearAll()">Đặt lại tất cả</button>
    <span id="dl"></span>
  </div>
  <div class="row jump" id="jump" style="margin-top:8px"></div>
  <div class="hint" style="margin-top:6px">
    <b>Kéo thả</b> khung hình để đổi thứ hạng — vị trí #1 là dòng đầu của bài nộp.
    Bấm một khung hình để đưa thẳng nó lên #1.
    Phím <b>1–9</b> đưa khung hình thứ n lên đầu, <b>j/k</b> chuyển câu.
    Rê chuột rồi bấm <b>🔍</b> (hoặc <b>Shift</b>+bấm) để phóng to đọc chữ trên hình.
    Câu <span class="tag warn">cần xem kỹ</span> là câu hệ thống không chắc; câu Q&amp;A phải điền đáp án.
    Xong hết thì bấm <b>Tải submission.zip</b> — file nộp được tạo ngay trong trình duyệt.
    Mọi thay đổi được lưu lại, đóng trang mở lại vẫn còn.
  </div>
  <pre id="cmd"></pre>
</div>
<div id="lb" onclick="this.style.display='none'"><img alt=""><div id="lbc"></div></div>
<div id="vi">
  <div id="vibox">
    <div id="vihead">
      <b id="vitit"></b><span class="sub" id="visub"></span>
      <span style="flex:1"></span>
      <button onclick="closeInspector()">Đóng (Esc)</button>
    </div>
    <div id="viq"></div>
    <div id="viframe"></div>
    <div class="virow" id="vievrow" style="display:none">
      <label>Sự kiện:</label><span class="evtabs" id="vievtabs"></span>
    </div>
    <div class="virow">
      <span id="vinow"></span><span id="vilive" style="color:#9aa3b2;font:12px ui-monospace,monospace"></span>
      <button onclick="stepFrame(-1)" title="Lùi 1 frame (←)">◀ 1 frame</button>
      <button onclick="togglePlay()" id="viplay" title="Phát / dừng (Space)">⏸ Dừng</button>
      <button onclick="stepFrame(1)" title="Tiến 1 frame (→)">1 frame ▶</button>
      <button onclick="captureNow()" class="watch" title="Lấy đúng thời điểm đang dừng (C)">
        ◉ Lấy thời điểm đang xem (C)</button>
    </div>
    <div class="virow">
      <label>hoặc nhập giây:</label>
      <input type="number" id="visec" step="0.04" min="0">
      <button onclick="seekToSeconds()">Nhảy tới</button>
      <label style="margin-left:14px">Đặt vào vị trí:</label>
      <input type="number" id="vipos" min="1" max="100" value="1" style="width:80px">
      <button class="pri" onclick="useCurrentFrame()" id="vibtn">Chốt frame</button>
      <span id="vidone"></span>
    </div>
    <div class="vihint">
      <b>Cách nhanh nhất:</b> để video chạy, bấm <b>Space</b> dừng đúng khoảnh khắc, tinh chỉnh
      bằng <b>←</b> <b>→</b> (mỗi lần 1 frame; giữ <b>Shift</b> để nhảy cả keyframe), rồi bấm
      <b>C</b> để lấy đúng thời điểm đó. Không phải đọc số giây, không phải gõ tay.
      Khi đã đúng, chọn <b>vị trí</b> muốn đặt
      (không nhất thiết #1 — chưa chắc lắm thì để #2 hay #3) và bấm <b>Chốt frame</b>.
      Một thẻ tím sẽ xuất hiện trong danh sách của câu đó, kéo thả và xoá được như mọi thẻ khác.
      Frame <b>không cần</b> nằm trong danh sách gợi ý: cửa sổ đáp án rộng khoảng 10 frame còn
      keyframe cách nhau ~55, nên chốt tay thường chính xác hơn.
      Với TRAKE, chọn tab <b>E1…EN</b> trước rồi chốt từng sự kiện. <b>Enter</b> = Chốt frame.
    </div>
    <div id="vistrip"></div>
  </div>
</div>
{body}
<script>
{alloc_js}
</script>
<script>
const QDIR = "{qdir}", OUTDIR = "{outdir}";
const KEY = "aic_order_{tag}";
const DATA = {data_json};
const PLAN = {plan_json};
const VID = {vid_json};
const warn = new Set({warnlist});
const qids = [...document.querySelectorAll('.q')].map(q => q.dataset.qid);
let cur = 0;

// state[qid] = {{order: [id...], extra: [{{v,f}}...], answer, touched, frames}}
//
// An id is "cN" for the Nth retrieved candidate and "xN" for the Nth frame the
// operator marked while watching the video. Both live in the SAME order array,
// so a hand-marked frame is a first-class candidate that can sit at any rank —
// it used to be a separate `custom` field pinned to rank 1, which meant the
// button appeared to do nothing (no card ever appeared) and the rank was not
// the operator's to choose.
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ state = {{}}; }}
for (const id of qids) {{
  const n = (DATA[id].shown || 0);
  const st = state[id] || {{}};
  const extra = Array.isArray(st.extra)
    ? st.extra.filter(e => e && typeof e.v === 'string' && Number.isInteger(e.f)) : [];
  const valid = new Set([...Array(n).keys()].map(i => 'c' + i)
                        .concat(extra.map((_, i) => 'x' + i)));
  // a stored order from an older build can be stale; rebuild rather than
  // silently drop or duplicate a candidate
  let ord = (Array.isArray(st.order) ? st.order : []).filter(k => valid.has(k));
  ord = [...new Set(ord)];
  for (const k of valid) if (!ord.includes(k)) ord.push(k);
  state[id] = {{
    order: ord,
    extra: extra,
    answer: typeof st.answer === 'string' ? st.answer : '',
    touched: !!st.touched,
    frames: Array.isArray(st.frames) ? st.frames : null,   // hand-marked TRAKE events
  }};
}}
function save() {{ localStorage.setItem(KEY, JSON.stringify(state)); }}

function candOf(qid, key) {{
  const st = state[qid], d = DATA[qid];
  if (key[0] === 'x') {{
    const e = st.extra[+key.slice(1)];
    return e ? [e.v, e.f, (VID[e.v] || {{}}).l] : null;
  }}
  return d.cands[+key.slice(1)] || null;
}}
function orderedCands(qid) {{
  const st = state[qid], d = DATA[qid];
  const seen = new Set();
  const out = [];
  for (const k of st.order) {{
    const c = candOf(qid, k);
    if (!c) continue;
    const sig = c[0] + ':' + c[1];
    if (seen.has(sig)) continue;      // a marked frame may duplicate a candidate
    seen.add(sig);
    out.push(c);
  }}
  for (const c of d.cands.slice(d.shown)) {{
    const sig = c[0] + ':' + c[1];
    if (!seen.has(sig)) {{ seen.add(sig); out.push(c); }}
  }}
  return out;
}}

function itemsOf(qid) {{
  const sel = DATA[qid].task === 'trake' ? '.chain' : 'figure[data-q]';
  return [...document.getElementById(qid).querySelectorAll(sel)];
}}

// a card for a frame the operator marked while watching: the nearest keyframe's
// picture, labelled with the real frame number and the distance, so nobody
// mistakes the illustration for the frame that will actually be submitted
function extraCard(qid, key, e) {{
  const info = VID[e.v] || {{f: 25, k: []}};
  let near = 0, best = Infinity;
  info.k.forEach((f, i) => {{ const d = Math.abs(f - e.f); if (d < best) {{ best = d; near = i; }} }});
  const img = info.k.length
    ? CDN + '/' + e.v + '/' + String(near + 1).padStart(3, '0') + '.jpg' : '';
  const el = document.createElement('figure');
  el.dataset.q = qid; el.dataset.id = key; el.dataset.v = e.v; el.dataset.ff = e.f;
  el.className = 'manual';
  el.innerHTML =
    '<img loading="lazy" src="' + img + '" alt="">' +
    '<button class="zoom" title="Phóng to">🔍</button>' +
    '<button class="play" title="Xem lại">▶ xem</button>' +
    '<button class="del" title="Bỏ frame này">✕</button>' +
    '<span class="grab">⠿</span>' +
    '<figcaption><span class="rank"></span><span>' + e.v + '</span>' +
    '<span>' + e.f + '</span></figcaption>' +
    '<div class="objs">chốt tay · ' + (e.f / info.f).toFixed(2) + 's' +
    (best ? ' · ảnh là keyframe cách ' + best + ' frame' : '') + '</div>';
  wire(el);
  return el;
}}

function reorderDom(qid) {{
  const st = state[qid];
  const items = itemsOf(qid);
  const parent = items[0] ? items[0].parentNode
                          : document.getElementById(qid).querySelector('.grid');
  if (!parent) return;
  const byId = new Map(items.map(el => [el.dataset.id, el]));
  st.order.forEach((key, pos) => {{
    let el = byId.get(key);
    if (!el && key[0] === 'x') {{
      el = extraCard(qid, key, st.extra[+key.slice(1)]);
      byId.set(key, el);
    }}
    if (!el) return;
    parent.appendChild(el);
    const r = el.querySelector('.rank');
    if (r) {{ r.textContent = '#' + (pos + 1); r.classList.toggle('n1', pos === 0); }}
    el.classList.toggle('pick', pos === 0 && st.touched);
  }});
  // drop cards for extras that were deleted
  byId.forEach((el, key) => {{ if (!st.order.includes(key)) el.remove(); }});
}}

function paint() {{
  for (const id of qids) reorderDom(id);
  document.querySelectorAll('.q').forEach(q => {{
    const id = q.dataset.qid, st = state[id];
    const box = q.querySelector('.ansbox');
    const answered = !box || !!st.answer.trim();
    const done = st.touched && answered;
    q.classList.toggle('done', done);
    if (box) box.classList.toggle('ok', answered);
    const b = q.querySelector('.badge');
    if (b) {{
      const top = candOf(id, st.order[0]);
      let what = top ? (st.order[0][0] === 'x' ? top[0] + ':' + top[1] : top[0]) : '';
      if (st.order[0][0] === 'x') what += ' (chốt tay)';
      if (st.frames) what += ' (frame chốt tay)';
      b.textContent = !st.touched ? '' : (answered ? '✓ ' + what : '⚠ thiếu đáp án');
      b.className = answered ? 'tag ok badge' : 'tag warn badge';
    }}
  }});
  document.getElementById('prog').textContent =
    document.querySelectorAll('.q.done').length + ' / {nq} câu đã chọn';
  document.getElementById('jump').innerHTML = qids.map((id, i) =>
    `<a href="#${{id}}" class="${{state[id].touched ? 'd' : (warn.has(id) ? 'w' : '')}}"
        onclick="cur=${{i}}">${{id.replace('query-p1-','')}}</a>`).join('');
}}

function moveTo(qid, key, pos) {{
  const ord = state[qid].order;
  const from = ord.indexOf(key);
  if (from < 0) return;
  ord.splice(from, 1);
  ord.splice(Math.max(0, Math.min(ord.length, pos)), 0, key);
  state[qid].touched = true;
  save(); paint();
}}

function removeExtra(qid, key) {{
  const st = state[qid];
  const i = +key.slice(1);
  st.order = st.order.filter(k => k !== key);
  // renumber the ids after the hole so they keep pointing at the right entry
  st.extra.splice(i, 1);
  st.order = st.order.map(k => (k[0] === 'x' && +k.slice(1) > i) ? 'x' + (+k.slice(1) - 1) : k);
  save(); paint();
}}

// ------------------------------------------------------------ interaction

let dragging = null;

function wire(el) {{
  const qid = el.dataset.q;
  el.draggable = true;
  el.addEventListener('dragstart', ev => {{
    dragging = el;
    el.classList.add('drag');
    ev.dataTransfer.effectAllowed = 'move';
    ev.dataTransfer.setData('text/plain', el.dataset.id);
  }});
  el.addEventListener('dragend', () => {{
    dragging = null;
    document.querySelectorAll('.drag,.over').forEach(x => x.classList.remove('drag', 'over'));
  }});
  el.addEventListener('dragover', ev => {{
    if (!dragging || dragging.dataset.q !== qid || dragging === el) return;
    ev.preventDefault();                       // without this, drop never fires
    ev.dataTransfer.dropEffect = 'move';
    el.classList.add('over');
  }});
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', ev => {{
    el.classList.remove('over');
    if (!dragging || dragging.dataset.q !== qid) return;
    ev.preventDefault();
    ev.stopPropagation();
    moveTo(qid, dragging.dataset.id, state[qid].order.indexOf(el.dataset.id));
  }});
  el.addEventListener('click', ev => {{
    if (ev.target.classList.contains('del')) {{
      removeExtra(qid, el.dataset.id);
      return;
    }}
    if (ev.target.classList.contains('play')) {{
      const fig = ev.target.closest('figure');
      const evIdx = DATA[qid].task === 'trake'
        ? [...el.querySelectorAll('figure')].indexOf(fig) : null;
      openInspector(qid, fig.dataset.v, +fig.dataset.ff, evIdx);
      return;
    }}
    if (ev.shiftKey || ev.target.classList.contains('zoom')) {{ lightbox(ev.target); return; }}
    if (ev.target.classList.contains('grab')) return;
    moveTo(qid, el.dataset.id, 0);             // one click promotes to rank 1
  }});
}}
document.querySelectorAll('figure[data-q], .chain').forEach(wire);

document.querySelectorAll('.ansbox input').forEach(inp => {{
  const q = inp.dataset.q;
  inp.value = state[q].answer || '';
  inp.oninput = () => {{
    // `,` and `"` would be RFC-quoted into the CSV and then split apart by a
    // comma-splitting grader, and `;` separates picks in the fallback command,
    // so all three are replaced here where the operator can see it happen
    const clean = inp.value.replace(/[;,"\\n\\r]/g, ' ').replace(/\\s+/g, ' ').trim();
    if (clean !== inp.value.trim()) inp.value = clean;
    state[q].answer = clean;
    save(); paint();
  }};
}});

// ------------------------------------------------------- the video inspector
//
// Every one of the 873 videos has a YouTube watch_url in the organisers' own
// media-info, and metadata.json carries pts_time and fps per keyframe. Together
// that means the operator can watch the actual moment rather than judge a
// 158-pixel thumbnail — which is the only way to settle a TRAKE chain, read an
// answer off the screen for a Q&A, or break a tie the retriever is unsure about.

const CDN = "{cdn}";
let vi = null;   // {{qid, video, frame, ev}}

// ---- the YouTube player -----------------------------------------------
//
// A plain <iframe src="...?start=N"> can only jump to a whole second and can
// never be asked where it is. The IFrame API gives seekTo() with fractional
// seconds and getCurrentTime(), which is what turns "watch the video" into
// "capture the exact frame": the operator pauses on the instant and presses one
// key, instead of squinting at the progress bar and typing a number.
//
// This is how the two AIC 2025 systems that scored Outstanding on TRAKE work
// (MERVIN computes "frame positions on-the-fly from playback time and FPS";
// U-CESE binds a key to append the current timestamp to the answer field).
let ytPlayer = null, ytReady = false, ytTimer = null, ytPending = null;

const _ytTag = document.createElement('script');
_ytTag.src = 'https://www.youtube.com/iframe_api';
document.head.appendChild(_ytTag);
window.onYouTubeIframeAPIReady = () => {{
  ytReady = true;
  if (ytPending) {{ const p = ytPending; ytPending = null; mountPlayer(p[0], p[1]); }}
}};

function mountPlayer(videoId, startSec) {{
  const host = document.getElementById('viframe');
  if (!ytReady) {{
    ytPending = [videoId, startSec];
    host.innerHTML = '<div style="color:#8b94a6;padding:20px">đang tải trình phát…</div>';
    return;
  }}
  if (ytPlayer && ytPlayer.loadVideoById) {{
    ytPlayer.loadVideoById({{videoId: videoId, startSeconds: startSec}});
    return;
  }}
  host.innerHTML = '<div id="ytmount"></div>';
  ytPlayer = new YT.Player('ytmount', {{
    videoId: videoId,
    playerVars: {{autoplay: 1, rel: 0, modestbranding: 1, playsinline: 1, start: Math.floor(startSec)}},
    events: {{onReady: e => e.target.seekTo(startSec, true)}},
  }});
}}

// a live read-out of where the playhead is, in frames — the number that gets submitted
function startTicker() {{
  clearInterval(ytTimer);
  ytTimer = setInterval(() => {{
    if (!vi || !ytPlayer || !ytPlayer.getCurrentTime) return;
    const fps = VID[vi.video].f;
    const t = ytPlayer.getCurrentTime();
    const el = document.getElementById('vilive');
    if (el) el.textContent = 'đang ở: frame ' + Math.round(t * fps) + '  (' + t.toFixed(2) + 's)';
  }}, 200);
}}

function playerTime() {{
  return (ytPlayer && ytPlayer.getCurrentTime) ? ytPlayer.getCurrentTime() : null;
}}
function seekAbs(sec) {{
  if (ytPlayer && ytPlayer.seekTo) ytPlayer.seekTo(Math.max(0, sec), true);
}}
function togglePlay() {{
  if (!ytPlayer || !ytPlayer.getPlayerState) return;
  const playing = ytPlayer.getPlayerState() === 1;
  if (playing) ytPlayer.pauseVideo(); else ytPlayer.playVideo();
  document.getElementById('viplay').textContent = playing ? '▶ Phát' : '⏸ Dừng';
}}

// One SOURCE frame, computed from this video's own fps. YouTube's ","/"."
// shortcuts are undocumented inside an embed and step the rendered stream, not
// the frame numbering the grader uses, so they are not relied on.
function stepFrame(n) {{
  const t = playerTime();
  if (t === null) return;
  if (ytPlayer.pauseVideo) ytPlayer.pauseVideo();
  const fps = VID[vi.video].f;
  const target = Math.max(0, Math.round(t * fps) + n) / fps;
  seekAbs(target);
  vi.frame = Math.round(target * fps);
  renderInspector(false);
}}

// the whole point: the operator paused on the instant, so take it verbatim
function captureNow() {{
  const t = playerTime();
  if (t === null) {{ alert('Trình phát chưa sẵn sàng.'); return; }}
  if (ytPlayer.pauseVideo) ytPlayer.pauseVideo();
  vi.frame = Math.round(t * VID[vi.video].f);
  renderInspector(false);
  document.getElementById('vidone').textContent =
    '↑ đã lấy frame ' + vi.frame + ' — chọn vị trí rồi bấm Chốt frame';
}}

function kfUrl(v, i) {{
  return CDN + "/" + v + "/" + String(i + 1).padStart(3, "0") + ".jpg";
}}
function nearestKf(v, frame) {{
  const k = VID[v].k;
  let lo = 0, hi = k.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (k[mid] < frame) lo = mid + 1; else hi = mid;
  }}
  if (lo > 0 && Math.abs(k[lo - 1] - frame) <= Math.abs(k[lo] - frame)) lo -= 1;
  return lo;
}}

function openInspector(qid, video, frame, ev) {{
  if (!VID[video]) {{ alert('Không có link video cho ' + video); return; }}
  vi = {{qid, video, frame: Math.trunc(frame), ev: ev === undefined ? null : ev}};
  document.getElementById('vi').style.display = 'flex';
  renderInspector(true);
  startTicker();
}}
function closeInspector() {{
  clearInterval(ytTimer);
  if (ytPlayer && ytPlayer.stopVideo) ytPlayer.stopVideo();
  document.getElementById('viframe').innerHTML = '';
  ytPlayer = null;                       // a fresh mount next time, so no stale state
  document.getElementById('vi').style.display = 'none';
  vi = null;
}}

function renderInspector(reload) {{
  if (!vi) return;
  const info = VID[vi.video], d = DATA[vi.qid];
  const sec = vi.frame / info.f;
  document.getElementById('vitit').textContent = vi.video;
  document.getElementById('visub').textContent =
    (info.t || '') + '  ·  ' + info.f + ' fps  ·  ' + info.k.length + ' keyframe';
  document.getElementById('vinow').textContent =
    'frame ' + vi.frame + '  =  ' + sec.toFixed(2) + ' giây';
  document.getElementById('visec').value = sec.toFixed(2);
  // the question stays on screen: for a Q&A the answer is usually spoken or
  // written in the shot, and nobody should be scrolling back to re-read it
  document.getElementById('viq').textContent = d.text || '';

  if (reload) mountPlayer(info.y, Math.max(0, sec - 2));

  // event tabs, for a TRAKE chain
  const evrow = document.getElementById('vievrow');
  if (d.task === 'trake') {{
    const frames = trakeFrames(vi.qid);
    evrow.style.display = 'flex';
    document.getElementById('vievtabs').innerHTML = frames.map((f, j) =>
      `<button class="${{j === vi.ev ? 'on' : ''}}" onclick="pickEvent(${{j}})">E${{j + 1}} · ${{f}}</button>`
    ).join('');
    document.getElementById('vibtn').textContent =
      'Đặt frame này cho E' + ((vi.ev || 0) + 1);
  }} else {{
    evrow.style.display = 'none';
    document.getElementById('vibtn').textContent = 'Đặt frame này lên #1';
  }}

  // keyframes around the anchor, so the exact instant can be hunted visually
  const c = nearestKf(vi.video, vi.frame);
  const from = Math.max(0, c - 9), to = Math.min(info.k.length, c + 10);
  const strip = [];
  for (let i = from; i < to; i++) {{
    strip.push(
      `<figure class="${{i === c ? 'cur' : ''}}" onclick="gotoKf(${{i}})">` +
      `<img loading="lazy" src="${{kfUrl(vi.video, i)}}" alt="">` +
      `<figcaption><span>${{info.k[i]}}</span>` +
      `<span>${{(info.k[i] / info.f).toFixed(1)}}s</span></figcaption></figure>`);
  }}
  const el = document.getElementById('vistrip');
  el.innerHTML = strip.join('');
  const cur = el.querySelector('figure.cur');
  if (cur) cur.scrollIntoView({{block: 'nearest', inline: 'center'}});
}}

function gotoKf(i) {{
  vi.frame = VID[vi.video].k[i];
  seekAbs(vi.frame / VID[vi.video].f);
  renderInspector(false);
}}
function pickEvent(j) {{
  vi.ev = j;
  vi.frame = trakeFrames(vi.qid)[j];
  seekAbs(Math.max(0, vi.frame / VID[vi.video].f - 2));
  renderInspector(false);
}}
function seekToSeconds() {{
  const s = parseFloat(document.getElementById('visec').value);
  if (!isFinite(s) || s < 0) return;
  vi.frame = Math.round(s * VID[vi.video].f);
  seekAbs(s);
  renderInspector(false);
}}

function trakeFrames(qid) {{
  const st = state[qid], d = DATA[qid];
  if (st.frames && st.frames.length) return st.frames;
  const rows = d.chainRows[+st.order[0].slice(1)];
  return rows && rows.length ? rows[0].slice(1) : [];
}}

function useCurrentFrame() {{
  const d = DATA[vi.qid], st = state[vi.qid];
  const note = document.getElementById('vidone');

  if (d.task === 'trake') {{
    const chainVideo = candOf(vi.qid, st.order[0])[0];
    if (vi.video !== chainVideo) {{
      alert('Frame này thuộc ' + vi.video + ' nhưng chuỗi đang chọn là ' + chainVideo +
            '.\\nMọi sự kiện phải nằm trong CÙNG một video — sai video là 0 điểm.');
      return;
    }}
    const frames = trakeFrames(vi.qid).slice();
    frames[vi.ev || 0] = vi.frame;
    st.frames = frames;
    st.touched = true;
    save(); paint(); renderInspector(false);
    note.textContent = '✓ E' + ((vi.ev || 0) + 1) + ' = frame ' + vi.frame;
    return;
  }}

  // A marked frame becomes a real candidate card at the rank the operator asked
  // for. It used to be stored in a hidden field pinned to rank 1: the button
  // looked like it had done nothing, and there was no way to say "put this at
  // #3" — which is what you want when you are fairly sure but not certain.
  const pos = Math.max(1, Math.min(100, parseInt(document.getElementById('vipos').value, 10) || 1));
  const dup = st.extra.findIndex(e => e.v === vi.video && e.f === vi.frame);
  let key;
  if (dup >= 0) {{
    key = 'x' + dup;
  }} else {{
    st.extra.push({{v: vi.video, f: vi.frame}});
    key = 'x' + (st.extra.length - 1);
    st.order.push(key);
  }}
  st.touched = true;
  moveTo(vi.qid, key, pos - 1);     // moveTo saves and repaints
  note.textContent = '✓ frame ' + vi.frame + ' đặt ở vị trí #' + pos +
                     (dup >= 0 ? ' (đã có, chỉ đổi vị trí)' : '');
  const card = document.querySelector('figure[data-q="' + vi.qid + '"][data-id="' + key + '"]');
  if (card) card.scrollIntoView({{block: 'center', behavior: 'smooth'}});
}}

// burned-in captions ("NẤM RƠM CẮT ĐÔI") are unreadable at thumbnail size but
// often name the exact thing the query asks for, so full size is one click away
function lightbox(el) {{
  const fig = el.closest('figure');
  if (!fig) return;
  const lb = document.getElementById('lb');
  lb.querySelector('img').src = fig.querySelector('img').src;
  document.getElementById('lbc').textContent =
    fig.dataset.v + '  ·  frame ' + (fig.dataset.ff || fig.dataset.f);
  lb.style.display = 'flex';
}}

// ------------------------------------------------------------- the upload

function rowsFor(qid) {{
  const d = DATA[qid], st = state[qid];
  if (d.task === 'trake') {{
    // one video takes every row, so the chain sitting at #1 decides everything
    const video = candOf(qid, st.order[0])[0];
    if (st.frames && st.frames.length) {{
      // the operator watched the video and marked at least one event by hand
      return allocateTrakeRows(video, st.frames, PLAN.budget, PLAN.step,
                               (VID[video] || {{}}).l);
    }}
    return d.chainRows[+st.order[0].slice(1)] || [];
  }}
  const rows = allocateHybridRows(
    orderedCands(qid).map(c => ({{v: c[0], f: c[1], last: c[2]}})), PLAN.nFlat, PLAN);
  return d.task === 'qa' ? rows.map(r => [r[0], r[1], st.answer]) : rows;
}}

function exportZip() {{
  const missing = qids.filter(id => DATA[id].task === 'qa' && !state[id].answer.trim());
  if (missing.length && !confirm(
      missing.length + ' câu Q&A chưa có đáp án (' + missing.join(', ') +
      ').\\nNhững câu đó sẽ được 0 điểm. Vẫn tải file?')) return;

  const files = qids.map(id => ({{name: 'submission/' + id + '.csv', text: rowsToCsv(rowsFor(id))}}));
  const bytes = buildZip(files);
  const url = URL.createObjectURL(new Blob([bytes], {{type: 'application/zip'}}));
  const a = document.createElement('a');
  a.href = url; a.download = 'submission.zip';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);

  const rows = files.reduce((n, f) => n + f.text.trim().split('\\n').length, 0);
  document.getElementById('dl').textContent =
    `đã tạo ${{files.length}} CSV, ${{rows}} dòng` +
    (missing.length ? ` — ${{missing.length}} câu Q&A còn trống!` : ' — kiểm tra rồi nộp');
  const el = document.getElementById('cmd');
  el.style.display = 'block';
  el.textContent =
    '# Kiểm tra định dạng trước khi nộp (một lệnh, vài giây):\\n' +
    'python scripts/verify_zip.py ~/Downloads/submission.zip --queries ' + QDIR;
}}

function show() {{
  const parts = qids.filter(id => state[id].touched).map(id => {{
    const d = DATA[id], st = state[id];
    if (d.task === 'trake') {{
      const fr = trakeFrames(id);
      const v = candOf(id, st.order[0])[0];
      return fr.length ? `${{id}}=${{v}}:${{fr.join('|')}}` : null;
    }}
    const c = candOf(id, st.order[0]);
    return `${{id}}=${{c[0]}}:${{c[1]}}` + (st.answer.trim() ? `:${{st.answer.trim()}}` : '');
  }}).filter(Boolean);
  const el = document.getElementById('cmd');
  el.style.display = 'block';
  if (!parts.length) {{ el.textContent = 'Chưa đổi thứ hạng câu nào.'; return; }}
  el.textContent =
    '# Cách cũ: chỉ áp dụng lựa chọn #1 của mỗi câu, không giữ thứ tự bạn kéo.\\n' +
    '# Muốn giữ nguyên thứ tự thì dùng nút "Tải submission.zip".\\n' +
    `python scripts/apply_picks.py --queries ${{QDIR}} --out ${{OUTDIR}} \\\\\n  --picks "${{parts.join(';')}}"`;
}}
function copyCmd() {{
  show();
  const txt = document.getElementById('cmd').textContent
    .split('\\n').filter(l => !l.startsWith('#')).join('\\n');
  navigator.clipboard.writeText(txt).then(() => alert('Đã copy. Dán vào terminal và chạy.'));
}}
// TRAKE, Q&A and the queries the retriever is unsure about are the ones that
// cannot be settled from a thumbnail — a chain has to be watched in order, a
// Q&A answer has to be read or heard, and a tie has to be broken.
function toggleScrutiny() {{
  const on = document.body.classList.toggle('onlyscrutiny');
  document.getElementById('scbtn').textContent =
    on ? 'Hiện lại tất cả câu' : 'Chỉ hiện câu cần soi video';
}}
// open the player straight on whatever is currently ranked #1
function watchTop(qid) {{
  const d = DATA[qid], st = state[qid];
  const c = candOf(qid, st.order[0]);
  if (d.task === 'trake') openInspector(qid, c[0], trakeFrames(qid)[0], 0);
  else openInspector(qid, c[0], c[1]);
}}

function clearAll() {{
  if (!confirm('Bỏ mọi thay đổi và quay về thứ hạng gốc?')) return;
  for (const id of qids) {{
    state[id] = {{order: [...Array(DATA[id].shown).keys()].map(i => 'c' + i),
                 extra: [], answer: '', touched: false, frames: null}};
  }}
  document.querySelectorAll('.ansbox input').forEach(i => (i.value = ''));
  document.getElementById('cmd').style.display = 'none';
  document.getElementById('dl').textContent = '';
  save(); paint();
}}

document.addEventListener('keydown', e => {{
  const lb = document.getElementById('lb');
  const viOpen = document.getElementById('vi').style.display === 'flex';
  if (e.key === 'Escape') {{
    lb.style.display = 'none';
    if (viOpen) closeInspector();
    return;
  }}
  if (viOpen) {{
    if (e.target.tagName === 'INPUT') return;
    // one SOURCE frame per arrow; shift-arrow jumps a whole keyframe
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {{
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      if (e.shiftKey) {{
        const i = nearestKf(vi.video, vi.frame) + dir;
        if (i >= 0 && i < VID[vi.video].k.length) gotoKf(i);
      }} else stepFrame(dir);
      e.preventDefault();
    }} else if (e.key === ' ') {{ togglePlay(); e.preventDefault(); }}
    else if (e.key === 'c' || e.key === 'C') {{ captureNow(); e.preventDefault(); }}
    else if (e.key === 'Enter') {{ useCurrentFrame(); e.preventDefault(); }}
    return;
  }}
  if (e.target.tagName === 'INPUT') return;
  if (lb.style.display === 'flex') return;
  if (e.key === 'j' || e.key === 'k') {{
    cur = Math.max(0, Math.min(qids.length - 1, cur + (e.key === 'j' ? 1 : -1)));
    document.getElementById(qids[cur]).scrollIntoView({{behavior:'smooth', block:'start'}});
  }} else if (e.key >= '1' && e.key <= '9') {{
    const items = itemsOf(qids[cur]);
    const el = items.find(x => state[qids[cur]].order.indexOf(x.dataset.id) === +e.key - 1);
    if (el) moveTo(qids[cur], el.dataset.id, 0);
  }}
}});

// keep `cur` pointing at whichever query is on screen, so 1-9 hits the right one
const seen = new IntersectionObserver(entries => entries.forEach(en => {{
  if (!en.isIntersecting) return;
  const i = qids.indexOf(en.target.dataset.qid);
  if (i >= 0) cur = i;
}}), {{rootMargin: '-40% 0px -55% 0px'}});
document.querySelectorAll('.q').forEach(q => seen.observe(q));

paint();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-out", default=None, help="the --out used for make_submission.py")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--top-trake", type=int, default=6, help="how many whole chains to show")
    ap.add_argument(
        "--transcripts",
        default=str(ROOT.parent / "transcripts_full"),
        help="comma-separated folders of transcript JSON; data/captions is always added",
    )
    ap.add_argument("--no-transcripts", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument(
        "--pool",
        type=int,
        default=80,
        help="candidates embedded for the in-browser export. Only the first --top are "
        "draggable; the rest fill the ladder ranks the operator never sees but the "
        "allocator needs, so the exported zip matches what make_submission would write",
    )
    args = ap.parse_args()
    if args.pool < args.top:
        args.pool = args.top

    qdir = Path(args.queries)
    out = Path(args.out) if args.out else qdir.parent / "review.html"
    run_out = args.run_out or str(qdir.parent / "run1")
    qfiles = sorted(
        p for p in qdir.glob("*.txt") if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    )

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta_by_key = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}

    # first pass: retrieve, so we know which frames need object captions
    retrieved = []
    trake_eng = None
    for qf in qfiles:
        text = read_query_text(qf) or ""
        task = detect_task(qf.name)
        # the SAME retrieval make_submission uses, including a hand-written
        # .en.txt.  Ranking the page differently from the CSV would have the
        # operator approving a frame that is not the one at row 1 — and the
        # runbook tells the team to write those .en.txt files.
        en = read_en_override(qf)

        if task == "trake":
            # Showing candidates for event 1 alone would be a lie: the CSV is
            # produced by aligning the WHOLE ordered chain inside one video, and
            # a wrong video scores 0 no matter how good its event-1 frame looks.
            # So review exactly what gets submitted — whole chains, best first.
            from src.task3_trake import TRAKEEngine

            if trake_eng is None:
                trake_eng = TRAKEEngine(engine=eng).load_index()
            events = split_events(en or text)
            import re as _re

            first = bool(_re.search(r"đầu tiên|lần đầu|first", text, _re.IGNORECASE))
            chains = trake_eng.align_sequence(
                events, first_occurrence=first, top_k=args.top_trake
            )
            if chains:
                retrieved.append((qf, text, task, [], chains, events))
            print(f"  {qf.stem:24s} {len(chains)} chains x {len(events)} events")
            continue

        probe = split_qa(text)[0] if task == "qa" else text
        hits = ranked_hits(eng, probe, en)[: args.pool]
        if hits:
            retrieved.append((qf, text, task, hits, None, None))
        print(f"  {qf.stem:24s} {len(hits)} candidates ({min(len(hits), args.top)} shown)")

    wanted = set()
    for _qf, _t, _task, hits, chains, _ev in retrieved:
        pairs = [(h.video_id, h.frame_idx) for h in hits[: args.top]]
        for c in chains or []:
            pairs += [(c["video_id"], f) for f in c["sequence_frames"]]
        for vid, fidx in pairs:
            m = meta_by_key.get((vid, fidx))
            if m:
                wanted.add((vid, Path(m["frame_filename"]).stem))
    # The videos that can appear anywhere on the page: candidates and chains.
    # Titles are needed by the block builder below, so this runs before it.
    page_videos = {v for v, _stem in wanted}
    for _qf, _t, _task, hits, chains, _ev in retrieved:
        page_videos.update(h.video_id for h in hits)
        page_videos.update(c["video_id"] for c in (chains or []))
    video_info = load_video_info(Path(args.data), page_videos)

    fps_lookup = {m["video_id"]: float(m["fps"]) for m in eng.metadata}

    colour_idx = None
    if not args.no_ocr:
        try:
            from src.core.ocr import ColourIndex

            colour_idx = ColourIndex(args.data)
            for v in page_videos:
                colour_idx._video(v)
            if colour_idx.n_frames:
                print(f"mau: {colour_idx.n_frames} khung hinh da do")
            else:
                colour_idx = None
        except Exception:  # noqa: BLE001
            colour_idx = None

    ocr = None
    if not args.no_ocr:
        try:
            from src.core.ocr import OCRIndex

            ocr = OCRIndex(args.data)
            for v in page_videos:
                ocr._video(v)          # warm the per-video cache files
            if ocr.n_frames:
                print(f"OCR: {ocr.n_frames} khung hinh da doc chu")
            else:
                print("OCR: chua doc khung hinh nao (chay scripts/run_ocr.py)")
                ocr = None
        except Exception as exc:  # noqa: BLE001
            print(f"  ! bo qua OCR ({type(exc).__name__}: {exc})")
            ocr = None
    tx = None
    if not args.no_transcripts:
        try:
            from src.core.transcripts import TranscriptIndex

            tx_tokenise = __import__('src.core.transcripts', fromlist=['tokenise']).tokenise
            tx = TranscriptIndex().load_dir(
                *[Path(d) for d in args.transcripts.split(",") if d.strip()],
                Path(args.data) / "captions",
            )
            print(f"loi thoai: {tx.n_videos} video")
            if not tx.n_videos:
                tx = None
        except Exception as exc:  # noqa: BLE001 - a nicety must not break the page
            print(f"  ! bo qua loi thoai ({type(exc).__name__}: {exc})")
            tx = None

    obj_labels = load_object_labels(Path(args.data), wanted)
    if obj_labels:
        print(f"\nobject captions for {len(obj_labels):,} frames")
    else:
        print("\n(no objects-aic25-b1.zip in data/ — thumbnails will have no object captions)")


    def vtitle(video_id: str, query_text: str) -> str:
        """The video's title, with words the query also uses marked.

        For TRAKE especially, the candidate videos are visually near-identical —
        six lion-dance performances from the same competition, or six episodes of
        the same cooking show — and the discriminator is the dish or the troupe,
        which is in the title and NOT in the pixels. Cross-checking round-1 by
        hand found that exactly two videos in the whole corpus mention "củ năng",
        and exactly one matches "măng tây" + "chiên"; neither was our top pick.

        Shown to the operator, never scored: video-level metadata was measured on
        the ground truth and made KIS retrieval WORSE (R@1 43.3% -> 40.0%).
        """
        info = video_info.get(video_id)
        if not info or not info.get("t"):
            return ""
        title = info["t"]
        words = {
            w for w in re.findall(r"\w{4,}", query_text.lower())
            if w not in _STOP
        }
        marked = []
        for tok in re.split(r"(\W+)", title):
            marked.append(f"«{tok}»" if tok.lower() in words else tok)
        return "".join(marked)

    def ocr_words(query_text: str) -> set:
        return {
            w.lower()
            for w in re.findall(r"[0-9A-Za-zÀ-ỹ]{3,}", query_text)
            if w.lower() not in _STOP
        }

    def said_panel(stem: str, query_text: str, shown_videos) -> str:
        """Videos whose SPEECH matches this query, with the passage and its time.

        Measured and deliberately kept OFF the scoring path. On the 60
        ground-truth queries a transcript bonus is negative at every weight, and
        only +0.5% (noise) even when gated on decisive evidence — because those
        queries are pure visual-scene descriptions ("a dark red sedan with a rear
        spoiler") that nobody says out loud.

        The round-1 set is not like that. Searching the transcripts by hand found
        MĂNG TÂY CHIÊN BIA for query-p1-4 at rank 1 and CỦ NĂNG OM NẤM CHAY for
        query-p1-18, neither of which the visual ranking surfaced. So the channel
        goes where the evidence supports: to the operator, marked as agreeing
        with the visual shortlist or contradicting it, and with a timestamp that
        opens the player at the moment the words were spoken.
        """
        if tx is None or not tx.n_videos:
            return ""
        # Only videos ALREADY on the shortlist. An earlier version also guessed
        # which un-ranked videos the transcript pointed at, and it was wrong in
        # both directions: it missed query-p1-4, whose discriminative unit is the
        # bigram "măng tây" rather than any single rare word, and volunteered
        # passages about unrelated robots for query-p1-21. Guessing badly is
        # worse than not guessing, because the operator has to check every row.
        # Finding a video the picture missed is a job for the search tool, where
        # the operator supplies the words. Here the transcript only supplies
        # CONTEXT for candidates the visual side already proposed — which is
        # exactly what settles a choice between two near-identical cooking shows.
        rows_html = []
        terms = {w for w in set(tx_tokenise(query_text)) if "_" not in w and len(w) > 2}
        for vid in shown_videos:
            if len(rows_html) >= 6:
                break
            seg = tx.best_segment(query_text, vid)
            if not seg:
                continue
            at, quote = seg
            marked, found = [], 0
            for tok in re.split(r"(\W+)", quote[:170]):
                if tok.lower() in terms:
                    marked.append(f"<b>{html.escape(tok)}</b>")
                    found += 1
                else:
                    marked.append(html.escape(tok))
            if found < 2:
                continue                      # one common word in common is not evidence
            frame_at = int(at * fps_lookup.get(vid, 25.0))
            rows_html.append(
                (found, at,
                 f'<div class="row2 inlist" onclick="openInspector('
                 f"'{html.escape(stem)}','{html.escape(vid)}',{frame_at})\">"
                 f'<span class="vid">{html.escape(vid)}</span>'
                 f'<span class="at">{int(at) // 60:d}:{int(at) % 60:02d}</span>'
                 f'<span class="quote">“{"".join(marked)}”</span></div>')
            )
        if not rows_html:
            return ""
        rows_html.sort(key=lambda r: (-r[0], r[1]))
        qq = query_text.replace("\n", " ").replace('"', "'")[:70]
        return (
            '<div class="said"><h4>🎙 Trong các video ứng viên, đây là chỗ lời nói '
            "khớp câu hỏi nhất — bấm để mở video đúng lúc đó.<br>"
            "Nghi hệ thống bỏ sót video? Tìm theo lời nói: "
            f'<code>python scripts/search_transcripts.py "{html.escape(qq)}"</code></h4>'
            + "".join(r[2] for r in rows_html)
            + "</div>"
        )

    def thumb(vid: str, fidx: int, label: str, extra: str = "") -> str:
        m = meta_by_key.get((vid, fidx))
        fn = m["frame_filename"] if m else None
        if fn is None:
            return ""
        objs = obj_labels.get((vid, Path(fn).stem), "")
        obj_html = f'<div class="objs">{html.escape(objs)}</div>' if objs else ""
        # What is WRITTEN in the shot. A news lower-third names the story, a
        # recipe card names the dish, a banner names the commune — none of which
        # an image embedding represents. OCR on the frame currently submitted for
        # query-p1-19 reads "Trích Văn bia THOẠI NGỌC HẦU"; the query is about
        # Nguyễn Trung Trực, so that answer is on the wrong video and nothing
        # else in this page could have said so.
        # Colour of the DETECTED SUBJECT, matched against colours the query asks
        # for. The lion-dance query names a yellow-black-white lion and the
        # shortlist came back with red ones; a global histogram could not have
        # said so, because the stage is red in every candidate.
        col_html = ""
        if colour_idx is not None and want_colours:
            have = colour_idx.names(vid, fidx)
            if have:
                bits = []
                for c in have[:3]:
                    bits.append(f"<b>{html.escape(c)}</b>" if c in want_colours else html.escape(c))
                missing = [c for c in want_colours if c not in have]
                tail = (f" · <i>thiếu {html.escape(', '.join(missing))}</i>"
                        if len(missing) == len(want_colours) else "")
                col_html = f'<div class="col">🎨 {", ".join(bits)}{tail}</div>'

        ocr_html = ""
        if ocr is not None:
            said = ocr.text_of(vid, fidx)
            if len(said) > 8:
                marked = []
                for tok in re.split(r"(\W+)", said[:220]):
                    marked.append(
                        f"<b>{html.escape(tok)}</b>" if tok.lower() in ocr_terms
                        else html.escape(tok)
                    )
                ocr_html = f'<div class="ocr" title="chu doc duoc tren khung hinh">🔤 {"".join(marked)}</div>'
        return (
            f'<figure data-v="{html.escape(vid)}" data-ff="{fidx}"{extra}>'
            f'<img loading="lazy" src="{CDN}/{vid}/{fn}" alt="">'
            f'<button class="zoom" title="Phóng to đọc chữ">🔍</button>'
            f'<button class="play" title="Xem video tại đúng khoảnh khắc này">▶ xem</button>'
            f'<span class="grab" title="Kéo để đổi thứ hạng">⠿</span>'
            f"{label}{col_html}{obj_html}{ocr_html}</figure>"
        )

    blocks = []
    warn_ids: List[str] = []
    ocr_terms: set = set()
    want_colours: list = []
    # everything the browser needs to rebuild the upload itself: the candidate
    # pool in retriever order, and — for TRAKE, whose lattice walk is not worth
    # a second implementation — the finished rows for each offered chain
    page_data: dict = {}
    for qf, text, task, hits, chains, events in retrieved:
        tags = f'<span class="tag">{task}</span>'
        ocr_terms = ocr_words(text)
        want_colours = colours_in_query(text)

        if chains is not None:
            top = chains[0]
            second = chains[1]["score"] if len(chains) > 1 else 0.0
            uncertain = top["score"] - second < 0.15 * max(abs(top["score"]), 1e-6)
            rows = []
            chain_rows = []
            for rank, c in enumerate(chains, 1):
                frames = c["sequence_frames"]
                cells = "".join(
                    thumb(
                        c["video_id"],
                        f,
                        f'<figcaption><span class="rank">E{j}</span>'
                        f"<span>frame {f}</span></figcaption>"
                        f'<div class="ev">{html.escape((events[j - 1] if j <= len(events) else "")[:70])}</div>',
                    )
                    for j, f in enumerate(frames, 1)
                )
                rows.append(
                    f'<div class="chain" data-q="{html.escape(qf.stem)}" data-id="c{rank - 1}"'
                    f' data-v="{html.escape(c["video_id"])}"'
                    f' data-f="{"|".join(str(f) for f in frames)}">'
                    f'<div class="ch"><span class="rank">#{rank}</span>'
                    f'<b>{html.escape(c["video_id"])}</b>'
                    f'<span class="vtitle">{html.escape(vtitle(c["video_id"], text))}</span>'
                    f'<span>điểm {c["score"]:.3f}</span>'
                    f'<span class="grab" style="position:static;opacity:1">⠿ kéo</span>'
                    f'<span>frames {", ".join(str(f) for f in frames)}</span></div>'
                    f'<div class="grid">{cells}</div></div>'
                )
                # the lattice walk in allocate_trake_rows is heap-ordered and not
                # worth reimplementing in JS just to risk disagreeing with it, so
                # the finished rows for every offered chain ride along
                chain_rows.append(
                    [
                        [v, *fr]
                        for v, fr in allocate_trake_rows(
                            c["video_id"], frames, budget=MAX_ROWS, step=10,
                            video_last_frame=eng.last_frame.get(c["video_id"]),
                        )
                    ]
                )
            page_data[qf.stem] = {
                "task": "trake",
                "text": text[:600],
                "shown": len(chains),
                "cands": [[c["video_id"], c["sequence_frames"][0], None] for c in chains],
                "chainRows": chain_rows,
            }
            if uncertain:
                tags += '<span class="tag warn">cần xem kỹ</span>'
                warn_ids.append(qf.stem)
            # a chain is a sequence in time; it can only really be judged by
            # watching it happen, so TRAKE always gets the video treatment
            blocks.append(
                f'<div class="q scrutiny" id="{html.escape(qf.stem)}"'
                f' data-qid="{html.escape(qf.stem)}">'
                f'<div class="qh"><span class="qid">{html.escape(qf.stem)}</span>'
                f'{tags}<span class="tag">top {html.escape(top["video_id"])}</span>'
            f'<span class="vtitle">{html.escape(vtitle(top["video_id"], text))}</span>'
                f'<span class="tag ok badge"></span>'
                f'<button class="watch" onclick="watchTop(\'{html.escape(qf.stem)}\')">'
                f'▶ Xem video chuỗi này</button></div>'
                f'<div class="qtext">{html.escape(text[:400])}</div>'
                f'{said_panel(qf.stem, text, {c["video_id"] for c in chains})}'
                f'{"".join(rows)}</div>'
            )
            continue

        shown = hits[: args.top]
        vids = Counter(h.video_id for h in shown)
        top_share = vids.most_common(1)[0][1] / len(shown)
        others = [h.score for h in shown if h.video_id != shown[0].video_id]
        import numpy as np

        sd = float(np.std([h.score for h in shown])) + 1e-6
        margin = (shown[0].score - max(others)) / sd if others else 9.9
        # How far ahead is the best video, as a FRACTION of its own score?
        # Checked against the round-1 submission the operator hand-corrected:
        # all five queries they had to fix had a rank-1-vs-rank-2 video gap of
        # 1.8% or less, and no query above 4% needed fixing. The sigma-based
        # margin alone missed query-p1-24 (0.99 sigma but a 1.6% gap), so both
        # signals are used — a missed query costs a whole point, an extra
        # flagged one costs a minute of the operator's time.
        rel = (
            (shown[0].score - max(others)) / abs(shown[0].score)
            if others and shown[0].score
            else 1.0
        )
        uncertain = margin <= 0.8 or rel <= 0.02

        figs = [
            thumb(
                h.video_id,
                h.frame_idx,
                f'<figcaption><span class="rank">#{rank}</span>'
                f'<span>{html.escape(h.video_id)}</span>'
                f"<span>{h.score:.3f}</span></figcaption>",
                extra=f' data-q="{html.escape(qf.stem)}" data-id="c{rank - 1}"'
                f' data-f="{h.frame_idx}"',
            )
            for rank, h in enumerate(shown, 1)
        ]
        page_data[qf.stem] = {
            "task": task,
            "text": text[:600],
            "shown": len(shown),
            # the pool beyond --top is never drawn but the allocator needs it to
            # fill ranks 31-100, so the exported zip matches make_submission
            "cands": [[h.video_id, int(h.frame_idx), h.video_last_frame] for h in hits],
        }

        if uncertain:
            tags += '<span class="tag warn">cần xem kỹ</span>'
            warn_ids.append(qf.stem)

        # A Q&A row without an answer string scores zero however good the frame
        # is, and the Gemini answerer needs an API key that is not always there.
        # Reading the frame is something the operator can always do, so the page
        # takes a typed answer directly.
        ansbox = ""
        if task == "qa":
            ansbox = (
                f'<div class="ansbox"><label>Đáp án cho câu này '
                f"(bắt buộc — bỏ trống là 0 điểm; phóng to khung hình để đọc chữ)</label>"
                f'<input type="text" data-q="{html.escape(qf.stem)}"'
                f' data-v="{html.escape(hits[0].video_id)}" data-f="{hits[0].frame_idx}"'
                f' placeholder="ví dụ: Xã Vạn Thắng"></div>'
            )
            if qf.stem not in warn_ids:
                warn_ids.append(qf.stem)

        # Q&A needs the video because the answer is usually spoken or written on
        # screen; an uncertain query needs it because nothing else will settle it
        scrutiny = " scrutiny" if (task == "qa" or uncertain) else ""
        blocks.append(
            f'<div class="q{scrutiny}" id="{html.escape(qf.stem)}"'
            f' data-qid="{html.escape(qf.stem)}">'
            f'<div class="qh"><span class="qid">{html.escape(qf.stem)}</span>'
            f'{tags}<span class="tag">top {html.escape(shown[0].video_id)}</span>'
            f'<span class="vtitle">{html.escape(vtitle(shown[0].video_id, text))}</span>'
            f'<span class="tag ok badge"></span>'
            f'<button class="watch" onclick="watchTop(\'{html.escape(qf.stem)}\')">'
            f'▶ Xem video</button></div>'
            f'<div class="qtext">{html.escape(text[:400])}</div>'
            f'{said_panel(qf.stem, text, {h.video_id for h in shown})}'
            f'{ansbox}'
            f'<div class="grid">{"".join(figs)}</div></div>'
        )

    # Per-video data for the inspector: the YouTube id, the fps, and the full
    # keyframe timeline. Everything else is derivable — frame_filename is always
    # f"{n:03d}.jpg", n is 1-based and consecutive, and pts_time == frame_idx/fps
    # (all three verified across the whole 177,321-frame corpus), so storing the
    # frame_idx array alone keeps the page a few hundred KB instead of megabytes.
    inspect_videos = {c[0] for d in page_data.values() for c in d["cands"][: d["shown"]]}
    kf_by_video: dict = {}
    fps_by_video: dict = {}
    for m in eng.metadata:
        v = m["video_id"]
        if v in inspect_videos:
            kf_by_video.setdefault(v, []).append(int(m["frame_idx"]))
            fps_by_video[v] = float(m["fps"])
    info = video_info
    vid_data = {
        v: {
            "y": info.get(v, {}).get("y", ""),
            "t": info.get(v, {}).get("t", ""),
            "f": fps_by_video.get(v, 25.0),
            "l": int(eng.last_frame.get(v, max(ks))),
            "k": sorted(ks),
        }
        for v, ks in kf_by_video.items()
    }
    missing = [v for v, d in vid_data.items() if not d["y"]]
    print(f"video inspector: {len(vid_data)} videos"
          + (f", {len(missing)} without a YouTube link" if missing else ", all with a YouTube link"))

    alloc_js = (ROOT / "scripts" / "review_export.js").read_text(encoding="utf-8")
    plan = {
        "breadthCost": 1.0,
        "depthCost": DEFAULT_DEPTH_COST,
        "step": 10,
        "budget": MAX_ROWS,
        "maxDepth": 24,
        "nFlat": DEFAULT_N_FLAT,
    }

    page = PAGE.format(
        body="\n".join(blocks),
        qdir=qdir.as_posix(),
        outdir=run_out.replace("\\", "/"),
        nq=len(blocks),
        warnlist=json.dumps(warn_ids),
        tag=qdir.parent.name or "round",
        alloc_js=alloc_js,
        data_json=json.dumps(page_data, separators=(",", ":")),
        plan_json=json.dumps(plan),
        vid_json=json.dumps(vid_data, separators=(",", ":")),
        cdn=CDN,
    )
    out.write_text(page, encoding="utf-8")

    print(f"\nwrote {out}   ({len(blocks)} queries, {len(warn_ids)} flagged uncertain, "
          f"{len(page) // 1024} KB)")
    print("Mở file đó, kéo thả khung hình đúng lên #1, điền đáp án Q&A,")
    print("rồi bấm 'Tải submission.zip' — file nộp được tạo ngay trong trình duyệt.")
    print(f"Kiểm tra trước khi nộp:  python scripts/verify_zip.py <file> --queries {qdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
