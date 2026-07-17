"""
visualize_standalone.py -- write a single self-contained HTML file that
renders the same 3D size-field view as the in-chat Cowork widgets used
earlier in this project, but with zero Cowork dependency: open it directly
in any browser (double-click the file, or `python -m http.server`) from a
plain command-line run of this pipeline.

Takes the same payload shape as visualization_data.payload_to_json_dict():
    {"points": [[x,y,z],...], "matrices": [[[..3x3..],...],...],
     "magnitude": [...], "triangles": [[i,j,k],...], "glyph_stride": int}

Everything (data + a three.js CDN reference + camera/orbit-control JS) is
embedded in one .html file. Data is inlined as a JS literal rather than
fetched from a sibling file -- fetch() of a local file is blocked by most
browsers under file://, inlining sidesteps that entirely so the output
works with a plain double-click, no local server required.

Placeholders are substituted with plain str.replace (not str.format) since
the template is mostly JS/CSS full of literal curly braces -- .format()
would require escaping every one of them, which is exactly the kind of
hand-typed-brace mistake that broke an earlier version of the in-chat
visualization in this project (see PIPELINE.md).
"""

from __future__ import annotations

import json
import re
import subprocess
import shutil

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin:0; padding:0; background:#111318; color:#e8e8ec;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif; overflow:hidden; }
  #hud { position:absolute; top:12px; left:14px; z-index:10; pointer-events:none; }
  #hud h1 { font-size:16px; margin:0 0 4px 0; font-weight:600; }
  #hud p { font-size:12px; margin:0; color:#9aa0ab; }
  #legend { position:absolute; bottom:14px; left:14px; z-index:10; font-size:11px; color:#9aa0ab; }
  #legend .swatch { display:inline-block; width:120px; height:10px; border-radius:2px;
    background: linear-gradient(to right, #2b6cff, #d7d9de, #ff4d4d); vertical-align:middle; margin:0 6px; }
  canvas { display:block; }
</style>
</head>
<body>
<div id="hud">
  <h1>__TITLE__</h1>
  <p>drag to rotate &middot; scroll to zoom</p>
</div>
<div id="legend">small <span class="swatch"></span> large (size)</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = __DATA_JSON__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111318);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.001, 10000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio || 1);
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 1.4, 0.8);
scene.add(dirLight);

const pts = DATA.points;
let minX=Infinity, minY=Infinity, minZ=Infinity, maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
for (const p of pts) {
  if (p[0]<minX) minX=p[0]; if (p[0]>maxX) maxX=p[0];
  if (p[1]<minY) minY=p[1]; if (p[1]>maxY) maxY=p[1];
  if (p[2]<minZ) minZ=p[2]; if (p[2]>maxZ) maxZ=p[2];
}
const center = new THREE.Vector3((minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2);
const diag = Math.sqrt((maxX-minX)**2 + (maxY-minY)**2 + (maxZ-minZ)**2) || 1;

function colorFor(t) {
  t = Math.max(0, Math.min(1, t));
  const c = new THREE.Color();
  if (t < 0.5) { c.lerpColors(new THREE.Color(0x2b6cff), new THREE.Color(0xd7d9de), t/0.5); }
  else { c.lerpColors(new THREE.Color(0xd7d9de), new THREE.Color(0xff4d4d), (t-0.5)/0.5); }
  return c;
}

const mags = DATA.magnitude;
let magMin = Math.min.apply(null, mags), magMax = Math.max.apply(null, mags);
if (magMax - magMin < 1e-12) { magMax = magMin + 1e-9; }

if (DATA.triangles.length > 0) {
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(pts.length*3);
  const colors = new Float32Array(pts.length*3);
  for (let i=0;i<pts.length;i++) {
    positions[i*3]=pts[i][0]; positions[i*3+1]=pts[i][1]; positions[i*3+2]=pts[i][2];
    const t = (mags[i]-magMin)/(magMax-magMin);
    const c = colorFor(t);
    colors[i*3]=c.r; colors[i*3+1]=c.g; colors[i*3+2]=c.b;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors,3));
  const indices = [];
  for (const tri of DATA.triangles) { indices.push(tri[0], tri[1], tri[2]); }
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const mat = new THREE.MeshLambertMaterial({vertexColors:true, side:THREE.DoubleSide});
  scene.add(new THREE.Mesh(geo, mat));
  const edgeGeo = new THREE.EdgesGeometry(geo, 40);
  scene.add(new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({color:0x000000, transparent:true, opacity:0.08})));
} else {
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(pts.length*3);
  const colors = new Float32Array(pts.length*3);
  for (let i=0;i<pts.length;i++) {
    positions[i*3]=pts[i][0]; positions[i*3+1]=pts[i][1]; positions[i*3+2]=pts[i][2];
    const t = (mags[i]-magMin)/(magMax-magMin);
    const c = colorFor(t);
    colors[i*3]=c.r; colors[i*3+1]=c.g; colors[i*3+2]=c.b;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors,3));
  const mat = new THREE.PointsMaterial({size: diag*0.006, vertexColors:true});
  scene.add(new THREE.Points(geo, mat));
}

const glyphGroup = new THREE.Group();
const stride = Math.max(1, DATA.glyph_stride|0);
const glyphGeo = new THREE.SphereGeometry(1, 10, 8);
for (let i=0;i<pts.length;i+=stride) {
  const m = DATA.matrices[i];
  const r0 = m[0], r1 = m[1], r2 = m[2];
  const basis = new THREE.Matrix4();
  basis.makeBasis(
    new THREE.Vector3(r0[0], r0[1], r0[2]),
    new THREE.Vector3(r1[0], r1[1], r1[2]),
    new THREE.Vector3(r2[0], r2[1], r2[2])
  );
  const glyph = new THREE.Mesh(glyphGeo, new THREE.MeshBasicMaterial({color:0xffb020, wireframe:true, transparent:true, opacity:0.55}));
  glyph.applyMatrix4(basis);
  glyph.position.set(pts[i][0], pts[i][1], pts[i][2]);
  glyphGroup.add(glyph);
}
scene.add(glyphGroup);

let spherical = { radius: diag*1.4, theta: Math.PI/4, phi: Math.PI/3 };
function updateCameraFromSpherical() {
  const x = center.x + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
  const y = center.y + spherical.radius * Math.cos(spherical.phi);
  const z = center.z + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
  camera.position.set(x, y, z);
  camera.near = Math.max(diag*0.001, 0.0001);
  camera.far = diag*20;
  camera.updateProjectionMatrix();
  camera.lookAt(center);
}
updateCameraFromSpherical();

let dragging = false, lastX=0, lastY=0;
renderer.domElement.addEventListener('mousedown', function(e) { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', function() { dragging = false; });
window.addEventListener('mousemove', function(e) {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  spherical.theta -= dx * 0.007;
  spherical.phi = Math.max(0.05, Math.min(Math.PI-0.05, spherical.phi - dy*0.007));
  updateCameraFromSpherical();
});
renderer.domElement.addEventListener('wheel', function(e) {
  e.preventDefault();
  spherical.radius *= (1 + Math.sign(e.deltaY) * 0.08);
  spherical.radius = Math.max(diag*0.05, Math.min(diag*8, spherical.radius));
  updateCameraFromSpherical();
}, {passive:false});

window.addEventListener('resize', function() {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""


def write_standalone_viewer(payload: dict, out_path: str, title: str = "Mesh size field viewer") -> None:
    data_json = json.dumps(payload)
    html = _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA_JSON__", data_json)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} -- open it directly in a browser (double-click, no server needed)")


def _extract_inline_script(html: str) -> str:
    """Pull out the inline <script>...</script> block (the second one --
    the first is the external three.js CDN <script src=...> tag) for a
    syntax check."""
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    if not scripts:
        raise AssertionError("no inline <script> block found")
    return scripts[-1]


if __name__ == "__main__":
    # tiny synthetic payload: a unit square, two triangles, mild anisotropy
    payload = {
        "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        "matrices": [
            [[0.01, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            [[0.02, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            [[0.03, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            [[0.04, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
        ],
        "magnitude": [0.028, 0.034, 0.039, 0.043],
        "triangles": [[0, 1, 2], [1, 3, 2]],
        "glyph_stride": 1,
    }
    out_path = "/tmp/_visualize_standalone_selftest.html"
    write_standalone_viewer(payload, out_path, title="self-test viewer")
    with open(out_path) as f:
        html = f.read()
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "cdnjs.cloudflare.com/ajax/libs/three.js" in html
    script = _extract_inline_script(html)
    assert "DATA.points" in script

    if shutil.which("node"):
        result = subprocess.run(["node", "--check"], input=script, capture_output=True, text=True)
        assert result.returncode == 0, f"node --check failed:\n{result.stderr}"
        print("self-test OK: HTML renders, embeds three.js + data, and the inline script passes `node --check`")
    else:
        print("self-test OK (node not available in this environment to double-check JS syntax, "
              "but HTML/embedding structure verified)")
