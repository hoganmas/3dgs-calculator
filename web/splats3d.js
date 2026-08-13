import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const splatVertexShader = /* glsl */ `
  attribute float aWeight;
  attribute float aScale;

  varying vec2 vUv;
  varying float vWeight;

  void main() {
    vUv = uv;
    vWeight = aWeight;

    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vec2 quad = (uv - 0.5) * 2.0 * aScale;
    mvPosition.xy += quad;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

// Writes a *signed* scalar into the float render target (R channel).
const splatFragmentShader = /* glsl */ `
  varying vec2 vUv;
  varying float vWeight;

  uniform float uGain;

  void main() {
    vec2 p = (vUv - 0.5) * 2.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    float g = exp(-0.5 * r2 * 4.0);
    float signedValue = vWeight * uGain * g;
    gl_FragColor = vec4(signedValue, 0.0, 0.0, 1.0);
  }
`;

const compositeVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// Map signed field → diverging colors; negatives cancel positives in the RT first.
const compositeFragmentShader = /* glsl */ `
  varying vec2 vUv;

  uniform sampler2D tField;
  uniform float uIntensity;
  uniform vec3 uPosColor;
  uniform vec3 uNegColor;

  void main() {
    float field = texture2D(tField, vUv).r * uIntensity;
    float mag = abs(field);
    if (mag < 1e-5) discard;

    vec3 color = field >= 0.0 ? uPosColor : uNegColor;
    float tone = tanh(mag);
    float alpha = clamp(tone, 0.0, 1.0);
    gl_FragColor = vec4(color * tone, alpha);
  }
`;

function evalPoly(coeffs, x) {
  let y = 0;
  let xp = 1;
  for (const c of coeffs) {
    y += c * xp;
    xp *= x;
  }
  return y;
}

function centerPos(c) {
  return {
    x: c.x ?? c.center ?? 0,
    y: c.y ?? 0,
    z: c.z ?? 0,
    weight: c.weight,
  };
}

export class SplatViewer {
  /**
   * @param {HTMLElement} container
   */
  constructor(container) {
    this.container = container;
    this.centers = [];
    this.coeffs = [];
    this.dims = 1;
    this.intensity = 1.0;
    this.baseGain = 1.0;

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setClearColor(0x14110e, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.autoClear = false;
    this.container.appendChild(this.renderer.domElement);

    // World helpers / reference curve
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x14110e, 18, 42);

    // Signed splat accumulation (no helpers)
    this.splatScene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.camera.position.set(4.5, 2.8, 7.5);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0.4, 0);
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 40;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xfff2dd, 0.85);
    key.position.set(4, 8, 2);
    this.scene.add(key);

    this._buildHelpers();
    this._buildComposite();
    this.splatMesh = null;
    this.curveLine = null;
    this.fieldTarget = null;

    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    this.resize();
    this._raf = requestAnimationFrame(() => this._frame());
  }

  _buildHelpers() {
    const grid = new THREE.GridHelper(16, 16, 0x3a342c, 0x2a251f);
    grid.position.y = -0.001;
    this.scene.add(grid);

    const axisMat = new THREE.LineBasicMaterial({ color: 0x8a7d68 });
    const axes = [
      [new THREE.Vector3(-6, 0, 0), new THREE.Vector3(6, 0, 0)],
      [new THREE.Vector3(0, 0, -6), new THREE.Vector3(0, 0, 6)],
    ];
    for (const [a, b] of axes) {
      const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
      this.scene.add(new THREE.Line(geo, axisMat));
    }
  }

  _buildComposite() {
    this.compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.compositeScene = new THREE.Scene();
    this.compositeMaterial = new THREE.ShaderMaterial({
      vertexShader: compositeVertexShader,
      fragmentShader: compositeFragmentShader,
      uniforms: {
        tField: { value: null },
        uIntensity: { value: this.intensity },
        uPosColor: { value: new THREE.Color("#3ecf9f") },
        uNegColor: { value: new THREE.Color("#ff6b4a") },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial);
    this.compositeScene.add(quad);
  }

  _ensureFieldTarget(width, height) {
    const tw = Math.max(1, Math.floor(width));
    const th = Math.max(1, Math.floor(height));
    if (
      this.fieldTarget &&
      this.fieldTarget.width === tw &&
      this.fieldTarget.height === th
    ) {
      return;
    }
    if (this.fieldTarget) this.fieldTarget.dispose();

    this.fieldTarget = new THREE.WebGLRenderTarget(tw, th, {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.compositeMaterial.uniforms.tField.value = this.fieldTarget.texture;
  }

  /**
   * User-facing overall intensity / alpha gain (default 1).
   * @param {number} value
   */
  setIntensity(value) {
    this.intensity = Math.max(0, Number(value) || 0);
    this.compositeMaterial.uniforms.uIntensity.value = this.intensity;
  }

  resize() {
    const width = this.container.clientWidth || 640;
    const height = this.container.clientHeight || 420;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    const pr = this.renderer.getPixelRatio();
    this._ensureFieldTarget(width * pr, height * pr);
  }

  setResult(result) {
    this.centers = (result.centers || [])
      .map(centerPos)
      .filter((c) => Math.abs(c.weight) > 1e-12);
    this.coeffs = result.coeffs || [];
    this.dims = result.dims || 1;
    this._rebuildSplats();
    this._rebuildCurve();
  }

  _rebuildSplats() {
    if (this.splatMesh) {
      this.splatScene.remove(this.splatMesh);
      this.splatMesh.geometry.dispose();
      this.splatMesh.material.dispose();
      this.splatMesh = null;
    }

    const n = this.centers.length;
    if (!n) return;

    const maxW = Math.max(...this.centers.map((c) => Math.abs(c.weight)), 1e-9);
    this.baseGain =
      this.dims >= 2 ? 0.35 / Math.pow(maxW, 0.4) : 0.55 / Math.sqrt(maxW);

    const geometry = new THREE.PlaneGeometry(1, 1);
    const weights = new Float32Array(n);
    const scales = new Float32Array(n);
    const baseScale = this.dims >= 2 ? 1.15 : 2.2;
    const boost = this.dims >= 2 ? 0.9 : 1.4;

    for (let i = 0; i < n; i += 1) {
      const { weight } = this.centers[i];
      weights[i] = weight;
      const strength = Math.abs(weight) / maxW;
      scales[i] = baseScale + boost * strength;
    }

    geometry.setAttribute("aWeight", new THREE.InstancedBufferAttribute(weights, 1));
    geometry.setAttribute("aScale", new THREE.InstancedBufferAttribute(scales, 1));

    const material = new THREE.ShaderMaterial({
      vertexShader: splatVertexShader,
      fragmentShader: splatFragmentShader,
      uniforms: {
        uGain: { value: this.baseGain },
      },
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneFactor,
    });

    const mesh = new THREE.InstancedMesh(geometry, material, n);
    mesh.frustumCulled = false;
    const dummy = new THREE.Object3D();
    for (let i = 0; i < n; i += 1) {
      const { x, y, z } = this.centers[i];
      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    this.splatMesh = mesh;
    this.splatScene.add(mesh);

    const xs = this.centers.map((c) => c.x);
    const ys = this.centers.map((c) => c.y);
    const zs = this.centers.map((c) => c.z);
    const mid = new THREE.Vector3(
      0.5 * (Math.min(...xs) + Math.max(...xs)),
      0.5 * (Math.min(...ys) + Math.max(...ys)),
      0.5 * (Math.min(...zs) + Math.max(...zs))
    );
    const span = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...ys) - Math.min(...ys),
      Math.max(...zs) - Math.min(...zs),
      2
    );
    this.controls.target.copy(mid);
    this.camera.position.set(
      mid.x + span * 1.1,
      mid.y + span * 0.7,
      mid.z + span * 1.4
    );
    this.controls.update();
  }

  _rebuildCurve() {
    if (this.curveLine) {
      this.scene.remove(this.curveLine);
      this.curveLine.geometry.dispose();
      this.curveLine.material.dispose();
      this.curveLine = null;
    }
    if (this.dims !== 1 || !this.coeffs.length) return;

    const points = [];
    const xMin = -4;
    const xMax = 4;
    const samples = 240;
    let peak = 1e-9;
    const raw = [];
    for (let i = 0; i < samples; i += 1) {
      const x = xMin + ((xMax - xMin) * i) / (samples - 1);
      const y = evalPoly(this.coeffs, x) * Math.exp(-0.5 * x * x);
      raw.push({ x, y });
      peak = Math.max(peak, Math.abs(y));
    }
    const amp = 1.8 / peak;
    for (const { x, y } of raw) {
      points.push(new THREE.Vector3(x, 0.2 + y * amp, 0));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xf0e2b8,
      transparent: true,
      opacity: 0.9,
    });
    this.curveLine = new THREE.Line(geometry, material);
    this.scene.add(this.curveLine);
  }

  _sortSplats() {
    if (!this.splatMesh || !this.centers.length) return;

    const cam = this.camera.position;
    const order = this.centers
      .map((c, i) => ({
        i,
        d: (c.x - cam.x) ** 2 + (c.y - cam.y) ** 2 + (c.z - cam.z) ** 2,
      }))
      .sort((a, b) => b.d - a.d);

    const dummy = new THREE.Object3D();
    const weights = this.splatMesh.geometry.getAttribute("aWeight");
    const scales = this.splatMesh.geometry.getAttribute("aScale");
    const maxW = Math.max(...this.centers.map((c) => Math.abs(c.weight)), 1e-9);
    const baseScale = this.dims >= 2 ? 1.15 : 2.2;
    const boost = this.dims >= 2 ? 0.9 : 1.4;

    for (let k = 0; k < order.length; k += 1) {
      const src = order[k].i;
      const { x, y, z, weight } = this.centers[src];
      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      this.splatMesh.setMatrixAt(k, dummy.matrix);
      weights.array[k] = weight;
      const strength = Math.abs(weight) / maxW;
      scales.array[k] = baseScale + boost * strength;
    }
    weights.needsUpdate = true;
    scales.needsUpdate = true;
    this.splatMesh.instanceMatrix.needsUpdate = true;
  }

  _frame() {
    this.controls.update();
    this._sortSplats();

    const size = new THREE.Vector2();
    this.renderer.getDrawingBufferSize(size);
    this._ensureFieldTarget(size.x, size.y);

    // 1) Background helpers
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(0x14110e, 1);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);

    // 2) Signed field accumulation (float RT): +W adds, −W subtracts
    if (this.splatMesh && this.fieldTarget) {
      this.splatMesh.material.uniforms.uGain.value = this.baseGain;
      this.renderer.setRenderTarget(this.fieldTarget);
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.clear();
      this.renderer.render(this.splatScene, this.camera);

      // 3) Composite signed field over the background
      this.compositeMaterial.uniforms.uIntensity.value = this.intensity;
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.compositeScene, this.compositeCamera);
    }

    this._raf = requestAnimationFrame(() => this._frame());
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    window.removeEventListener("resize", this._onResize);
    this.controls.dispose();
    if (this.fieldTarget) this.fieldTarget.dispose();
    this.renderer.dispose();
  }
}
