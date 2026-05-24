/**
 * ImageVisualizer.jsx  —  v1.2
 * Centre panel: image display with laser scan FX and canvas overlay.
 *
 * v1.0: red crosshairs (angular) + blue rings (round)
 * v1.1: gravitational aura, connection lines, EM-break border, instability ring
 * v1.2: bounding boxes for detected_objects; click on a box → object focus mode
 *        (highlights selected box in gold, calls onObjectSelect)
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ImageProcessor } from '../utils/imageProcessor';

/**
 * @param {{
 *   imageUrl: string | null,
 *   isScanning: boolean,
 *   scanComplete: boolean,
 *   result: object | null,
 *   selectedObjectId: string | null,
 *   onObjectSelect: (id: string | null) => void,
 *   sectorMode: string,
 * }} props
 */
export default function ImageVisualizer({ imageUrl, isScanning, scanComplete, result, selectedObjectId, onObjectSelect, sectorMode = 'INDIVIDUUM' }) {
  const imgRef     = useRef(null);
  const canvasRef  = useRef(null);
  const wrapperRef = useRef(null);

  // Stores letterbox geometry so the click handler can convert canvas coords → ratios
  const layoutRef = useRef({ ox: 0, oy: 0, iw: 0, ih: 0 });

  // v1.0: sharp / round markers
  const [markers, setMarkers] = useState([]);
  // v1.1: gravitational centroids and circle-cluster positions
  const [gravZones,      setGravZones]      = useState([]);
  const [circleClusters, setCircleClusters] = useState([]);

  // ─── Compute all overlay data when image loads ─────────────────────────────

  const computeOverlayData = useCallback(() => {
    if (!imgRef.current?.complete) return;
    try {
      const proc = new ImageProcessor(imgRef.current);
      setMarkers(proc.findMarkerPositions(22));
      setGravZones(proc.findGravitationalCentroids());      // v1.1
      setCircleClusters(proc.findCircleClusters());         // v1.1
    } catch {
      // Canvas taint (cross-origin image) — skip gracefully
      setMarkers([]);
      setGravZones([]);
      setCircleClusters([]);
    }
  }, []);

  // ─── Draw full canvas overlay ──────────────────────────────────────────────

  useEffect(() => {
    if (!scanComplete || !canvasRef.current || !imgRef.current) return;

    const canvas = canvasRef.current;
    const img    = imgRef.current;

    // Match canvas resolution to its actual CSS size (it covers the full flex container).
    // Then compute letterbox offsets so all markers land on the image, not the padding.
    const canvasRect = canvas.getBoundingClientRect();
    const imgRect    = img.getBoundingClientRect();
    canvas.width  = Math.round(canvasRect.width);
    canvas.height = Math.round(canvasRect.height);

    // ox/oy: pixel offset from canvas origin to the image's top-left corner
    const ox = Math.round(imgRect.left - canvasRect.left);
    const oy = Math.round(imgRect.top  - canvasRect.top);
    const iw = Math.round(imgRect.width);
    const ih = Math.round(imgRect.height);

    // Persist layout so the click handler can convert mouse coords without DOM access
    layoutRef.current = { ox, oy, iw, ih };

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const qfm = result?.dimensions_analysis?.quantum_field_metrics;

    // ── Layer 1: Gravitational Aura (v1.1) ────────────────────────────────────
    if (qfm && qfm.gravitational_density > 0.10) {
      gravZones.forEach(({ xRatio, yRatio, intensity }) => {
        drawGravitationalAura(ctx, ox + xRatio * iw, oy + yRatio * ih, intensity);
      });
    }

    // ── Layer 2: v1.0 markers ─────────────────────────────────────────────────
    markers.forEach(({ xRatio, yRatio, type }) => {
      const x = ox + xRatio * iw;
      const y = oy + yRatio * ih;
      if (type === 'sharp') drawCrosshair(ctx, x, y, 10, '#ff2d55');
      else                  drawRing(ctx, x, y, 10, '#00d4ff');
    });

    // ── Layer 3: Stabilization-Atom connection lines (v1.1) ───────────────────
    if (qfm && qfm.detected_particles.circle_count > 0) {
      const sharpMarkers = markers.filter(m => m.type === 'sharp');
      drawConnectionLines(ctx, circleClusters, sharpMarkers, ox, oy, iw, ih);
    }

    // ── Layer 4: Instability ring (v1.1) ──────────────────────────────────────
    if (qfm && qfm.stability_status.includes('Unstable')) {
      drawInstabilityRing(ctx, ox, oy, iw, ih);
    }

    // ── Layer 5: Detected-object bounding boxes (v1.2) ────────────────────────
    const objects = result?.detected_objects ?? [];
    objects.forEach(obj => {
      drawBoundingBox(ctx, obj, ox, oy, iw, ih, obj.id === selectedObjectId);
    });

    // ── Layer 6: Focus zone overlay (v1.28) — drawn last so it sits on top ────
    if (sectorMode === 'CARDS' || sectorMode === 'INDIVIDUUM') {
      drawFocusZone(ctx, sectorMode, ox, oy, iw, ih);
    }

  }, [scanComplete, markers, gravZones, circleClusters, result, selectedObjectId, sectorMode]);

  // ─── v1.2: Bounding box click handler ─────────────────────────────────────

  const handleCanvasClick = useCallback((e) => {
    const objects = result?.detected_objects;
    if (!objects?.length || !canvasRef.current) return;

    const { ox, oy, iw, ih } = layoutRef.current;
    if (iw === 0 || ih === 0) return;

    const rect  = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Convert canvas-pixel click to image ratios
    const xRatio = (clickX - ox) / iw;
    const yRatio = (clickY - oy) / ih;

    const hit = objects.find(obj => {
      const [bx1, by1, bx2, by2] = obj.bounding_box;
      return xRatio >= bx1 && xRatio <= bx2 && yRatio >= by1 && yRatio <= by2;
    });

    // Toggle: clicking the already-selected object deselects it
    onObjectSelect?.(hit?.id === selectedObjectId ? null : (hit?.id ?? null));
  }, [result, selectedObjectId, onObjectSelect]);

  // ─── Quantum field state flags for JSX badges / border ────────────────────

  const qfm        = result?.dimensions_analysis?.quantum_field_metrics;
  const emBreak    = qfm?.electromagnetic_break_detected ?? false;
  const isUnstable = qfm?.stability_status?.includes('Unstable') ?? false;
  const luxury     = qfm?.luxury_particle_detected ?? false;
  const compress   = qfm?.active_hyperdimensional_compression ?? false;
  const hasObjects = (result?.detected_objects?.length ?? 0) > 0;

  return (
    <div
      ref={wrapperRef}
      className={`panel flex flex-col h-full overflow-hidden transition-all duration-500
        ${emBreak    ? 'border-[#ff6b1a]/60' : ''}
        ${isUnstable ? 'border-[#ff2d55]/60' : ''}
        ${luxury     ? 'border-[#ffd700]/40' : ''}
      `}
    >
      {/* ── Panel header ── */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#1a1a3e] flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#00d4ff] animate-pulse" />
          <span className="text-[10px] tracking-[0.25em] text-[#00d4ff]/70 uppercase">
            Visual Decoder  <span className="text-[#4a5568]">v1.2</span>
          </span>
        </div>

        {scanComplete && result && (
          <div className="flex items-center gap-2 flex-wrap">
            <Badge color="orange" label={result.dimensions_analysis.dim_9_geometry.dominant_shape} />

            {result.dimensions_analysis.symmetry.is_susy && (
              <Badge color="cyan" label="SUSY" />
            )}
            {result.dimensions_analysis.symmetry.is_symmetric && !result.dimensions_analysis.symmetry.is_susy && (
              <Badge color="green" label="SYMMETRIC" />
            )}

            {/* v1.1 quantum badges */}
            {emBreak    && <Badge color="orange" label="⚡ EM BREAK" pulse />}
            {isUnstable && <Badge color="red"    label="⚠ UNSTABLE" pulse />}
            {luxury     && <Badge color="gold"   label="◆ LUXURY ×" />}
            {compress   && <Badge color="purple" label="▣ COMPRESS" />}
            {/* v1.2 object count badge */}
            {hasObjects && (
              <Badge
                color={selectedObjectId ? 'gold' : 'cyan'}
                label={selectedObjectId ? `OBJ ${selectedObjectId.toUpperCase()}` : `${result.detected_objects.length} OBJECTS`}
              />
            )}
          </div>
        )}
      </div>

      {/* ── Image area ── */}
      <div className="relative flex-1 flex items-center justify-center overflow-hidden">
        {imageUrl ? (
          <>
            {/* Source image */}
            <img
              ref={imgRef}
              src={imageUrl}
              alt="Target for dimensional analysis"
              className="max-h-full max-w-full object-contain select-none"
              draggable={false}
              onLoad={computeOverlayData}
              style={{ display: 'block' }}
            />

            {/* Vector grid overlay */}
            {scanComplete && (
              <div className="absolute inset-0 vector-grid pointer-events-none" />
            )}

            {/* Canvas: all marker, aura, and bounding-box layers.
                Pointer events enabled when objects are present so users can click boxes. */}
            {scanComplete && (
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 ${hasObjects ? 'cursor-crosshair' : 'pointer-events-none'}`}
                style={{ width: '100%', height: '100%' }}
                onClick={hasObjects ? handleCanvasClick : undefined}
              />
            )}

            {/* EM Break pulsing border overlay (v1.1) */}
            {scanComplete && emBreak && (
              <div
                className="absolute inset-0 pointer-events-none rounded-sm"
                style={{
                  boxShadow: 'inset 0 0 20px #ff6b1a60, inset 0 0 6px #ff6b1a40',
                  animation: 'scanPulse 1.4s ease-in-out infinite',
                }}
              />
            )}

            {/* Laser beam during scan */}
            {isScanning && <div className="laser-beam" />}

            {isScanning && (
              <div className="absolute inset-0 flex items-end justify-center pb-6 pointer-events-none">
                <span className="text-[#00ff88] text-xs tracking-[0.4em] animate-pulse uppercase">
                  Analysing Dimensional Vectors...
                </span>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center gap-4 text-[#2a2a4e]">
            <HexagonIcon />
            <p className="text-xs tracking-widest uppercase">Awaiting Input Signal</p>
            <p className="text-[10px] text-[#1a1a3e]">Upload an image to begin decoding</p>
          </div>
        )}
      </div>

      {/* ── Dimension bars (v1.0) + QV bar (v1.1) ── */}
      {scanComplete && result && (
        <div className="border-t border-[#1a1a3e] px-4 py-2 grid grid-cols-5 gap-3">
          <DimBar
            label="DIM 5"
            color={colorForFreq(
              // color_label (v1.3) carries the specific hue; fall back to dominant_frequency
              // for any cached responses that pre-date this field.
              result.dimensions_analysis.dim_5_color.color_label
              ?? result.dimensions_analysis.dim_5_color.dominant_frequency
            )}
            value={result.dimensions_analysis.dim_5_color.vector_value}
          />
          <DimBar label="DIM 6-8" color="#ff6b1a" value={result.dimensions_analysis.dim_6_8_rhythm.coherence_score} />
          <DimBar label="DIM 9"  color="#00d4ff"  value={result.dimensions_analysis.dim_9_geometry.edge_curvature_index} />
          <DimBar label="DIM 10" color="#a855f7"  value={Math.min(result.dimensions_analysis.dim_10_frequency.impulse_rate_hz / 20, 1)} />
          {/* v1.1: QV Multiplier bar — capped at 3.0 */}
          <DimBar
            label="QV ×"
            color={qfm?.luxury_particle_detected ? '#ffd700' : '#6366f1'}
            value={Math.min((qfm?.quantum_value_multiplier ?? 1) / 3, 1)}
            rawLabel={qfm ? `×${qfm.quantum_value_multiplier.toFixed(2)}` : null}
          />
        </div>
      )}
    </div>
  );
}

// ─── Canvas Drawing — v1.0 ────────────────────────────────────────────────────

/** Red crosshair at (x, y). */
function drawCrosshair(ctx, x, y, size, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 6;
  ctx.globalAlpha = 0.85;

  ctx.beginPath(); ctx.moveTo(x - size, y); ctx.lineTo(x + size, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y - size); ctx.lineTo(x, y + size); ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

/** Hollow ring at (x, y). */
function drawRing(ctx, x, y, radius, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 8;
  ctx.globalAlpha = 0.75;

  ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();

  ctx.fillStyle   = color;
  ctx.globalAlpha = 0.4;
  ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

// ─── Canvas Drawing — v1.1 ────────────────────────────────────────────────────

/**
 * Draws a radial gravitational-field aura at (x, y).
 * The aura is a multi-layer radial gradient mimicking spacetime curvature.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x        Canvas X coordinate (absolute pixels)
 * @param {number} y        Canvas Y coordinate (absolute pixels)
 * @param {number} intensity Dark-zone intensity 0-1 (controls radius and opacity)
 */
function drawGravitationalAura(ctx, x, y, intensity) {
  ctx.save();
  const radius = 30 + intensity * 55;  // 30-85 px based on density

  // Outer diffuse glow
  const outerGrad = ctx.createRadialGradient(x, y, 0, x, y, radius);
  outerGrad.addColorStop(0,   `rgba(80, 0, 160, ${0.22 * intensity})`);
  outerGrad.addColorStop(0.4, `rgba(40, 0, 80,  ${0.14 * intensity})`);
  outerGrad.addColorStop(1,   'rgba(0, 0, 0, 0)');
  ctx.fillStyle   = outerGrad;
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();

  // Inner bright core ring
  ctx.strokeStyle = `rgba(120, 40, 220, ${0.55 * intensity})`;
  ctx.lineWidth   = 1;
  ctx.shadowColor = '#7828dc';
  ctx.shadowBlur  = 10;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.28, 0, Math.PI * 2);
  ctx.stroke();

  // Second concentric ring
  ctx.strokeStyle = `rgba(80, 20, 140, ${0.35 * intensity})`;
  ctx.lineWidth   = 0.8;
  ctx.shadowBlur  = 4;
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.62, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

/**
 * Draws dashed cyan vector lines from each circle-cluster centre to its nearest
 * sharp marker, visualising the Stabilization-Atom ↔ Complement pairing.
 * Coordinates are expressed as ratios and mapped onto the letterbox-corrected image area.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {{ xRatio: number, yRatio: number }[]} circles  Smooth region centres
 * @param {{ xRatio: number, yRatio: number }[]} sharps   Angular region centres
 * @param {number} ox  Horizontal offset from canvas origin to image left edge
 * @param {number} oy  Vertical offset from canvas origin to image top edge
 * @param {number} iw  Rendered image width in canvas pixels
 * @param {number} ih  Rendered image height in canvas pixels
 */
function drawConnectionLines(ctx, circles, sharps, ox, oy, iw, ih) {
  if (circles.length === 0 || sharps.length === 0) return;

  ctx.save();
  ctx.strokeStyle = 'rgba(0, 212, 255, 0.45)';
  ctx.lineWidth   = 0.8;
  ctx.setLineDash([4, 6]);
  ctx.shadowColor = '#00d4ff';
  ctx.shadowBlur  = 5;
  ctx.globalAlpha = 0.6;

  circles.forEach(circ => {
    const cx = ox + circ.xRatio * iw;
    const cy = oy + circ.yRatio * ih;

    // Find the nearest sharp marker by Euclidean distance
    let nearestX = null, nearestY = null, bestDist = Infinity;
    sharps.forEach(sharp => {
      const sx = ox + sharp.xRatio * iw;
      const sy = oy + sharp.yRatio * ih;
      const dist = Math.hypot(cx - sx, cy - sy);
      if (dist < bestDist) { bestDist = dist; nearestX = sx; nearestY = sy; }
    });

    // Only draw if the complement is within a meaningful distance
    if (nearestX !== null && bestDist < iw * 0.45) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nearestX, nearestY);
      ctx.stroke();
    }
  });

  ctx.restore();
}

/**
 * Draws a pulsing double red ring around the image area to signal structural
 * instability (Stabilization Atom without complement).
 * Rings are inset from the image edges, not the canvas edges.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} ox  Horizontal offset from canvas origin to image left edge
 * @param {number} oy  Vertical offset from canvas origin to image top edge
 * @param {number} iw  Rendered image width in canvas pixels
 * @param {number} ih  Rendered image height in canvas pixels
 */
function drawInstabilityRing(ctx, ox, oy, iw, ih) {
  ctx.save();
  const inset = 6;

  // Outer ring
  ctx.strokeStyle = 'rgba(255, 45, 85, 0.60)';
  ctx.lineWidth   = 2.5;
  ctx.shadowColor = '#ff2d55';
  ctx.shadowBlur  = 14;
  ctx.globalAlpha = 0.85;
  ctx.strokeRect(ox + inset, oy + inset, iw - inset * 2, ih - inset * 2);

  // Inner ring — tighter, less opaque
  ctx.strokeStyle = 'rgba(255, 45, 85, 0.30)';
  ctx.lineWidth   = 1;
  ctx.shadowBlur  = 4;
  ctx.strokeRect(ox + inset + 5, oy + inset + 5, iw - (inset + 5) * 2, ih - (inset + 5) * 2);

  ctx.restore();
}

// ─── Canvas Drawing — v1.2 ────────────────────────────────────────────────────

/**
 * Draws a dashed bounding box around a detected object.
 * The selected object gets a solid gold highlight; others get a subtle cyan outline.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {{ bounding_box: number[], label: string }} obj
 * @param {number} ox  Letterbox X offset
 * @param {number} oy  Letterbox Y offset
 * @param {number} iw  Image width in canvas pixels
 * @param {number} ih  Image height in canvas pixels
 * @param {boolean} isSelected
 */
function drawBoundingBox(ctx, obj, ox, oy, iw, ih, isSelected) {
  const [bx1, by1, bx2, by2] = obj.bounding_box;
  const px  = ox + bx1 * iw;
  const py  = oy + by1 * ih;
  const pw  = (bx2 - bx1) * iw;
  const ph  = (by2 - by1) * ih;

  ctx.save();

  if (isSelected) {
    // Selected: solid gold box + stronger glow
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.90)';
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.shadowColor = '#ffd700';
    ctx.shadowBlur  = 18;
    ctx.globalAlpha = 0.95;
  } else {
    // Unselected: dashed cyan box
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.55)';
    ctx.lineWidth   = 1;
    ctx.setLineDash([5, 4]);
    ctx.shadowColor = '#00d4ff';
    ctx.shadowBlur  = 6;
    ctx.globalAlpha = 0.75;
  }

  ctx.strokeRect(px, py, pw, ph);

  // Corner accent marks (top-left, top-right, bottom-right, bottom-left)
  const cs = isSelected ? 10 : 7;
  const cc = isSelected ? 'rgba(255,215,0,0.9)' : 'rgba(0,212,255,0.8)';
  ctx.setLineDash([]);
  ctx.strokeStyle = cc;
  ctx.lineWidth   = isSelected ? 2.5 : 1.5;
  ctx.shadowBlur  = isSelected ? 10 : 4;
  [
    [px, py, cs, 0, cs, 0],           // top-left
    [px + pw, py, -cs, 0, -cs, 0],    // top-right
    [px + pw, py + ph, -cs, 0, -cs, 0], // bottom-right
    [px, py + ph, cs, 0, cs, 0],      // bottom-left
  ].forEach(([x, y, dx1, dy1, dx2, dy2]) => {
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx1, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + (dy1 === 0 ? (y === py ? cs : -cs) : 0)); ctx.stroke();
  });

  // Label text
  const labelColor = isSelected ? '#ffd700' : '#00d4ff';
  ctx.fillStyle   = isSelected ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.55)';
  ctx.shadowBlur  = 0;
  ctx.globalAlpha = 1;

  const labelText = obj.label.split('(')[0].trim();  // short form
  ctx.font = `${isSelected ? 'bold ' : ''}9px "JetBrains Mono", monospace`;
  const textW = ctx.measureText(labelText).width;
  ctx.fillRect(px + 3, py + 3, textW + 6, 14);
  ctx.fillStyle = labelColor;
  ctx.shadowColor = labelColor;
  ctx.shadowBlur  = isSelected ? 8 : 4;
  ctx.fillText(labelText, px + 6, py + 13);

  ctx.restore();
}

// ─── Canvas Drawing — v1.28 ───────────────────────────────────────────────────

/**
 * Draws the analysis focus-zone overlay for CARDS and INDIVIDUUM sectors.
 * - Outer margin zones get a dark crosshatch tint + "DAMPED" label.
 * - Inner core zone gets a bright dashed border + "CORE ELEMENT" label.
 *
 * Zone ratios mirror _get_focus_zone() in analyzer.py  (v1.3 values):
 *   CARDS:      mx=20%, my=22%   — wide enough to fully exclude the card frame
 *   INDIVIDUUM: mx=15%, my=5%,  max_y=85%
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} sectorMode
 * @param {number} ox  Letterbox X offset
 * @param {number} oy  Letterbox Y offset
 * @param {number} iw  Image width in canvas pixels
 * @param {number} ih  Image height in canvas pixels
 */
function drawFocusZone(ctx, sectorMode, ox, oy, iw, ih) {
  ctx.save();

  // Derive margin fractions
  let mxFrac, myFrac, maxYFrac;
  if (sectorMode === 'CARDS') {
    mxFrac = 0.20; myFrac = 0.22; maxYFrac = 1.0 - myFrac;
  } else {
    mxFrac = 0.15; myFrac = 0.05; maxYFrac = 0.85;
  }

  const mx  = mxFrac * iw;
  const my  = myFrac * ih;
  const cw  = iw - mx * 2;
  const ch  = (maxYFrac - myFrac) * ih;

  // Core zone inner-rect in canvas coords
  const cx1 = ox + mx;
  const cy1 = oy + my;

  // ── Dark crosshatch tint on the outer margin areas ──────────────────────────
  // We draw it as a semi-transparent dark rect over the whole image, then
  // punch out the core zone with 'destination-out' (transparent composite).
  ctx.globalAlpha = 0.32;
  ctx.fillStyle   = '#000010';
  // Top margin
  ctx.fillRect(ox, oy, iw, my);
  // Bottom margin
  const bMarginY = oy + maxYFrac * ih;
  ctx.fillRect(ox, bMarginY, iw, ih - maxYFrac * ih);
  // Left margin
  ctx.fillRect(ox, oy + my, mx, maxYFrac * ih - my);
  // Right margin
  ctx.fillRect(ox + iw - mx, oy + my, mx, maxYFrac * ih - my);

  // ── Crosshatch pattern over margin tint ─────────────────────────────────────
  ctx.globalAlpha = 0.10;
  ctx.strokeStyle = '#ff6b1a';
  ctx.lineWidth   = 0.5;
  const spacing   = 10;
  ctx.setLineDash([2, 8]);
  // Horizontal lines across margins only (approximate with full-image clip)
  for (let y = oy; y < oy + ih; y += spacing) {
    ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + iw, y); ctx.stroke();
  }

  // ── Bright dashed border on core zone ───────────────────────────────────────
  ctx.globalAlpha = 0.80;
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth   = 1.2;
  ctx.setLineDash([6, 4]);
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur  = 8;
  ctx.strokeRect(cx1, cy1, cw, ch);

  // Corner accent ticks
  ctx.setLineDash([]);
  ctx.lineWidth   = 2;
  ctx.shadowBlur  = 12;
  const tk = 8;
  [[cx1, cy1, tk, tk], [cx1 + cw, cy1, -tk, tk], [cx1 + cw, cy1 + ch, -tk, -tk], [cx1, cy1 + ch, tk, -tk]]
    .forEach(([bx, by, dx, dy]) => {
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + dx, by); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx, by + dy); ctx.stroke();
    });

  // ── "CORE ELEMENT" label on inner border ───────────────────────────────────
  ctx.shadowBlur  = 6;
  ctx.globalAlpha = 0.90;
  ctx.font        = 'bold 8px "JetBrains Mono", monospace';
  const coreLabel = '◈ CORE ELEMENT';
  ctx.fillStyle   = 'rgba(0,0,0,0.65)';
  ctx.fillRect(cx1 + 4, cy1 + 4, ctx.measureText(coreLabel).width + 8, 14);
  ctx.fillStyle   = '#00ff88';
  ctx.fillText(coreLabel, cx1 + 8, cy1 + 14);

  // ── "DAMPED" label on one of the margin areas ───────────────────────────────
  ctx.globalAlpha = 0.55;
  ctx.font        = '7px "JetBrains Mono", monospace';
  const dampLabel = 'DAMPED — HIGH FREQ AREA';
  // Place in top margin if tall enough, else left margin
  if (my >= 14) {
    const tw = ctx.measureText(dampLabel).width;
    ctx.fillStyle = 'rgba(0,0,0,0.50)';
    ctx.fillRect(ox + (iw - tw) / 2 - 4, oy + 3, tw + 8, 12);
    ctx.fillStyle = '#ff6b1a';
    ctx.fillText(dampLabel, ox + (iw - tw) / 2, oy + 12);
  } else if (mx >= 14) {
    ctx.save();
    ctx.translate(ox + mx - 4, oy + my + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#ff6b1a';
    ctx.fillText(dampLabel, -ctx.measureText(dampLabel).width / 2, 0);
    ctx.restore();
  }

  ctx.restore();
}

// ─── JSX Sub-components ───────────────────────────────────────────────────────

function Badge({ color, label, pulse = false }) {
  const colorMap = {
    orange: 'text-[#ff6b1a] border-[#ff6b1a]/40 bg-[#ff6b1a]/10',
    cyan:   'text-[#00d4ff] border-[#00d4ff]/40 bg-[#00d4ff]/10',
    green:  'text-[#00ff88] border-[#00ff88]/40 bg-[#00ff88]/10',
    red:    'text-[#ff2d55] border-[#ff2d55]/40 bg-[#ff2d55]/10',
    gold:   'text-[#ffd700] border-[#ffd700]/40 bg-[#ffd700]/10',
    purple: 'text-[#a855f7] border-[#a855f7]/40 bg-[#a855f7]/10',
  };
  return (
    <span
      className={`text-[9px] tracking-widest uppercase border px-2 py-0.5 rounded-sm
        ${colorMap[color] ?? colorMap.orange}
        ${pulse ? 'animate-pulse' : ''}
      `}
    >
      {label}
    </span>
  );
}

/** Dimension progress bar with optional raw label override (v1.1 QV bar). */
function DimBar({ label, color, value, rawLabel = null }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] text-[#4a5568] tracking-widest">{label}</span>
      <div className="h-1 bg-[#1a1a3e] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${Math.round(Math.max(0, Math.min(value, 1)) * 100)}%`,
            backgroundColor: color,
            boxShadow: `0 0 6px ${color}`,
          }}
        />
      </div>
      <span className="text-[9px]" style={{ color }}>
        {rawLabel ?? `${Math.round(value * 100)}%`}
      </span>
    </div>
  );
}

/**
 * Maps a specific color_label (or dominant_frequency fallback) to a CSS hex colour
 * for the DIM 5 progress bar and related UI elements.
 *
 * colour_label values (v1.3):
 *   Yellow → vivid yellow  (high-luminance warm signal)
 *   Orange → fire orange   (vivid warm / fire tones)
 *   Brown  → earth amber   (earth tones / tan / beige)
 *   Pink   → rose-pink     (pink-band signal)
 *   Green  → neon green
 *   Cyan   → interface cyan
 *   Blue   → cognitive blue
 *   Violet → prestige violet
 */
function colorForFreq(label) {
  const map = {
    Yellow: '#ffdd00',
    Orange: '#ff6b1a',
    Brown:  '#c47c3a',
    Pink:   '#ff69b4',
    Green:  '#00ff88',
    Cyan:   '#00d4ff',
    Blue:   '#3b82f6',
    Violet: '#a855f7',
  };
  return map[label] ?? '#ff6b1a';   // default: orange for unknown labels
}

function HexagonIcon() {
  return (
    <svg className="w-20 h-20 opacity-20" viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="1">
      <polygon points="50,5 93,27.5 93,72.5 50,95 7,72.5 7,27.5" />
      <polygon points="50,20 80,35 80,65 50,80 20,65 20,35" />
      <line x1="50" y1="5"  x2="50" y2="20" />
      <line x1="93" y1="27.5" x2="80" y2="35" />
      <line x1="93" y1="72.5" x2="80" y2="65" />
      <line x1="50" y1="95" x2="50" y2="80" />
      <line x1="7"  y1="72.5" x2="20" y2="65" />
      <line x1="7"  y1="27.5" x2="20" y2="35" />
    </svg>
  );
}
