// MuJoCo WASM out-parameter wrappers.
//
// The WASM bindings take out-parameters as `DoubleBuffer` -- a handle to memory
// inside the WASM heap -- not as JS typed arrays. Passing a Float64Array is
// accepted with no throw and no warning, and nothing is written: the array
// comes back exactly as it went in. A zeroed array is a plausible-looking
// result for every function here, so the failure is completely silent. An
// all-zero Jacobian, for instance, just means the arm never moves.
//
// Every call site goes through this module so that fact lives in one place.
//
// Buffers are pooled per (module instance, slot) because these run at the
// control rate and a fresh DoubleBuffer per call is a WASM malloc/free per
// tick. Each wrapper copies its result out before returning, so a slot is only
// live for the duration of one call and slots are never aliased across calls.

const pools = new WeakMap();

/** Scratch heap buffer of at least n doubles, unique to `key`. */
function slot(mj, key, n) {
  let byKey = pools.get(mj);
  if (!byKey) pools.set(mj, byKey = new Map());
  let entry = byKey.get(key);
  if (!entry || entry.n < n) {
    if (entry) entry.buf.delete();
    entry = { buf: new mj.DoubleBuffer(n), n };
    byKey.set(key, entry);
  }
  return entry.buf;
}

function copyOut(buf, out, n) {
  const view = buf.GetView();
  for (let i = 0; i < n; i++) out[i] = view[i];
  return out;
}

// Inputs are declared as number[] and are all short (<= 9 elements), so the
// copy is cheaper than finding out the hard way which array-likes Embind
// accepts.
const arr = (a) => (Array.isArray(a) ? a : Array.from(a));

/** @returns {Float64Array} out, = quaternion (wxyz) of 3x3 row-major `mat` */
export function mat2Quat(mj, out, mat) {
  const b = slot(mj, 'mat2Quat', 4);
  mj.mju_mat2Quat(b, arr(mat));
  return copyOut(b, out, 4);
}

/** @returns {Float64Array} out, = qa * qb */
export function mulQuat(mj, out, qa, qb) {
  const b = slot(mj, 'mulQuat', 4);
  mj.mju_mulQuat(b, arr(qa), arr(qb));
  return copyOut(b, out, 4);
}

/** @returns {Float64Array} out, = conjugate of q */
export function negQuat(mj, out, q) {
  const b = slot(mj, 'negQuat', 4);
  mj.mju_negQuat(b, arr(q));
  return copyOut(b, out, 4);
}

/** Rotation vector v such that qb * quat(v) = qa, in qb's frame.
 *  @returns {Float64Array} out (3) */
export function subQuat(mj, out, qa, qb) {
  const b = slot(mj, 'subQuat', 3);
  mj.mju_subQuat(b, arr(qa), arr(qb));
  return copyOut(b, out, 3);
}

/** Site Jacobian. Fills jacp/jacr as 3 x nv row-major. */
export function jacSite(mj, model, data, jacp, jacr, siteId) {
  const n = 3 * model.nv;
  const p = slot(mj, 'jacp', n);
  const r = slot(mj, 'jacr', n);
  mj.mj_jacSite(model, data, p, r, siteId);
  copyOut(p, jacp, n);
  copyOut(r, jacr, n);
}
