Each feature's probe module is validated by the installed node, which is how a
host detects the extension (not by a version number). Core-only modules need
none of them to be absent: the scalar backend uses only the first three, all of
which a default node accepts.

  $ ./wasm_features_gen.exe .
  $ cat > check.js <<'JS'
  > const fs = require("fs");
  > for (const f of ["bulk-memory", "nontrapping-float-to-int", "sign-extension", "simd128"]) {
  >   const bytes = fs.readFileSync(f + ".wasm");
  >   const ok = WebAssembly.validate(bytes);
  >   const broken = WebAssembly.validate(bytes.subarray(0, bytes.length - 1));
  >   console.log(f + ": validates " + ok + ", truncated validates " + broken);
  > }
  > JS
  $ node check.js
  bulk-memory: validates true, truncated validates false
  nontrapping-float-to-int: validates true, truncated validates false
  sign-extension: validates true, truncated validates false
  simd128: validates true, truncated validates false
