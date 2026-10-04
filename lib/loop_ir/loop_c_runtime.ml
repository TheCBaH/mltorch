(* The C text a generated translation unit shares: the ABI declarations (emitted
   into every file from this one definition) and the helper functions a kernel
   names. A helper is emitted only if a kernel used it, in [Name] order. Each
   helper's semantics is the reference's ([Expr.Value], [Expr.Max_op], [Half]);
   the JavaScript counterparts are in [Loop_js_runtime]. *)

module Name = struct
  type t =
    | Bf16_to_float
    | Coord_failure
    | Erf
    | Erf_f32
    | F16_to_float
    | F32_prelude
    | Float_max
    | Float_max_f32
    | Floor_div
    | I64_div
    | I64_from_float
    | I64_from_float_failure
    | Idx_clamp_low
    | Idx_max
    | Idx_min
    | Vector_prelude
    | Vector_prelude_f32
    | Vector_prelude_fma_f32

  let all =
    [
      Bf16_to_float;
      Coord_failure;
      Erf;
      Erf_f32;
      F16_to_float;
      F32_prelude;
      Float_max;
      Float_max_f32;
      Floor_div;
      I64_div;
      I64_from_float;
      I64_from_float_failure;
      Idx_clamp_low;
      Idx_max;
      Idx_min;
      Vector_prelude;
      Vector_prelude_f32;
      Vector_prelude_fma_f32;
    ]

  let to_string = function
    | Bf16_to_float -> "bf16_to_float"
    | Coord_failure -> "coord_failure"
    | Erf -> "erf_approx"
    | Erf_f32 -> "erf_approx_f32"
    | F16_to_float -> "f16_to_float"
    | F32_prelude -> "f32_prelude"
    | Float_max -> "float_max"
    | Float_max_f32 -> "float_max_f32"
    | Floor_div -> "floor_div"
    | I64_div -> "i64_div"
    | I64_from_float -> "i64_from_float"
    | I64_from_float_failure -> "i64_from_float_failure"
    | Idx_clamp_low -> "idx_clamp_low"
    | Idx_max -> "idx_max"
    | Idx_min -> "idx_min"
    | Vector_prelude -> "vector_prelude"
    | Vector_prelude_f32 -> "vector_prelude_f32"
    | Vector_prelude_fma_f32 -> "vector_prelude_fma_f32"
end

(* The failure record: [kind] is the position of the failure's kind in
   [Loop_js_failure.Kind.all] (closed and alphabetical, so the number is a
   stable ABI), and [v] holds the kind's fields in [Loop_js_failure.fields]
   order, a [Coord] field taking six slots. [invocation] is filled by the
   schedule, never by a kernel: a shared kernel does not know which call it is. *)
let error_words = 12

let kind_index k =
  let rec go i = function
    | [] -> invalid_arg "Loop_c_runtime.kind_index"
    | k' :: rest -> if k' = k then i else go (i + 1) rest
  in
  go 0 Loop_js_failure.Kind.all

let prelude_in dialect =
  let error_record words =
    [
      "struct model_error {";
      "  int32_t kind;";
      "  int32_t invocation;";
      "  int64_t v[" ^ words ^ "];";
      "};";
      "";
    ]
  in
  let fail_set =
    [
      "static inline int fail_set(struct model_error *e, int32_t kind) {";
      "  e->kind = kind;";
      "  e->invocation = -1;";
      "  memset(e->v, 0, sizeof e->v);";
      "  return 1;";
      "}";
      "";
    ]
  in
  String.concat "\n"
    (match dialect with
    | Loop_c_dialect.Gnu ->
        [
          "#include <math.h>";
          "#include <stddef.h>";
          "#include <stdint.h>";
          "#include <string.h>";
          "";
          Printf.sprintf "#define MODEL_ERROR_WORDS %d" error_words;
        ]
        @ error_record "MODEL_ERROR_WORDS"
        @ fail_set
    | Loop_c_dialect.Compcert_scalar ->
        [
          "typedef signed int int32_t;";
          "typedef signed long int64_t;";
          "typedef unsigned char uint8_t;";
          "typedef unsigned short uint16_t;";
          "typedef unsigned int uint32_t;";
          "typedef unsigned long uint64_t;";
          "typedef unsigned long size_t;";
          "extern void *memcpy(void *, const void *, size_t);";
          "extern void *memset(void *, int, size_t);";
          "extern double cos(double);";
          "extern double exp(double);";
          "extern double fabs(double);";
          "extern float fabsf(float);";
          "extern double fma(double, double, double);";
          "extern float fmaf(float, float, float);";
          "extern double ldexp(double, int);";
          "extern double log(double);";
          "extern double sin(double);";
          "extern double sqrt(double);";
          "extern float sqrtf(float);";
          "extern double trunc(double);";
          "extern float truncf(float);";
          "";
        ]
        @ error_record (string_of_int error_words)
        @ fail_set)

(* The 128-bit-vector idiom of the vectorized C: four binary64 lanes as a GCC/
   Clang generic vector, which each target lowers to its own registers (two
   NEON or SSE2 registers, one AVX register). Loads and stores go through
   [memcpy], so alignment and aliasing are never assumed. A lane is its scalar
   iteration: the arithmetic is the C operator per lane (the build disables FP
   contraction), a narrowing to binary32 is the vector conversion (round to
   nearest even), and the operations C has no vector form for (the maximum with
   its NaN and signed-zero rule, a square root, a transcendental) run lane by
   lane through the same scalar function. *)
let vector_prelude =
  String.concat "\n"
    [
      "typedef double v4df __attribute__((vector_size(32)));";
      "typedef float v4sf __attribute__((vector_size(16)));";
      "typedef int64_t v4di __attribute__((vector_size(32)));";
      "typedef int32_t v4si __attribute__((vector_size(16)));";
      "static inline v4df vf_splat(double x) { return (v4df){x, x, x, x}; }";
      "static inline v4df vf_load_f32(const float *p) { v4sf s; memcpy(&s, p, \
       sizeof s); return __builtin_convertvector(s, v4df); }";
      "static inline v4df vf_load_f64(const double *p) { v4df v; memcpy(&v, p, \
       sizeof v); return v; }";
      "static inline v4df vf_load_i32(const int32_t *p) { v4si s; memcpy(&s, \
       p, sizeof s); return __builtin_convertvector(s, v4df); }";
      "static inline void vf_store_f32(float *p, v4df v) { v4sf s = \
       __builtin_convertvector(v, v4sf); memcpy(p, &s, sizeof s); }";
      "static inline v4df vf_round_f32(v4df v) { return \
       __builtin_convertvector(__builtin_convertvector(v, v4sf), v4df); }";
      "static inline v4df vf_sel(v4di m, v4df a, v4df b) { return \
       (v4df)(((v4di)a & m) | ((v4di)b & ~m)); }";
      "static inline v4df vf_max(v4df a, v4df b) { v4df r; for (int k = 0; k < \
       4; k++) r[k] = float_max(a[k], b[k]); return r; }";
      "static inline v4df vf_sqrt(v4df a) { v4df r; for (int k = 0; k < 4; \
       k++) r[k] = sqrt(a[k]); return r; }";
      "static inline v4df vf_trunc(v4df a) { v4df r; for (int k = 0; k < 4; \
       k++) r[k] = trunc(a[k]); return r; }";
      "static inline v4df vf_cos(v4df a) { v4df r; for (int k = 0; k < 4; k++) \
       r[k] = cos(a[k]); return r; }";
      "static inline v4df vf_exp(v4df a) { v4df r; for (int k = 0; k < 4; k++) \
       r[k] = exp(a[k]); return r; }";
      "static inline v4df vf_log(v4df a) { v4df r; for (int k = 0; k < 4; k++) \
       r[k] = log(a[k]); return r; }";
      "static inline v4df vf_sin(v4df a) { v4df r; for (int k = 0; k < 4; k++) \
       r[k] = sin(a[k]); return r; }";
      "static inline v4df vf_erf(v4df a) { v4df r; for (int k = 0; k < 4; k++) \
       r[k] = erf_approx(a[k]); return r; }";
      "";
    ]

let f32_lanes = 16

(* The binary32 vector idiom: [f32_lanes] floats as a generic vector, which each
   target lowers to its registers (two NEON or SSE2 registers, one AVX). Every
   lane is its scalar iteration in binary32: the arithmetic is the C operator
   per lane with contraction off, and what C has no vector form for (the maximum
   with its NaN and signed-zero rule, a square root, a transcendental) runs lane
   by lane through the same scalar function the scalar kernel uses. *)
let vector_prelude_f32 =
  let n = f32_lanes in
  let lane_fn name body =
    Printf.sprintf
      "static inline v%dsf vs_%s(v%dsf a) { v%dsf r; for (int k = 0; k < %d; \
       k++) r[k] = %s; return r; }"
      n name n n n body
  in
  String.concat "\n"
    [
      Printf.sprintf "typedef float v%dsf __attribute__((vector_size(%d)));" n
        (4 * n);
      Printf.sprintf "typedef int32_t v%dsi __attribute__((vector_size(%d)));" n
        (4 * n);
      Printf.sprintf
        "static inline __attribute__((always_inline)) v%dsf vs_splat(float x) \
         { return (v%dsf){%s}; }"
        n n
        (String.concat ", " (List.init n (fun _ -> "x")));
      Printf.sprintf
        "static inline __attribute__((always_inline)) v%dsf vs_load(const \
         float *p) { v%dsf v; memcpy(&v, p, sizeof v); return v; }"
        n n;
      Printf.sprintf
        "static inline __attribute__((always_inline)) void vs_store(float *p, \
         v%dsf v) { memcpy(p, &v, sizeof v); }"
        n;
      Printf.sprintf
        "static inline __attribute__((always_inline)) v%dsf vs_sel(v%dsi m, \
         v%dsf a, v%dsf b) { return (v%dsf)(((v%dsi)a & m) | ((v%dsi)b & ~m)); \
         }"
        n n n n n n n;
      Printf.sprintf
        "static inline v%dsf vs_max(v%dsf a, v%dsf b) { v%dsf r; for (int k = \
         0; k < %d; k++) r[k] = float_max_f32(a[k], b[k]); return r; }"
        n n n n n;
      lane_fn "sqrt" "sqrtf(a[k])";
      lane_fn "trunc" "truncf(a[k])";
      lane_fn "cos" "(float)cos((double)a[k])";
      lane_fn "exp" "(float)exp((double)a[k])";
      lane_fn "log" "(float)log((double)a[k])";
      lane_fn "sin" "(float)sin((double)a[k])";
      lane_fn "erf" "erf_approx_f32(a[k])";
      "";
    ]

let lit = Loop_numerics.f32_literal

let text dialect : Name.t -> string =
  let nan = Loop_c_dialect.nan dialect in
  function
  | Name.Bf16_to_float ->
      "static inline double bf16_to_float(uint16_t h) {\n\
      \  uint32_t b = (uint32_t)h << 16;\n\
      \  float f;\n\
      \  memcpy(&f, &b, sizeof f);\n\
      \  return (double)f;\n\
       }\n"
  | Name.Coord_failure ->
      Printf.sprintf
        "static inline int coord_failure(struct model_error *e, int64_t \
         buffer, const int64_t *extents, const int64_t *coord) {\n\
        \  for (int axis = 0; axis < 6; axis++) {\n\
        \    if (coord[axis] < 0 || coord[axis] >= extents[axis]) {\n\
        \      fail_set(e, %d);\n\
        \      e->v[0] = buffer;\n\
        \      e->v[1] = axis;\n\
        \      e->v[2] = coord[axis];\n\
        \      for (int k = 0; k < 6; k++) e->v[3 + k] = coord[k];\n\
        \      return 1;\n\
        \    }\n\
        \  }\n\
        \  return fail_set(e, %d);\n\
         }\n"
        (kind_index Loop_js_failure.Kind.Coord_out_of_range)
        (kind_index Loop_js_failure.Kind.Defect)
  | Name.Erf ->
      "static inline double erf_approx(double x) {\n\
      \  const double p = 0.3275911;\n\
      \  const double a1 = 0.254829592;\n\
      \  const double a2 = -0.284496736;\n\
      \  const double a3 = 1.421413741;\n\
      \  const double a4 = -1.453152027;\n\
      \  const double a5 = 1.061405429;\n\
      \  const double sign = x < 0.0 ? -1.0 : 1.0;\n\
      \  const double ax = fabs(x);\n\
      \  const double t = 1.0 / (1.0 + p * ax);\n\
      \  const double poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));\n\
      \  return sign * (1.0 - poly * exp(-ax * ax));\n\
       }\n"
  | Name.Erf_f32 ->
      (* [Loop_numerics.erf32] transcribed: one rounding per operation (every
         statement is one float operation, contraction off), and [exp] of the
         rounded argument rounded once. *)
      Printf.sprintf
        "static inline float erf_approx_f32(float x) {\n\
        \  const float p = %s;\n\
        \  const float a1 = %s;\n\
        \  const float a2 = %s;\n\
        \  const float a3 = %s;\n\
        \  const float a4 = %s;\n\
        \  const float a5 = %s;\n\
        \  const float sign = x < 0.0f ? -1.0f : 1.0f;\n\
        \  const float ax = fabsf(x);\n\
        \  const float pa = p * ax;\n\
        \  const float d = 1.0f + pa;\n\
        \  const float t = 1.0f / d;\n\
        \  float q = t * a5;\n\
        \  q = a4 + q; q = t * q;\n\
        \  q = a3 + q; q = t * q;\n\
        \  q = a2 + q; q = t * q;\n\
        \  q = a1 + q; q = t * q;\n\
        \  const float sq = ax * ax;\n\
        \  const float e = (float)exp((double)(-sq));\n\
        \  const float pe = q * e;\n\
        \  const float one = 1.0f - pe;\n\
        \  return sign * one;\n\
         }\n"
        (lit 0.3275911) (lit 0.254829592) (lit (-0.284496736)) (lit 1.421413741)
        (lit (-1.453152027)) (lit 1.061405429)
  | Name.F16_to_float ->
      Printf.sprintf
        "static inline double f16_to_float(uint16_t h) {\n\
        \  const uint32_t sign = (h >> 15) & 1u;\n\
        \  const uint32_t exp16 = (h >> 10) & 0x1fu;\n\
        \  const uint32_t mant = h & 0x3ffu;\n\
        \  double m;\n\
        \  if (exp16 == 0) m = (double)mant * 0x1p-24;\n\
        \  else if (exp16 == 0x1f) m = mant == 0 ? %s : %s;\n\
        \  else m = (double)(mant | 0x400u) * ldexp(1.0, (int)exp16 - 25);\n\
        \  return sign == 1 ? -m : m;\n\
         }\n"
        (Loop_c_dialect.infinity dialect)
        nan
  | Name.F32_prelude -> (
      (* Every float expression of an fp32 kernel is evaluated in binary32:
         refuse a host that widens (x87), and never let a literal promote.
         CompCert evaluates in the type's own precision. *)
      match dialect with
      | Loop_c_dialect.Gnu ->
          "#include <float.h>\n\
           _Static_assert(FLT_EVAL_METHOD == 0, \"binary32 expressions must \
           not widen\");\n"
      | Loop_c_dialect.Compcert_scalar -> "")
  | Name.Float_max_f32 ->
      Printf.sprintf
        "static inline float float_max_f32(float a, float b) {\n\
        \  if (a != a || b != b) return %s;\n\
        \  if (a == 0.0f && b == 0.0f) return %s ? b : a;\n\
        \  return a > b ? a : b;\n\
         }\n"
        nan
        (Loop_c_dialect.signbit dialect "a")
  | Name.Float_max ->
      Printf.sprintf
        "static inline double float_max(double a, double b) {\n\
        \  if (a != a || b != b) return %s;\n\
        \  if (a == 0.0 && b == 0.0) return %s ? b : a;\n\
        \  return a > b ? a : b;\n\
         }\n"
        nan
        (Loop_c_dialect.signbit dialect "a")
  | Name.Floor_div ->
      "static inline int64_t floor_div(int64_t n, int64_t d) {\n\
      \  int64_t q = n / d;\n\
      \  return n % d < 0 ? q - 1 : q;\n\
       }\n"
  | Name.I64_div ->
      (* Unreachable with a zero or overflowing divisor: a [Fail_if] precedes
         every division. Total anyway, so the helper has no undefined case. *)
      "static inline int64_t i64_div(int64_t a, int64_t b) {\n\
      \  if (b == 0) return 0;\n\
      \  if (b == -1) return (int64_t)(0u - (uint64_t)a);\n\
      \  return a / b;\n\
       }\n"
  | Name.I64_from_float ->
      (* Total for the same reason: the guard rejects NaN and out-of-range. *)
      "static inline int64_t i64_from_float(double x) {\n\
      \  if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0)) \
       return 0;\n\
      \  return (int64_t)x;\n\
       }\n"
  | Name.I64_from_float_failure ->
      Printf.sprintf
        "static inline int i64_from_float_failure(struct model_error *e, \
         double x) {\n\
        \  if (x != x) return fail_set(e, %d);\n\
        \  if (!%s) return fail_set(e, %d);\n\
        \  fail_set(e, %d);\n\
        \  memcpy(&e->v[0], &x, sizeof x);\n\
        \  return 1;\n\
         }\n"
        (kind_index Loop_js_failure.Kind.I64_from_float_nan)
        (Loop_c_dialect.isfinite dialect "x")
        (kind_index Loop_js_failure.Kind.I64_from_float_infinite)
        (kind_index Loop_js_failure.Kind.I64_from_float_out_of_range)
  | Name.Idx_clamp_low ->
      "static inline int64_t idx_clamp_low(int64_t a) { return a < 0 ? 0 : a; }\n"
  | Name.Idx_max ->
      "static inline int64_t idx_max(int64_t a, int64_t b) { return a > b ? a \
       : b; }\n"
  | Name.Idx_min ->
      "static inline int64_t idx_min(int64_t a, int64_t b) { return a < b ? a \
       : b; }\n"
  | Name.Vector_prelude -> vector_prelude
  | Name.Vector_prelude_f32 -> vector_prelude_f32
  | Name.Vector_prelude_fma_f32 ->
      (* A fused multiply-add per lane through the host's [fmaf], one rounding,
         written as straight-line lane statements: the compiler's SLP
         vectorizer turns the sixteen of them into four [fmla] with the vector
         in registers (a rolled lane loop, or a union of NEON quarters, left it
         round-tripping through the stack: measured). Where it cannot, it is
         sixteen scalar [fmaf], the same bits. [Loop_numerics.fma32] is checked
         against [fmaf] bit for bit. No contraction flag is needed or wanted. *)
      let n = f32_lanes in
      Printf.sprintf
        "static inline __attribute__((always_inline)) v%dsf vs_fma(v%dsf a, \
         v%dsf b, v%dsf c) {\n\
        \  v%dsf r;\n\
         %s\n\
        \  return r;\n\
         }\n"
        n n n n n
        (String.concat "\n"
           (List.init n (fun k ->
                Printf.sprintf "  r[%d] = fmaf(a[%d], b[%d], c[%d]);" k k k k)))

let prelude = prelude_in Loop_c_dialect.Gnu

let helpers ?(dialect = Loop_c_dialect.Gnu) names =
  String.concat "\n"
    (List.filter_map
       (fun n -> if List.mem n names then Some (text dialect n) else None)
       Name.all)
