open Ssa_ir
open Machine_lower

(* The checked-in support matrix: one representative of every SSA constructor
   (and access shape, decode, encode, conversion and unary operator) with its
   expansion, slice, observable and classified failing conditions. *)

let v n ty = { Ssa_value.id = Ssa_id.Value.of_int n; ty }
let ix = v 0 (Ssa_type.Scalar Ssa_type.Index)
let f = v 1 (Ssa_type.Scalar Ssa_type.F64)
let i = v 2 (Ssa_type.Scalar Ssa_type.I64)
let p = v 3 (Ssa_type.Scalar Ssa_type.Pred)
let l = v 4 Ssa_type.Local
let buf = Ssa_id.Buffer.of_int 0

let coord =
  Ssa_access.Coord (Expr.Coord.make ~n:ix ~t:ix ~d:ix ~h:ix ~w:ix ~c:ix)

let flat = Ssa_access.Flat ix
let var = Expr.Builder.run Expr.Builder.fresh_local
let lanes = Ssa_type.Lanes.of_int 4
let steps = Expr.Coord.make ~n:0L ~t:0L ~d:0L ~h:0L ~w:1L ~c:0L
let at = Expr.Coord.make ~n:ix ~t:ix ~d:ix ~h:ix ~w:ix ~c:ix

let representatives =
  let decodes =
    Ssa_op.Decode.
      [
        Bf16_to_f64;
        Bool_to_f64;
        F16_to_f64;
        F32_to_f64;
        F64_to_f64;
        I16_dequant;
        I32_to_f64;
        I64;
        I64_to_f64;
        I8_dequant;
      ]
  in
  let encodes = Ssa_op.Encode.[ Bool_nonzero; F32_round; I64 ] in
  [
    Ssa_op.Check_access { buffer = buf; at = coord };
    Ssa_op.Check_access { buffer = buf; at = flat };
    Ssa_op.Check_gather { raw = i; extent = 4L };
    Ssa_op.Check_local { var; at = ix; extent = 4L };
    Ssa_op.Check_scan
      { var = Some var; row = ix; lane = ix; row_extent = 2L; lane_extent = 2L };
    Ssa_op.Const (Ssa_const.F64 1.);
  ]
  @ List.map
      (fun c -> Ssa_op.Convert (c, f))
      Ssa_op.Convert.
        [
          F32_to_f64;
          F64_to_f32;
          I64_to_f32;
          I64_to_f64;
          Index_to_f64;
          Index_to_i64;
        ]
  @ List.map
      (fun o -> Ssa_op.Float_binary (o, f, f))
      Expr.Value.[ Add; Div; Mul; Sub ]
  @ List.map (fun c -> Ssa_op.Float_compare (c, f, f)) Ssa_op.Compare.[ Eq; Lt ]
  @ [
      Ssa_op.Float_fma (f, f, f); Ssa_op.Float_max (f, f); Ssa_op.Float_to_i64 f;
    ]
  @ List.map
      (fun u -> Ssa_op.Float_unary (u, f))
      Expr.Value.[ Cos; Erf; Exp; Log; Sin; Sqrt; Trunc ]
  @ List.map
      (fun o -> Ssa_op.I64_arith (o, i, i))
      Ssa_op.I64_op.[ Add; Mul; Sub ]
  @ List.map (fun c -> Ssa_op.I64_compare (c, i, i)) Ssa_op.Compare.[ Eq; Lt ]
  @ [
      Ssa_op.I64_div (i, i);
      Ssa_op.Index_add (ix, ix);
      Ssa_op.Index_add_in_domain (ix, ix);
      Ssa_op.Index_ceil_div (2L, ix);
      Ssa_op.Index_clamp_low ix;
    ]
  @ List.map
      (fun c -> Ssa_op.Index_compare (c, ix, ix))
      Ssa_op.Compare.[ Eq; Lt ]
  @ [
      Ssa_op.Index_floor_div (2L, ix);
      Ssa_op.Index_max (ix, ix);
      Ssa_op.Index_min (ix, ix);
      Ssa_op.Index_of_i64 i;
      Ssa_op.Index_scale (3L, ix);
      Ssa_op.Index_scale_in_domain (3L, ix);
      Ssa_op.Lanewise (Ssa_op.Float_binary (Expr.Value.Add, f, f));
    ]
  @ List.concat_map
      (fun decode ->
        [
          Ssa_op.Load { buffer = buf; at = coord; decode };
          Ssa_op.Load { buffer = buf; at = flat; decode };
          Ssa_op.Load_in_bounds { buffer = buf; at = coord; decode };
        ])
      decodes
  @ [
      Ssa_op.Local_alloc { slots = 4L; var = Some var };
      Ssa_op.Local_read { local = l; at = ix };
      Ssa_op.Local_write { local = l; at = ix; value = f };
      Ssa_op.Mark Ssa_mark.Reduction;
      Ssa_op.Mark_lanes { mark = Ssa_mark.Reduction; lanes };
      Ssa_op.Meter_charge;
      Ssa_op.Meter_release 2L;
      Ssa_op.Meter_reserve 2L;
      Ssa_op.Meter_reset;
      Ssa_op.Pool_better (f, f);
      Ssa_op.Pred_not p;
      Ssa_op.Pred_or (p, p);
      Ssa_op.Select (p, f, f);
    ]
  @ List.map
      (fun encode ->
        Ssa_op.Store { buffer = buf; at = coord; encode; value = f })
      encodes
  @ [
      Ssa_op.Vec_extract { lane = Ssa_type.Lane.of_int 0; vector = f };
      Ssa_op.Vec_insert
        { lane = Ssa_type.Lane.of_int 0; vector = f; element = f };
      Ssa_op.Vec_iota { base = ix; step = 1L; lanes };
      Ssa_op.Vec_load
        { buffer = buf; at; steps; decode = Ssa_op.Decode.F32_to_f64; lanes };
      Ssa_op.Vec_splat { element = f; lanes };
      Ssa_op.Vec_store
        {
          buffer = buf;
          at;
          steps;
          encode = Ssa_op.Encode.F32_round;
          value = f;
          lanes;
        };
    ]

let%expect_test "every SSA operation name has a representative" =
  (* [Mir_census.op_row]'s match is exhaustive by construction; this counts
     the distinct operation names the matrix below shows *)
  let covered = List.sort_uniq compare (List.map Ssa_op.name representatives) in
  Fmt.pr "%d distinct operations, %d rows@." (List.length covered)
    (List.length representatives);
  [%expect {| 68 distinct operations, 99 rows |}]

let%expect_test "support matrix" =
  List.iter
    (fun op ->
      let r = Mir_census.op_row op in
      let disposition =
        match r.Mir_census.Row.disposition with
        | Mir_census.Disposition.Primitive -> "primitive"
        | Mir_census.Disposition.Helper h -> "helper:" ^ h
      in
      Fmt.pr "%-22s %-5s %-12s %-6s %s@." r.Mir_census.Row.op
        (Mir_census.Slice.name r.Mir_census.Row.slice)
        disposition
        (Mir_census.Observable.name r.Mir_census.Row.observable)
        r.Mir_census.Row.expansion;
      List.iter
        (fun (c : Mir_census.Condition.t) ->
          Fmt.pr "    %s -> %s@." c.Mir_census.Condition.test
            (match c.Mir_census.Condition.outcome with
            | Mir_census.Outcome.Defect -> "defect (invariant)"
            | Mir_census.Outcome.Failure k -> "failure " ^ k ^ " (block split)"))
        r.Mir_census.Row.conditions)
    representatives;
  [%expect
    {|
    check_access.coord     M3    primitive    -      per-axis sle/slt guard chain
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    check_access.flat      M3    primitive    -      nothing: no source failure
        flat offset outside the buffer -> defect (invariant)
    check_gather           M4.1  primitive    -      slt -extent; sle extent guards
        raw outside [-extent, extent) -> failure gather_index_out_of_range (block split)
    check_local            M4.3  primitive    -      slt/sle guard
        position outside [0, extent) -> failure unbound_local (block split)
    check_scan             M4.3  primitive    -      row guard, then lane guard
        row outside its extent (wins) -> failure scan_projection (block split)
        lane outside its extent -> failure scan_projection (block split)
    const                  M3    primitive    -      const (exact bits)
    convert.f32_to_f64     M3    primitive    -      fext.f32.f64
    convert.f64_to_f32     M3    primitive    -      fround.f64.f32
    convert.i64_to_f32     M4.1  primitive    -      scvt.i64.f32 (one rounding)
    convert.i64_to_f64     M4.1  primitive    -      scvt.i64.f64
    convert.index_to_f64   M3    primitive    -      sext.i64; scvt.i64.f64
    convert.index_to_i64   M3    primitive    -      sext.i64
    float.add              M3    primitive    -      fadd at the operand precision
    float.div              M3    primitive    -      fdiv at the operand precision
    float.mul              M3    primitive    -      fmul at the operand precision
    float.sub              M3    primitive    -      fsub at the operand precision
    float.compare.eq       M3    primitive    -      fcmp.oeq
    float.compare.lt       M3    primitive    -      fcmp.olt
    float.fma              M4.1  primitive    -      ffma (one rounding; the planning summary must permit contraction)
    float.max              M4.1  primitive    -      fmax (IEEE maximum: NaN propagates, +0 > -0)
    float.to_i64           M4.1  primitive    -      uno guard; |x| = inf guard; range guard; fcvt.f64.i64
        NaN -> failure i64_from_float_nan (block split)
        an infinity -> failure i64_from_float_infinite (block split)
        outside [-2^63, 2^63) -> failure i64_from_float_out_of_range (block split)
    float.cos              M4.4  helper:cos   -      call libm (binary32: fext; call; fround)
    float.erf              M4.4  helper:exp   -      owned: abs; A-S polynomial; call exp (binary32 steps at binary32)
    float.exp              M4.4  helper:exp   -      call libm (binary32: fext; call; fround)
    float.log              M4.4  helper:log   -      call libm (binary32: fext; call; fround)
    float.sin              M4.4  helper:sin   -      call libm (binary32: fext; call; fround)
    float.sqrt             M3    primitive    -      fsqrt (correctly rounded)
    float.trunc            M3    primitive    -      ftrunc
    i64.add                M3    primitive    -      add.i64 (modular)
    i64.mul                M3    primitive    -      mul.i64 (modular)
    i64.sub                M3    primitive    -      sub.i64 (modular)
    i64.compare.eq         M3    primitive    -      icmp.eq
    i64.compare.lt         M3    primitive    -      icmp.slt
    i64.div                M4.1  primitive    -      zero guard; min/-1 guard; sdiv
        divisor zero (first) -> failure i64_division_by_zero (block split)
        min_int / -1 -> failure i64_division_overflow (block split)
    index.add              M3    primitive    -      sext.i64 x2; add; domain guard; trunc.i32
        sum outside the index domain -> failure index_overflow (block split)
    index.add_in_domain    M3    primitive    -      sext.i64 x2; add; narrow.i32
        sum outside the index domain (proof) -> defect (invariant)
    index.ceil_div         M4.1  primitive    -      sdiv/srem by the positive literal; adjust
    index.clamp_low        M3    primitive    -      icmp.slt 0; select
    index.compare.eq       M3    primitive    -      icmp.eq
    index.compare.lt       M3    primitive    -      icmp.slt
    index.floor_div        M4.1  primitive    -      sdiv/srem by the positive literal; adjust
    index.max              M3    primitive    -      icmp.slt; select
    index.min              M3    primitive    -      icmp.slt; select
    index.of_i64           M4.1  primitive    -      narrow.i32
        int64 outside the index domain -> defect (invariant)
    index.scale            M3    primitive    -      sext.i64; mul; domain guard; trunc.i32
        product outside the index domain -> failure index_overflow (block split)
    index.scale_in_domain  M3    primitive    -      sext.i64; mul; narrow.i32
        product outside the index domain (proof) -> defect (invariant)
    lanes.float.add        M11   primitive    -      lane-wise generic vector operation
    load.coord             M4.2  primitive    reads  axis guards; row-major byte offset; load.i16; shl 16; bitcast.f32; fext
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.2  primitive    reads  byte offset; load.i16; shl 16; bitcast.f32; fext
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.2  primitive    reads  byte offset; load.i16; shl 16; bitcast.f32; fext
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.2  primitive    reads  axis guards; row-major byte offset; load.i8; icmp.ne 0; select 1.0 0.0
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.2  primitive    reads  byte offset; load.i8; icmp.ne 0; select 1.0 0.0
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.2  primitive    reads  byte offset; load.i8; icmp.ne 0; select 1.0 0.0
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.2  primitive    reads  axis guards; row-major byte offset; load.i16; binary16 decode by primitives
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.2  primitive    reads  byte offset; load.i16; binary16 decode by primitives
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.2  primitive    reads  byte offset; load.i16; binary16 decode by primitives
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M3    primitive    reads  axis guards; row-major byte offset; load.i32; bitcast.f32; fext.f32.f64
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M3    primitive    reads  byte offset; load.i32; bitcast.f32; fext.f32.f64
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M3    primitive    reads  byte offset; load.i32; bitcast.f32; fext.f32.f64
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M3    primitive    reads  axis guards; row-major byte offset; load.i64; bitcast.f64
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M3    primitive    reads  byte offset; load.i64; bitcast.f64
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M3    primitive    reads  byte offset; load.i64; bitcast.f64
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.2  primitive    reads  axis guards; row-major byte offset; load.i16; sext; sub zero_point; scvt; fmul scale (channel from C)
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.2  primitive    reads  byte offset; load.i16; sext; sub zero_point; scvt; fmul scale (channel from C)
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.2  primitive    reads  byte offset; load.i16; sext; sub zero_point; scvt; fmul scale (channel from C)
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.2  primitive    reads  axis guards; row-major byte offset; load.i32; sext; scvt.i64.f64
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.2  primitive    reads  byte offset; load.i32; sext; scvt.i64.f64
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.2  primitive    reads  byte offset; load.i32; sext; scvt.i64.f64
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.1  primitive    reads  axis guards; row-major byte offset; load.i64
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.1  primitive    reads  byte offset; load.i64
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.1  primitive    reads  byte offset; load.i64
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.1  primitive    reads  axis guards; row-major byte offset; load.i64; scvt.i64.f64
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.1  primitive    reads  byte offset; load.i64; scvt.i64.f64
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.1  primitive    reads  byte offset; load.i64; scvt.i64.f64
        coordinate outside (proof; byte range checked) -> defect (invariant)
    load.coord             M4.2  primitive    reads  axis guards; row-major byte offset; load.i8; sext; sub zero_point; scvt; fmul scale (channel from C)
        coordinate outside its axis, first axis in N T D H W C order -> failure coord_out_of_range (block split)
    load.flat              M4.2  primitive    reads  byte offset; load.i8; sext; sub zero_point; scvt; fmul scale (channel from C)
        flat offset outside the buffer -> defect (invariant)
    load.in_bounds         M4.2  primitive    reads  byte offset; load.i8; sext; sub zero_point; scvt; fmul scale (channel from C)
        coordinate outside (proof; byte range checked) -> defect (invariant)
    local.alloc            M4.3  primitive    local  per-site scratch region; undef; addr
    local.read             M4.3  primitive    local  bounds guard when named; load.i64; bitcast
        outside a named local's object -> failure unbound_local (block split)
        outside an anonymous local's object -> defect (invariant)
        a cell never written -> defect (invariant)
    local.write            M4.3  primitive    local  bitcast; store.i64
        outside its object -> defect (invariant)
    mark                   M3    primitive    event  event x1
    mark_lanes             M3    primitive    event  event x lanes
    meter.charge           M4.3  primitive    meter  load remaining; slt 0 guard; store remaining-1
        no update left (before the body) -> failure scan_meter (block split)
    meter.release          M4.3  primitive    meter  load live; sub; store
    meter.reserve          M4.3  primitive    meter  load live; add; sle limit guard; store
        live state over the peak -> failure scan_meter (block split)
    meter.reset            M4.3  primitive    meter  store limit; store 0
    pool_better            M4.1  primitive    -      fcmp.olt best value; fcmp.uno value; por
    pred.not               M3    primitive    -      pnot
    pred.or                M3    primitive    -      por (both computed)
    select                 M3    primitive    -      select (both computed)
    store                  M4.2  primitive    writes byte offset; fcmp.oeq 0.0; select 0 1; store.i8
        a store outside its buffer -> defect (invariant)
    store                  M3    primitive    writes byte offset; fround.f64.f32; bitcast.i32; store.i32
        a store outside its buffer -> defect (invariant)
    store                  M3    primitive    writes byte offset; store.i64
        a store outside its buffer -> defect (invariant)
    vec.extract            M11   primitive    -      lane extract
    vec.insert             M11   primitive    -      lane insert
    vec.iota               M11   primitive    -      lane constants + splat add
    vec.load               M11   primitive    reads  contiguous/broadcast/strided lanes
        a lane outside the buffer (proof) -> defect (invariant)
    vec.splat              M11   primitive    -      splat
    vec.store              M11   primitive    writes lanes in order
        a lane outside the buffer (proof) -> defect (invariant) |}]

let%expect_test "types" =
  List.iter
    (fun ty ->
      Fmt.pr "%a -> %s@." Ssa_type.pp ty
        (match Mir_census.machine_type ty with
        | Ok t -> Fmt.str "%a" Machine_ir.Mir_type.pp t
        | Error s -> "refused until " ^ Mir_census.Slice.name s))
    Ssa_type.
      [
        Effect;
        Local;
        Mask lanes;
        Scalar F32;
        Scalar F64;
        Scalar I64;
        Scalar Index;
        Scalar Pred;
        Vec (F32, lanes);
      ];
  [%expect
    {|
    effect -> order
    local -> ptr64
    mask<x4> -> mask<x4>
    f32 -> f32
    f64 -> f64
    i64 -> i64
    index -> i32
    pred -> pred
    vec<x4,f32> -> vec<x4,f32> |}]
