open Ssa_ir
module B = Ssa_builder

(* M4.4: the named math functions through the generic route against the SSA
   interpreter. [cos], [exp], [log] and [sin] are calls to helpers modelled by
   the host's libm, the oracle's own primitive; [erf] is owned, expanded into
   primitives and an [exp] call. A binary64 result is stored as three binary32
   cells — its rounding, then the rounding of each remainder — which keeps all
   53 bits for the magnitudes these inputs reach, so a one-ulp difference is
   visible. The binary32 programs are the SSA precision rewrite of the same
   programs, so the oracle is [Ssa_numerics.erf32] and the once-rounded libm
   results. *)

let buffer id ~w format role =
  {
    Ssa_buffer.id = Ssa_id.Buffer.of_int id;
    extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w ~c:1L;
    format;
    role;
  }

let inputs =
  [|
    0.;
    -0.;
    1.;
    -1.;
    0.5;
    -0.75;
    3.;
    1e-300;
    5e-324;
    1e300;
    709.;
    710.;
    -745.;
    Float.pi;
    1e22;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
    2.5e-8;
    -6.;
  |]

let samples =
  let st = Random.State.make [| 4242 |] in
  Array.init 64 (fun _ ->
      let m = Random.State.float st 2. -. 1. in
      let e = Random.State.int st 12 - 6 in
      Float.ldexp m e)

(* out[3k .. 3k+2] <- the three-cell split of [f in[k]] *)
let program u xs =
  let n = Int64.of_int (Array.length xs) in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program
       ~buffers:
         [
           buffer 0 ~w:n Ssa_format.F64 Ssa_buffer.Input;
           buffer 1 ~w:(Int64.mul 3L n) Ssa_format.F32 Ssa_buffer.Output;
         ]
       (fun bld ->
         let B.Nil =
           B.for_ bld ~lo:(B.index bld 0L) ~hi:(B.index bld n) ~init:B.Nil
             (fun bld k B.Nil ->
               let x =
                 B.load_f64 bld (Ssa_id.Buffer.of_int 0)
                   ~decode:Ssa_op.Decode.F64_to_f64 (B.Flat k)
               in
               let d = B.f64_unary bld u x in
               let rest d =
                 B.f64_binary bld Expr.Value.Sub d
                   (B.f32_to_f64 bld (B.f64_to_f32 bld d))
               in
               let r1 = rest d in
               let at j =
                 B.Flat
                   (B.index_add bld (B.index_scale bld 3L k) (B.index bld j))
               in
               List.iteri
                 (fun j v ->
                   B.store_f64 bld (Ssa_id.Buffer.of_int 1)
                     ~encode:Ssa_op.Encode.F32_round
                     (at (Int64.of_int j))
                     v)
                 [ d; r1; rest r1 ];
               B.Nil)
         in
         ()))

let run ?mutation ?(f32 = false) u xs =
  let p = program u xs in
  let p = if f32 then Ssa_precision.to_f32 p else p in
  let r =
    Mir_source.check_program ?mutation
      ~precision:
        (if f32 then Machine_ir.Mir_planning.Precision.F32
         else Machine_ir.Mir_planning.Precision.F64)
      p
      ~inputs:[ (0, Ssa_memory.Floats xs) ]
  in
  (* the cells are noise; the status and any disagreement are the result *)
  let status =
    match String.index_opt r ' ' with Some i -> String.sub r 0 i | None -> r
  in
  let tail =
    match Mir_storage_test.Str_find.find r " DISAGREE " with
    | Some i -> String.sub r i (String.length r - i)
    | None -> ""
  in
  Fmt.pr "%s%s: %s%s@." (Ssa_op.unary_name u)
    (if f32 then " (binary32)" else "")
    status tail

let%expect_test "libm helpers and the owned error function" =
  let all = Array.append inputs samples in
  List.iter
    (fun u ->
      run u all;
      run ~f32:true u all)
    Expr.Value.[ Cos; Erf; Exp; Log; Sin ];
  [%expect
    {|
    cos: ok
    cos (binary32): ok
    erf: ok
    erf (binary32): ok
    exp: ok
    exp (binary32): ok
    log: ok
    log (binary32): ok
    sin: ok
    sin (binary32): ok |}]

let%expect_test "erf: the operation order and the binary32 steps are seen" =
  let all = Array.append inputs samples in
  run ~mutation:Machine_lower.Mir_lower.Mutation.Erf_distributed Expr.Value.Erf
    all;
  run ~f32:true ~mutation:Machine_lower.Mir_lower.Mutation.Erf_single_rounding
    Expr.Value.Erf all;
  [%expect
    {|
    erf: ok DISAGREE structured vs generic: output t1[14]: 0x1p-53:f32 vs 0x0p+0:f32
    erf (binary32): ok DISAGREE structured vs generic: output t1[0]: 0x0p+0:f32 vs 0x1.12e0bep-30:f32 |}]
