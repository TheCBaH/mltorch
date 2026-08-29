open Ssa_ir
open Ssa_fixtures

(* Programs here are built from records, not through the typed builder: the
   verifier must reject what a transformation or a hand-built program could
   produce and the builder never would. *)

let v n ty = { Ssa_value.id = Ssa_id.Value.of_int n; ty }
let effect_ty = Ssa_type.Effect
let index_ty = Ssa_type.Scalar Ssa_type.Index
let f64_ty = Ssa_type.Scalar Ssa_type.F64
let pred_ty = Ssa_type.Scalar Ssa_type.Pred

let region n params body yields =
  { Ssa_region.id = Ssa_id.Region.of_int n; params; body; yields }

let instr ?token results op =
  Ssa_stmt.Instr { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }

let program ?(buffers = []) entry =
  {
    Ssa_program.revision = Ssa_id.Revision.of_int 0;
    buffers;
    entry;
    scan_limits = Expr.Scan_limits.default;
    next_value = Ssa_id.Value.Next.of_int 64;
    next_region = Ssa_id.Region.Next.of_int 64;
  }

let verdict p =
  match Err.payload (Ssa_verify.check p) with
  | Ok () -> Fmt.pr "ok@."
  | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e

let e0 = v 0 effect_ty
let i1 = v 1 index_ty

let%expect_test "a well-formed program verifies" =
  verdict
    (program
       (region 0 [ e0 ]
          [ instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 3L)) ]
          [ e0 ]));
  [%expect {| ok |}]

let%expect_test "duplicate definition" =
  verdict
    (program
       (region 0 [ e0 ]
          [
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 3L));
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 4L));
          ]
          [ e0 ]));
  [%expect {| r0, stmt1: v1 is defined twice |}]

let%expect_test "use before definition" =
  verdict
    (program
       (region 0 [ e0 ]
          [
            instr
              [ v 2 (Ssa_type.Scalar Ssa_type.I64) ]
              (Ssa_op.Convert (Ssa_op.Convert.Index_to_i64, i1));
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 3L));
          ]
          [ e0 ]));
  [%expect {| r0, stmt0: v1 is not in scope at its use |}]

let%expect_test "operand of the wrong type" =
  verdict
    (program
       (region 0 [ e0 ]
          [
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 3L));
            instr
              [ v 2 f64_ty ]
              (Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, i1));
          ]
          [ e0 ]));
  [%expect {| r0, stmt1: operand0: expected f32, found index |}]

let%expect_test "a use that retypes its definition" =
  verdict
    (program
       (region 0 [ e0 ]
          [
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 3L));
            instr
              [ v 2 f64_ty ]
              (Ssa_op.Convert (Ssa_op.Convert.Index_to_f64, v 1 f64_ty));
          ]
          [ e0 ]));
  [%expect {| r0, stmt1: v1 is defined as index and used as f64 |}]

let%expect_test "declared results disagree with the operation" =
  verdict
    (program
       (region 0 [ e0 ]
          [ instr [ v 1 f64_ty ] (Ssa_op.Const (Ssa_const.Index 3L)) ]
          [ e0 ]));
  [%expect {| r0, stmt0: results declared (f64), operation yields (index) |}]

let%expect_test "an effectful operation must consume the current effect" =
  let b = buffer 0 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Input in
  let load token e =
    instr ?token
      [ v 2 f64_ty; e ]
      (Ssa_op.Load
         {
           buffer = buf 0;
           at = Ssa_access.Flat i1;
           decode = Ssa_op.Decode.F32_to_f64;
         })
  in
  let base = instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 0L)) in
  (* forking: two loads both consume the entry effect *)
  verdict
    (program ~buffers:[ b ]
       (region 0 [ e0 ]
          [
            base;
            load (Some e0) (v 3 effect_ty);
            instr ~token:e0
              [ v 4 f64_ty; v 5 effect_ty ]
              (Ssa_op.Load
                 {
                   buffer = buf 0;
                   at = Ssa_access.Flat i1;
                   decode = Ssa_op.Decode.F32_to_f64;
                 });
          ]
          [ v 5 effect_ty ]));
  (* dropped: the load's effect is never yielded *)
  verdict
    (program ~buffers:[ b ]
       (region 0 [ e0 ] [ base; load (Some e0) (v 3 effect_ty) ] [ e0 ]));
  (* the token is missing *)
  verdict
    (program ~buffers:[ b ]
       (region 0 [ e0 ] [ base; load None (v 3 effect_ty) ] [ v 3 effect_ty ]));
  [%expect
    {|
    r0, stmt2: effect v0 is not the current effect v3
    r0, stmt2: effect v0 is not the current effect v3
    r0, stmt1: effect operand missing or unexpected
    |}]

let i64_ty = Ssa_type.Scalar Ssa_type.I64

let%expect_test "a value defined in a region does not escape it" =
  let inner =
    region 1
      [ v 2 index_ty; v 3 effect_ty ]
      [ instr [ v 4 index_ty ] (Ssa_op.Const (Ssa_const.Index 9L)) ]
      [ v 3 effect_ty ]
  in
  verdict
    (program
       (region 0 [ e0 ]
          [
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 0L));
            Ssa_stmt.For
              {
                lo = i1;
                hi = i1;
                step = 1L;
                inits = [ e0 ];
                results = [ v 5 effect_ty ];
                body = inner;
              };
            instr
              [ v 6 i64_ty ]
              (Ssa_op.Convert (Ssa_op.Convert.Index_to_i64, v 4 index_ty));
          ]
          [ v 5 effect_ty ]));
  [%expect {| r0, stmt2: v4 is not in scope at its use |}]

let%expect_test "loop signatures: step, arity and the effect it carries" =
  let loop ?(step = 1L) ?(params = [ v 2 index_ty; v 3 effect_ty ])
      ?(yields = [ v 3 effect_ty ]) ?(inits = [ e0 ])
      ?(results = [ v 5 effect_ty ]) () =
    program
      (region 0 [ e0 ]
         [
           instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 0L));
           Ssa_stmt.For
             {
               lo = i1;
               hi = i1;
               step;
               inits;
               results;
               body = region 1 params [] yields;
             };
         ]
         [ v 5 effect_ty ])
  in
  verdict (loop ());
  verdict (loop ~step:0L ());
  verdict (loop ~params:[ v 2 index_ty ] ~yields:[] ~inits:[] ());
  verdict (loop ~yields:[ v 2 index_ty ] ());
  verdict (loop ~results:[ v 5 f64_ty ] ());
  [%expect
    {|
    ok
    r0, stmt1: loop step 0 is not positive
    r0, stmt1: a region carries exactly one effect, signature ()
    r1, stmt0: a region carries exactly one effect, signature (index)
    r0, stmt1: signature (effect) expected, (f64) found
    |}]

let%expect_test "a loop body cannot reach the outer effect" =
  (* the body consumes e0, captured from outside, instead of its own parameter *)
  let b = buffer 0 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Output in
  verdict
    (program ~buffers:[ b ]
       (region 0 [ e0 ]
          [
            instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 0L));
            instr [ v 7 f64_ty ] (Ssa_op.Const (Ssa_const.F64 1.));
            Ssa_stmt.For
              {
                lo = i1;
                hi = i1;
                step = 1L;
                inits = [ e0 ];
                results = [ v 5 effect_ty ];
                body =
                  region 1
                    [ v 2 index_ty; v 3 effect_ty ]
                    [
                      instr ~token:e0
                        [ v 8 effect_ty ]
                        (Ssa_op.Store
                           {
                             buffer = buf 0;
                             at = Ssa_access.Flat i1;
                             encode = Ssa_op.Encode.F32_round;
                             value = v 7 f64_ty;
                           });
                    ]
                    [ v 8 effect_ty ];
              };
          ]
          [ v 5 effect_ty ]));
  [%expect {| r1, stmt0: effect v0 is not the current effect v3 |}]

let%expect_test "buffer permissions and formats" =
  let input = buffer 0 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Input in
  let shape = buffer 1 ~h:1L ~w:4L Ssa_format.I64 Ssa_buffer.Output in
  let prelude =
    [
      instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 0L));
      instr [ v 2 f64_ty ] (Ssa_op.Const (Ssa_const.F64 1.));
    ]
  in
  let attempt op =
    verdict
      (program ~buffers:[ input; shape ]
         (region 0 [ e0 ]
            (prelude @ [ instr ~token:e0 [ v 3 effect_ty ] op ])
            [ v 3 effect_ty ]))
  in
  attempt
    (Ssa_op.Store
       {
         buffer = buf 0;
         at = Ssa_access.Flat i1;
         encode = Ssa_op.Encode.F32_round;
         value = v 2 f64_ty;
       });
  attempt
    (Ssa_op.Store
       {
         buffer = buf 1;
         at = Ssa_access.Flat i1;
         encode = Ssa_op.Encode.F32_round;
         value = v 2 f64_ty;
       });
  attempt
    (Ssa_op.Store
       {
         buffer = buf 9;
         at = Ssa_access.Flat i1;
         encode = Ssa_op.Encode.F32_round;
         value = v 2 f64_ty;
       });
  [%expect
    {|
    r0, stmt2: b0 is an input and is never written
    r0, stmt2: b1 is declared i64 but accessed as f32
    r0, stmt2: b9 is not declared
    |}]

let%expect_test "an effect may be reused in mutually exclusive branches" =
  (* the builder produces this shape: both branches start from the incoming
     effect, only one runs *)
  let b = buffer 0 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Output in
  let p =
    build ~buffers:[ b ] (fun bld ->
        let c = Ssa_builder.pred bld true in
        let store bld x =
          let zero = Ssa_builder.index bld 0L in
          Ssa_builder.store_f64 bld (buf 0) ~encode:Ssa_op.Encode.F32_round
            (Ssa_builder.Flat zero) (Ssa_builder.f64 bld x)
        in
        let Ssa_builder.Nil =
          Ssa_builder.if_ bld c
            ~then_:(fun bld ->
              store bld 1.;
              Ssa_builder.Nil)
            ~else_:(fun bld ->
              store bld 2.;
              Ssa_builder.Nil)
        in
        ())
  in
  verdict p;
  (* a branch that skips its own effect link is rejected *)
  let broken =
    match p.Ssa_program.entry.Ssa_region.body with
    | [ c; Ssa_stmt.If i ] ->
        let else_ = { i.else_ with Ssa_region.yields = [ e0 ] } in
        {
          p with
          entry = { p.entry with body = [ c; Ssa_stmt.If { i with else_ } ] };
        }
    | _ -> invalid_arg "unexpected shape"
  in
  verdict broken;
  [%expect
    {|
    ok
    r1, stmt3: effect v0 is not the current effect v7
    |}]

let%expect_test "a per-channel buffer takes coordinate accesses only" =
  let per_channel =
    Ssa_format.I8
      (Ssa_format.Per_channel
         { scale = [| 0.5; 0.25 |]; zero_point = [| 0; 1 |] })
  in
  let declare channels =
    {
      Ssa_buffer.id = buf 0;
      extents =
        Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w:1L ~c:(Int64.of_int channels);
      format = per_channel;
      role = Ssa_buffer.Input;
    }
  in
  let load at =
    instr ~token:e0
      [ v 2 f64_ty; v 3 effect_ty ]
      (Ssa_op.Load { buffer = buf 0; at; decode = Ssa_op.Decode.I8_dequant })
  in
  let zero = instr [ i1 ] (Ssa_op.Const (Ssa_const.Index 0L)) in
  let flat = load (Ssa_access.Flat i1) in
  let coord =
    load
      (Ssa_access.Coord (Expr.Coord.make ~n:i1 ~t:i1 ~d:i1 ~h:i1 ~w:i1 ~c:i1))
  in
  verdict
    (program
       ~buffers:[ declare 2 ]
       (region 0 [ e0 ] [ zero; flat ] [ v 3 effect_ty ]));
  verdict
    (program
       ~buffers:[ declare 2 ]
       (region 0 [ e0 ] [ zero; coord ] [ v 3 effect_ty ]));
  (* the parameters must cover exactly the C extent *)
  verdict
    (program
       ~buffers:[ declare 3 ]
       (region 0 [ e0 ] [ zero; coord ] [ v 3 effect_ty ]));
  [%expect
    {|
    r0, stmt1: b0 is per-channel quantized and takes no flat access
    ok
    r0, stmt0: buffer b0 is declared twice, or with extents no index holds |}]

(* ---- proofs ------------------------------------------------------------------ *)

let%expect_test
    "an operation that carries a proof is accepted only where the range \
     analysis re-derives it" =
  let const n k = instr [ v n index_ty ] (Ssa_op.Const (Ssa_const.Index k)) in
  let sum a b r =
    instr
      [ v r index_ty ]
      (Ssa_op.Index_add_in_domain (v a index_ty, v b index_ty))
  in
  let top = 0x7FFF_FFFFL in
  let case a b =
    verdict
      (program (region 0 [ e0 ] [ const 1 a; const 2 b; sum 1 2 3 ] [ e0 ]))
  in
  case 1L 2L;
  case top 1L;
  case (-0x8000_0000L) (-1L);
  (* a product *)
  let scale a k =
    verdict
      (program
         (region 0 [ e0 ]
            [
              const 1 a;
              instr
                [ v 2 index_ty ]
                (Ssa_op.Index_scale_in_domain (k, v 1 index_ty));
            ]
            [ e0 ]))
  in
  scale 3L 2L;
  scale 0x4000_0000L 2L;
  (* a load: constants and an induction value against the extent *)
  let input extent = buffer 0 ~h:1L ~w:extent Ssa_format.F32 Ssa_buffer.Input in
  let load_at coord =
    instr ~token:e0
      [ v 9 f64_ty; v 10 effect_ty ]
      (Ssa_op.Load_in_bounds
         {
           buffer = buf 0;
           at = Ssa_access.Coord coord;
           decode = Ssa_op.Decode.F32_to_f64;
         })
  in
  let zero = v 1 index_ty in
  let pos k = v k index_ty in
  let coord w = Expr.Coord.make ~n:zero ~t:zero ~d:zero ~h:zero ~w ~c:zero in
  let attempt extent w =
    verdict
      (program
         ~buffers:[ input extent ]
         (region 0 [ e0 ]
            [
              instr [ zero ] (Ssa_op.Const (Ssa_const.Index 0L));
              instr [ pos 2 ] (Ssa_op.Const (Ssa_const.Index w));
              load_at (coord (pos 2));
            ]
            [ v 10 effect_ty ]))
  in
  attempt 4L 3L;
  attempt 4L 4L;
  attempt 4L (-1L);
  [%expect
    {|
    ok
    r0, stmt2: a proof of a sum staying in the index domain cannot be re-derived
    r0, stmt2: a proof of a sum staying in the index domain cannot be re-derived
    ok
    r0, stmt1: a proof of a product staying in the index domain cannot be re-derived
    ok
    r0, stmt2: a proof of an access staying in its buffer cannot be re-derived
    r0, stmt2: a proof of an access staying in its buffer cannot be re-derived |}]
