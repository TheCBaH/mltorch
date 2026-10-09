(* M12.3: the artifact native assembly receives. Only a realized program that
   re-passes the physical verifier and the checker is published, with its
   identity, symbols, relocations and origins. *)

open Machine_ir
open Machine_target_aarch64
module A = Machine_alloc.Mir_ref_alloc.Make (A64) (A64_regs)
module Fr = Machine_alloc.Mir_frame.Make (A64) (A64_frame)
module Pub = Machine_model.Mir_artifact.Make (A64)
module Art = Machine_model.Mir_artifact

let planning p =
  Mir_planning.make
    ~subject:(Machine_lower.Mir_lower.subject p)
    ~policy:"reference_f64" ~schedule:"scalar"
    ~precision:Mir_planning.Precision.F64 ~lanes:(Mir_type.Lanes.of_int 1)
    ~fma:Mir_planning.Fma.Forbidden ~capabilities:[]

let lowered p =
  let planning = planning p in
  ( planning,
    (Err.or_raise ~pp_error:Machine_lower.Mir_lower.Refusal.pp
       (Machine_lower.Mir_lower.program ~planning:(Some planning) p))
      .Machine_lower.Mir_lower.program )

let reference fmt = function
  | Mir_target.Reference.Call c -> Fmt.pf fmt "call %a" Mir_op.Callee.pp c
  | Mir_target.Reference.View (v, form) ->
      Fmt.pf fmt "%s %a" (Mir_target.Reference.Form.name form) Mir_id.View.pp v

let%expect_test "an exp kernel's artifact" =
  let p =
    Machine_source_test.Mir_math_test.program Expr.Value.Exp [| 1.; 2. |]
  in
  let planning, g = lowered p in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let v = res.A64_select.selected in
  let real = Result.get_ok (Fr.realize (A.allocate v)) in
  (match Pub.publish ~planning v real with
  | Error e -> Fmt.pr "refused: %s@." e
  | Ok a ->
      Fmt.pr "%a@." Art.pp_summary a;
      List.iter
        (fun (s : Art.Symbol.t) ->
          match s.Art.Symbol.kind with
          | Art.Symbol.Data { size; align; section; _ } ->
              Fmt.pr "  %s: %s, %Ld bytes, align %Ld@." s.Art.Symbol.name
                (Art.Section.name section) size align
          | Art.Symbol.External_function f ->
              Fmt.pr "  %s: external %s@." s.Art.Symbol.name f
          | Art.Symbol.Function f ->
              Fmt.pr "  %s: function %a@." s.Art.Symbol.name Mir_id.Func.pp f)
        (Art.symbols a);
      (* each distinct relocation, with how often it occurs *)
      let tally = Hashtbl.create 8 in
      List.iter
        (fun (r : Art.Relocation.t) ->
          let k =
            Fmt.str "%a -> %s+%Ld" reference r.Art.Relocation.reference
              r.Art.Relocation.symbol r.Art.Relocation.addend
          in
          Hashtbl.replace tally k
            (1 + Option.value ~default:0 (Hashtbl.find_opt tally k)))
        (Art.relocations a);
      List.iter
        (fun (k, n) -> Fmt.pr "  %dx %s@." n k)
        (List.sort compare (List.of_seq (Hashtbl.to_seq tally)));
      (* an instruction of a failure's record expansion has no CFG site *)
      Fmt.pr "  origins: %d, %d at a CFG site@."
        (List.length (Art.origins a))
        (List.length
           (List.filter
              (fun (_, (o : Mir_origin.t)) -> Option.is_some o.Mir_origin.cfg)
              (Art.origins a))));
  [%expect
    {|
    aarch64 (Arm ARM DDI 0487 K.a) features [fp], reference_f64/scalar/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 1 external [exp], 1 function; 21 relocations
      mir_region_0: bound, 16 bytes, align 16
      mir_region_1: bound, 24 bytes, align 16
      mir_region_1000000: bss, 104 bytes, align 16
      kernel: function fn0
      exp: external exp
      1x call helper1 -> exp+0
      1x page view0 -> mir_region_0+0
      3x page view1 -> mir_region_1+0
      6x page view1000000 -> mir_region_1000000+0
      1x page_offset view0 -> mir_region_0+0
      3x page_offset view1 -> mir_region_1+0
      6x page_offset view1000000 -> mir_region_1000000+0
      origins: 287, 286 at a CFG site |}]

let%expect_test "what publication refuses" =
  let p = Machine_source_test.Mir_math_test.program Expr.Value.Exp [| 1. |] in
  let planning, g = lowered p in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let v = res.A64_select.selected in
  (* an allocation whose frames were never realized *)
  (match Pub.publish ~planning v (A.allocate v) with
  | Error e -> Fmt.pr "unrealized: %s@." e
  | Ok _ -> Fmt.pr "unrealized: published@.");
  (* a helper modelled for the interpreters only *)
  let g =
    Result.get_ok
      (Err.payload (Mir_verify.generic Machine_alloc_test.Calls_test.program))
  in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let v = res.A64_select.selected in
  (match Fr.realize (A.allocate v) with
  | Error r -> Fmt.pr "frame: %a@." Machine_alloc.Mir_frame.Refusal.pp r
  | Ok real -> (
      match Pub.publish ~planning v real with
      | Error e -> Fmt.pr "interpreter-only helper: %s@." e
      | Ok _ -> Fmt.pr "interpreter-only helper: published@."));
  [%expect
    {|
    unrealized: fn0 has no realized frame
    interpreter-only helper: helper guard (version 1) has no native binding |}]
