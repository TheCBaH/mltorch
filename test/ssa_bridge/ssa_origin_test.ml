open Ssa_ir
open Loop_ir_test

(* The value-and-domain provenance the structural gate reads: which logical value
   each instruction came from, kept through the passes. *)

let rec instrs_of (r : Ssa_region.t) =
  List.concat_map
    (function
      | Ssa_stmt.Instr i -> [ i ]
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          instrs_of body
      | Ssa_stmt.If { then_; else_; _ } -> instrs_of then_ @ instrs_of else_)
    r.Ssa_region.body

(* Instructions per origin, in the order the origins first appear. *)
let attribution_in instrs =
  List.fold_left
    (fun acc (i : Ssa_instr.t) ->
      let o = i.Ssa_instr.origin in
      match List.find_opt (fun (o', _) -> Ssa_origin.equal o o') acc with
      | Some _ ->
          List.map
            (fun (o', n) ->
              if Ssa_origin.equal o o' then (o', n + 1) else (o', n))
            acc
      | None -> acc @ [ (o, 1) ])
    [] instrs

let attribution (p : Ssa_program.t) =
  attribution_in (instrs_of p.Ssa_program.entry)

let show_in name instrs =
  Fmt.pr "%s: %a@." name
    Fmt.(
      list ~sep:(any ", ") (fun ppf (o, n) -> pf ppf "%a x%d" Ssa_origin.pp o n))
    (attribution_in instrs)

let show name p = show_in name (instrs_of p.Ssa_program.entry)

let lowered plan =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower plan)

let%expect_test "lowering marks each stored value's instructions" =
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  show "stored" (lowered (Fusion_plan.default both));
  let only_t2 = Loop_programs.chain ~outputs:[ Loop_fixtures.tid 2 ] in
  let fused, _ = Fusion_plan.plan only_t2 in
  show "fused" (lowered fused);
  [%expect
    {|
    stored: value b1 [1x1x1x1x2x1] x18, value b2 [1x1x1x1x2x1] x18
    fused: value b2 [1x1x1x1x2x1] x22 |}]

let unknown p =
  match
    List.find_opt
      (fun (o, _) -> Ssa_origin.equal o Ssa_origin.Unknown)
      (attribution p)
  with
  | Some (_, n) -> n
  | None -> 0

let%expect_test "the rewriting passes keep every origin" =
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  let p = lowered (Fusion_plan.default both) in
  show "lowered" p;
  let q, _ =
    Ssa_opt.run ~passes:[ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.collapse ] p
  in
  show "after simplify, guards, collapse" q;
  Fmt.pr "instructions with no origin: %d@." (unknown q);
  [%expect
    {|
    lowered: value b1 [1x1x1x1x2x1] x18, value b2 [1x1x1x1x2x1] x18
    after simplify, guards, collapse: value b1 [1x1x1x1x2x1] x8, value b2 [1x1x1x1x2x1] x6
    instructions with no origin: 0 |}]

let%expect_test "an attribution swapped between two values is seen" =
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  let p = lowered (Fusion_plan.default both) in
  (* the same instructions, the same totals: the origins of the two values
     exchanged *)
  let swap (o : Ssa_origin.t) =
    match o with
    | Ssa_origin.Value v ->
        let n = Ssa_id.Buffer.to_int v.Ssa_origin.Value.buffer in
        let other = if n = 1 then 2 else if n = 2 then 1 else n in
        Ssa_origin.Value
          { v with Ssa_origin.Value.buffer = Ssa_id.Buffer.of_int other }
    | o -> o
  in
  let rec map_region (r : Ssa_region.t) =
    {
      r with
      Ssa_region.body =
        List.map
          (function
            | Ssa_stmt.Instr i ->
                Ssa_stmt.Instr
                  { i with Ssa_instr.origin = swap i.Ssa_instr.origin }
            | Ssa_stmt.For f -> Ssa_stmt.For { f with body = map_region f.body }
            | Ssa_stmt.If f ->
                Ssa_stmt.If
                  {
                    f with
                    then_ = map_region f.then_;
                    else_ = map_region f.else_;
                  }
            | Ssa_stmt.Ordered_sum f ->
                Ssa_stmt.Ordered_sum { f with body = map_region f.body })
          r.Ssa_region.body;
    }
  in
  let swapped = { p with Ssa_program.entry = map_region p.Ssa_program.entry } in
  let totals q = List.fold_left (fun n (_, k) -> n + k) 0 (attribution q) in
  Fmt.pr "same totals: %b; same attribution: %b@."
    (totals p = totals swapped)
    (attribution p = attribution swapped);
  [%expect {| same totals: true; same attribution: false |}]

(* What each pass does to the attribution. The two values are 16 and 8 wide, so
   an instruction carries the right origin only if both its buffer and its
   domain are its own. *)
let store_buffer (i : Ssa_instr.t) =
  match i.Ssa_instr.op with
  | Ssa_op.Store { buffer; _ } | Ssa_op.Vec_store { buffer; _ } -> Some buffer
  | _ -> None

let width_of buffer = if Ssa_id.Buffer.to_int buffer = 1 then 16 else 8

(* The origin is the value of [buffer], over its own domain. *)
let owns buffer = function
  | Ssa_origin.Value v ->
      Ssa_id.Buffer.equal buffer v.Ssa_origin.Value.buffer
      && List.map Core.Dim.to_int v.Ssa_origin.Value.domain
         = [ 1; 1; 1; 1; width_of buffer; 1 ]
  | Ssa_origin.Output _ | Ssa_origin.Unknown -> false

let count f l = List.length (List.filter f l)

let unknown_in instrs =
  count
    (fun (i : Ssa_instr.t) ->
      Ssa_origin.equal i.Ssa_instr.origin Ssa_origin.Unknown)
    instrs

let stores_misattributed instrs =
  count
    (fun (i : Ssa_instr.t) ->
      match store_buffer i with
      | Some b -> not (owns b i.Ssa_instr.origin)
      | None -> false)
    instrs

(* A top-level statement that stores to one buffer is that value's: every
   instruction inside it must say so, not only the store. *)
let nest_misattributed (p : Ssa_program.t) =
  let entry = p.Ssa_program.entry in
  List.fold_left
    (fun n s ->
      let inside = instrs_of { entry with Ssa_region.body = [ s ] } in
      match List.sort_uniq compare (List.filter_map store_buffer inside) with
      | [ buffer ] ->
          n + count (fun i -> not (owns buffer i.Ssa_instr.origin)) inside
      | _ -> n)
    0 entry.Ssa_region.body

let vector_accesses instrs =
  count
    (fun (i : Ssa_instr.t) ->
      match i.Ssa_instr.op with
      | Ssa_op.Vec_load _ | Ssa_op.Vec_store _ -> true
      | _ -> false)
    instrs

let report name instrs =
  Fmt.pr
    "%s: %d instructions (%d vector accesses), %d unknown, %d stores \
     misattributed@."
    name (List.length instrs) (vector_accesses instrs) (unknown_in instrs)
    (stores_misattributed instrs);
  show_in "  by value" instrs

let report_program name (p : Ssa_program.t) =
  report name (instrs_of p.Ssa_program.entry);
  Fmt.pr "  nests attributed to another value: %d@." (nest_misattributed p)

let cfg_instrs (c : Ssa_cfg.t) =
  List.concat_map
    (fun (b : Ssa_cfg_block.t) -> b.Ssa_cfg_block.body)
    c.Ssa_cfg.blocks

(* Two stored values over different domains, 16 and 8 wide, the second reading
   the first: loops long enough for a four-lane vector, and a domain that tells
   the two values apart. *)
let two_values () =
  let open Loop_fixtures in
  let ld id =
    Expr.Value.load
      (Expr_bridge.source_of_id (tid id))
      (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
  in
  let value id width src =
    {
      Kernel.Value.id = tid id;
      sg = sg id (shape_w width) f32;
      computation =
        Region_group.Ref.Solo
          (Region_program.pixel (Expr.Value.add (ld src) (Expr.Value.const 1.)));
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  lowered
    (Fusion_plan.default
       (Err.or_raise ~pp_error:Kernel.pp_error
          (Kernel.create
             ~inputs:
               [
                 {
                   Kernel.Input.id = tid 0;
                   sg = sg 0 (shape_w 16) f32;
                   binding = Kernel.Binding.Caller;
                 };
               ]
             ~values:[ value 1 16 0; value 2 8 1 ]
             ~outputs:[ tid 1; tid 2 ]
             ())))

let alias = Ssa_effects.Distinct_buffers
let target = Ssa_target.forced Ssa_target.neon128

let vectorized p =
  fst
    (Ssa_opt.run ~alias ~target ~passes:(Ssa_opt.pipeline ~target ~alias ()) p)

let%expect_test "lowering carries each value's domain" =
  report_program "lowered" (two_values ());
  [%expect
    {|
    lowered: 36 instructions (0 vector accesses), 0 unknown, 0 stores misattributed
      by value: value b1 [1x1x1x1x16x1] x18, value b2 [1x1x1x1x8x1] x18
      nests attributed to another value: 0 |}]

let%expect_test "precision keeps the value and its domain" =
  report_program "precision" (Ssa_precision.to_f32 ~exact:true (two_values ()));
  [%expect
    {|
    precision: 32 instructions (0 vector accesses), 0 unknown, 0 stores misattributed
      by value: value b1 [1x1x1x1x16x1] x16, value b2 [1x1x1x1x8x1] x16
      nests attributed to another value: 0 |}]

let%expect_test "vectorization keeps the value and its domain" =
  report_program "vectorized"
    (vectorized (Ssa_precision.to_f32 ~exact:true (two_values ())));
  [%expect
    {|
    vectorized: 11 instructions (4 vector accesses), 0 unknown, 0 stores misattributed
      by value: value b1 [1x1x1x1x16x1] x7, value b2 [1x1x1x1x8x1] x4
      nests attributed to another value: 0 |}]

let%expect_test "lane expansion keeps the value and its domain" =
  report_program "expanded"
    (Ssa_vec_expand.program
       (vectorized (Ssa_precision.to_f32 ~exact:true (two_values ()))));
  [%expect
    {|
    expanded: 52 instructions (0 vector accesses), 0 unknown, 0 stores misattributed
      by value: value b1 [1x1x1x1x16x1] x27, value b2 [1x1x1x1x8x1] x25
      nests attributed to another value: 0 |}]

let%expect_test "the control-flow lowering keeps the value and its domain" =
  let p = vectorized (Ssa_precision.to_f32 ~exact:true (two_values ())) in
  let c =
    Err.or_raise ~pp_error:Ssa_cfg_lower.pp_error (Ssa_cfg_lower.program p)
  in
  report "cfg" (cfg_instrs c);
  [%expect
    {|
    cfg: 17 instructions (4 vector accesses), 0 unknown, 0 stores misattributed
      by value: value b1 [1x1x1x1x16x1] x10, value b2 [1x1x1x1x8x1] x7 |}]

let%expect_test "unrolling keeps the value and its domain" =
  let p = vectorized (Ssa_precision.to_f32 ~exact:true (two_values ())) in
  let q, _ =
    Ssa_opt_unroll.pass ~unroll:(Ssa_blocking.Unroll.of_int 2) ~alias p
  in
  report_program "unrolled" q;
  [%expect
    {|
    unrolled: 23 instructions (8 vector accesses), 0 unknown, 0 stores misattributed
      by value: value b1 [1x1x1x1x16x1] x13, value b2 [1x1x1x1x8x1] x10
      nests attributed to another value: 0 |}]

(* One matmul value through blocking, row blocking and contraction: every
   instruction a pass makes or moves keeps the value and its domain, so a pass
   that attributed its new instructions to another value, or to none, shows. *)
let matmul () =
  lowered (Fusion_plan.default (Ssa_fixtures.matmul_kernel ~m:4 ~k:5 ~n:8))

let report_matmul name instrs =
  Fmt.pr "%s: %d instructions, %d unknown@." name (List.length instrs)
    (unknown_in instrs);
  show_in "  by value" instrs

let%expect_test "blocking keeps the value and its domain" =
  let q, _ =
    Ssa_opt.run ~alias
      ~passes:
        [
          Ssa_opt.simplify;
          Ssa_opt.guards;
          Ssa_opt.simplify;
          Ssa_opt.hoist ~alias;
          Ssa_opt.block ~alias ~group:Ssa_opt_block.Auto;
        ]
      (matmul ())
  in
  report_matmul "blocked" (instrs_of q.Ssa_program.entry);
  [%expect
    {|
    blocked: 43 instructions, 0 unknown
      by value: value b2 [1x1x1x4x8x1] x43 |}]

let%expect_test "row blocking keeps the value and its domain" =
  let v, _ = Ssa_opt.run ~alias ~target:Ssa_target.wasm128 (matmul ()) in
  let q, n =
    Ssa_opt_rows.program ~rows:(Ssa_blocking.Rows.of_int 2) ~policy:alias v
  in
  Fmt.pr "%d block(s)@." n;
  report_matmul "rows" (instrs_of q.Ssa_program.entry);
  [%expect
    {|
    1 block(s)
    rows: 28 instructions, 0 unknown
      by value: value b2 [1x1x1x4x8x1] x28 |}]

let%expect_test "contraction keeps the value and its domain" =
  let q, found =
    Ssa_contract.program ~numerics:Contraction.Numerics.Ordered
      (Ssa_plan.binary32_scalar (matmul ()))
  in
  Fmt.pr "%d contraction(s)@." (List.length found);
  report_matmul "contracted" (instrs_of q.Ssa_program.entry);
  [%expect
    {|
    1 contraction(s)
    contracted: 9 instructions, 0 unknown
      by value: value b2 [1x1x1x4x8x1] x9 |}]
