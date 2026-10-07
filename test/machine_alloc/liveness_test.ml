open Machine_ir
open Machine_target_aarch64
module L = Machine_alloc.Mir_liveness.Make (A64)
module S = A64_stage.Sel

(* Liveness sets against an independent brute-force definition — a value is
   live into a block when some path from the block's start reaches a use
   before any redefinition — on every selected function of the source
   programs; and the intervals of a small loop. *)

let selected_of_case (case : Machine_source_test.Mir_source.Case.t) =
  match Machine_aarch64_test.A64_harness.selected case with
  | Ok (res, _) -> Some (S.Verified.selected res.A64_select.selected)
  | Error _ -> None

(* [v] live into [b]: a path from [b]'s start reaches a use of [v] (an
   operand, a terminator value or an edge argument) before [v]'s definition. *)
let brute (f : (_, _) Mir_func.t) (v : Mir_value.t) b =
  let seen = Hashtbl.create 16 in
  let rec reach id =
    if Hashtbl.mem seen (Mir_id.Block.to_int id) then false
    else (
      Hashtbl.replace seen (Mir_id.Block.to_int id) ();
      let blk = Option.get (Mir_func.find_block f id) in
      if
        List.exists (Mir_value.equal v) blk.Mir_block.params
        && not (Mir_id.Block.equal id b)
      then false
      else
        let rec scan = function
          | [] ->
              List.exists (Mir_value.equal v)
                (S.Stage.term_values blk.Mir_block.terminator)
              || List.exists
                   (fun (e : Mir_edge.t) ->
                     List.exists (Mir_value.equal v) e.Mir_edge.args
                     || reach e.Mir_edge.target)
                   (Mir_sel.Terminator.edges blk.Mir_block.terminator)
          | (i : _ Mir_instr.t) :: rest ->
              if
                List.exists (Mir_value.equal v)
                  (S.Stage.operands i.Mir_instr.op)
              then true
              else if List.exists (Mir_value.equal v) i.Mir_instr.results then
                false
              else scan rest
        in
        scan blk.Mir_block.body)
  in
  let blk = Option.get (Mir_func.find_block f b) in
  (not (List.exists (Mir_value.equal v) blk.Mir_block.params)) && reach b

let%expect_test "liveness sets equal the path definition" =
  let open Loop_ir_test in
  let open Ssa_bridge_test.Ssa_fixtures in
  let cases =
    List.filter_map Result.to_option
      [
        Machine_source_test.Mir_source.case_of_plan
          (Fusion_plan.default (matmul_kernel ~m:3 ~k:5 ~n:2))
          ~bind:(matmul_bind ~m:3 ~k:5 ~n:2 ~a:(operand 3 15) ~b:(operand 5 10));
        Machine_source_test.Mir_source.case_of_plan
          (Fusion_plan.default Loop_programs.shifted_kernel)
          ~bind:
            (bind_data ~shape:(Loop_fixtures.shape_w 4) [| 0.; 0.; 0.; 0. |]);
        Machine_source_test.Mir_source.case_of_program
          (Alloc_test.recurrences 5) ~inputs:Alloc_test.inputs ();
      ]
  in
  List.iter
    (fun case ->
      match selected_of_case case with
      | None -> print_endline "not selected"
      | Some sel ->
          List.iter
            (fun (f : (_, _) Mir_func.t) ->
              let live_in, _ = L.sets f in
              let values =
                List.concat_map
                  (fun (b : (_, _) Mir_block.t) ->
                    b.Mir_block.params
                    @ List.concat_map
                        (fun (i : _ Mir_instr.t) -> i.Mir_instr.results)
                        b.Mir_block.body)
                  f.Mir_func.blocks
                |> List.filter (fun (v : Mir_value.t) ->
                    Mir_type.has_storage v.Mir_value.ty)
              in
              let checks = ref 0 and wrong = ref 0 in
              List.iter
                (fun (b : (_, _) Mir_block.t) ->
                  List.iter
                    (fun v ->
                      incr checks;
                      if
                        Mir_id.Value.Set.mem v.Mir_value.id
                          (live_in b.Mir_block.id)
                        <> brute f v b.Mir_block.id
                      then incr wrong)
                    values)
                f.Mir_func.blocks;
              Fmt.pr "%d blocks, %d values, %d checks, %d wrong@."
                (List.length f.Mir_func.blocks)
                (List.length values) !checks !wrong)
            sel.S.program.Mir_program.funcs)
    cases;
  [%expect
    {|
    11 blocks, 75 values, 825 checks, 0 wrong
    17 blocks, 173 values, 2941 checks, 0 wrong
    8 blocks, 82 values, 656 checks, 0 wrong |}]

let%expect_test "intervals of a loop, with holes" =
  match
    selected_of_case
      (Result.get_ok
         (Machine_source_test.Mir_source.case_of_program
            (Alloc_test.recurrences 2) ~inputs:Alloc_test.inputs ()))
  with
  | None -> print_endline "not selected"
  | Some sel ->
      let f = List.hd sel.S.program.Mir_program.funcs in
      List.iter
        (fun (iv : L.Interval.t) ->
          if
            List.length iv.L.Interval.ranges > 1
            || Mir_type.equal iv.L.Interval.value.Mir_value.ty Mir_type.F64
          then
            Fmt.pr "%a:%a %a uses %a@." Mir_value.pp iv.L.Interval.value
              Mir_type.pp iv.L.Interval.value.Mir_value.ty
              Fmt.(list ~sep:(any " ") (pair ~sep:(any ",") int int))
              iv.L.Interval.ranges
              Fmt.(list ~sep:(any ",") int)
              iv.L.Interval.uses)
        (L.intervals f);
      [%expect
        {|
        %18:f64 7,16 uses 15
        %19:f64 11,16 uses 15
        %3:f64 16,22 154,157 uses 21,156
        %2:i32 16,22 154,163 uses 18,162
        %4:f64 16,22 154,168 uses 21,156,167
        %6:f64 22,43 uses 42
        %7:f64 22,67 uses 66
        %45:f64 75,84 uses 83
        %46:f64 79,84 uses 83
        %9:i32 84,90 142,149 uses 86,148
        %10:f64 84,90 142,154 uses 89,153
        %11:f64 84,90 142,154 uses 89,153
        %13:f64 90,111 uses 110
        %14:f64 90,135 uses 134
        %77:f64 157,168 uses 167 |}]

(* Soundness: every use position and every block start a value is live into
   lies in one of its ranges. Coverage where a value is dead is imprecision,
   not error: the loop rule keeps a value live into a header through the
   whole linear range of its loop, including blocks laid out inside that
   range that never reach a use (a guard's failure exit). *)
let%expect_test "intervals cover the sets" =
  let open Ssa_bridge_test.Ssa_fixtures in
  let case =
    Result.get_ok
      (Machine_source_test.Mir_source.case_of_plan
         (Fusion_plan.default (matmul_kernel ~m:3 ~k:5 ~n:2))
         ~bind:(matmul_bind ~m:3 ~k:5 ~n:2 ~a:(operand 3 15) ~b:(operand 5 10)))
  in
  let sel = Option.get (selected_of_case case) in
  let f = List.hd sel.S.program.Mir_program.funcs in
  let live_in, _ = L.sets f in
  let spans = L.spans f in
  let covers (iv : L.Interval.t) p =
    List.exists (fun (a, b) -> a <= p && p < b) iv.L.Interval.ranges
  in
  let defined_in (b : (_, _) Mir_block.t) v =
    List.exists (Mir_value.equal v) b.Mir_block.params
    || List.exists
         (fun (i : _ Mir_instr.t) ->
           List.exists (Mir_value.equal v) i.Mir_instr.results)
         b.Mir_block.body
  in
  let unsound = ref 0 and imprecise = ref 0 and checks = ref 0 in
  List.iter
    (fun (iv : L.Interval.t) ->
      let v = iv.L.Interval.value in
      List.iter
        (fun u ->
          incr checks;
          if not (covers iv u) then incr unsound)
        iv.L.Interval.uses;
      List.iter
        (fun (bid, (from, _)) ->
          let b = Option.get (Mir_func.find_block f bid) in
          incr checks;
          let live = Mir_id.Value.Set.mem v.Mir_value.id (live_in bid) in
          if live && not (covers iv from) then incr unsound;
          if (not live) && (not (defined_in b v)) && covers iv from then
            incr imprecise)
        spans)
    (L.intervals f);
  Fmt.pr "%d checks: %d unsound, %d imprecise@." !checks !unsound !imprecise;
  [%expect {| 912 checks: 0 unsound, 5 imprecise |}]
