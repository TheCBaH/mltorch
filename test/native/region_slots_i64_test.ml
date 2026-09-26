(* Exercises [Region_local_i64]/[Region_slots_i64] (the typed scratch-segment
   machinery, kept deliberately SEPARATE from [Region_local]/[Region_slots]
   rather than generalizing them, so every existing
   float-only Region caller stays untouched). No real op wires this into
   [Region_program.t] yet (that combined-segment question is still open),
   so this test is the real caller proving the segment mechanism itself: slot
   layout, left-to-right visibility between locals, per-position vector
   fill, and exactness beyond float's 2^53 mantissa -- all through
   [Expr.Eval.value_i64], the exact entry point added alongside this same
   slice. *)

let one_cell = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let origin = Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int Vec6.origin)

let%expect_test
    "Region_slots_i64: scalar/vector locals fill and read back exactly" =
  let x = Tensor.materialize_i64 one_cell (fun _ -> 9_007_199_254_740_993L) in
  let x_sig =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:one_cell
      ~fmt:(Payload.Fmt Payload.I64) ()
  in
  let id_a, id_b, var, id_c =
    Expr.Builder.run
      (let open Expr.Builder.Syntax in
       let* id_a = Expr.Builder.fresh_local in
       let* id_b = Expr.Builder.fresh_local in
       let* var = Expr.Builder.fresh_reduce in
       let+ id_c = Expr.Builder.fresh_local in
       (id_a, id_b, var, id_c))
  in
  let locals =
    [
      (* [id_a]: an exact I64 load, beyond float's 2^53 mantissa. *)
      Region_local_i64.scalar ~id:id_a
        ~value:
          (Expr.Value.i64_load
             (Expr_bridge.source_of_id x_sig.Tensor_sig.id)
             (Expr_bridge.coord_of_vec6 (Vec6.of_fn (fun _ -> Expr.Index.zero))));
      (* [id_b]: reads [id_a] back through [Region_local_i64]'s own slot --
         proves left-to-right local visibility, the int64 twin of
         [Region_eval.evaluate_locals]'s float behavior. *)
      Region_local_i64.scalar ~id:id_b
        ~value:
          (Expr.Value.i64_add
             (Expr.Value.i64_local id_a)
             (Expr.Value.i64_const 1L));
      (* [id_c]: a 3-wide vector, each element [id_b] plus its own position
         (cast from the float [value_of_index] the reducer binder denotes) --
         proves per-position fill under [~reducer:[(var, p)]]. *)
      Region_local_i64.vector ~id:id_c ~var ~extent:(Slot.extent 3)
        ~value:
          (Expr.Value.i64_add
             (Expr.Value.i64_local id_b)
             (Expr.Value.float_to_i64
                (Expr.Value.value_of_index
                   (Expr.Index.of_position (Expr.Index.reduce var)))));
    ]
  in
  let slots = Region_slots_i64.of_locals locals in
  Fmt.pr "total slots: %a@." Slot.pp (Region_slots_i64.total slots);
  [%expect {| total slots: 5 |}];
  let env =
    Expr_bridge.env ~binding:(fun id ->
        if Tensor_id.equal id x_sig.Tensor_sig.id then Some x else None)
  in
  let values = Region_slots_i64.fill locals slots ~env ~output:origin in
  Fmt.pr "%a@." Fmt.(array ~sep:(any ",") int64) values;
  [%expect
    {| 9007199254740993,9007199254740994,9007199254740994,9007199254740995,9007199254740996 |}];
  let local_i64, local_at_i64 = Region_slots_i64.reader slots values in
  (* Reads [id_c] back at position 1 through the segment's own resolvers,
     via [Expr.Eval.value_i64] -- exact, no [i64_to_float] round trip. *)
  let read_c pos =
    Err.or_raise ~pp_error:Expr.Eval.pp_error
      (Expr.Eval.value_i64 ~local_i64 ~local_at_i64 env ~output:origin
         (Expr.Value.i64_local_at id_c
            (Expr.Index.assume_position (Expr.Index.const pos))))
  in
  Fmt.pr "c[0]=%Ld c[1]=%Ld c[2]=%Ld@." (read_c 0) (read_c 1) (read_c 2);
  [%expect
    {| c[0]=9007199254740994 c[1]=9007199254740995 c[2]=9007199254740996 |}];
  (* The exactness this whole segment exists for: the same "+1" collapses to
     a no-op once routed through float, which [id_b]'s own value above did
     NOT do. *)
  Fmt.pr "float-lossy would give %b@."
    (Float.equal (9_007_199_254_740_993. +. 1.) 9_007_199_254_740_993.);
  [%expect {| float-lossy would give true |}]
