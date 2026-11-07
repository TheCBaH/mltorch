open Ssa_ir
module B = Ssa_builder
open Ssa_ir_test.Ssa_fixtures

(* Two collapsible pairs in one nest, [(t, d)] outside [(h, w)], over the
   same buffers (a 1x1 convolution's shape in csatv2): one pass merges one
   pair of a buffer's axes, so the declared extents always match the loop
   that indexes them, and the next round merges the other pair. Both pairs
   once collapsed, a buffer is one axis of 2 * 3 * 2 * 4 cells. *)
let%expect_test "nested pairs over the same buffers collapse a round at a time"
    =
  let extents = Expr.Coord.make ~n:1L ~t:2L ~d:3L ~h:2L ~w:4L ~c:1L in
  let buffer id role =
    { Ssa_buffer.id = buf id; extents; format = Ssa_format.F32; role }
  in
  let idx bld n = B.index bld (Int64.of_int n) in
  let for_ bld n f =
    let B.Nil =
      B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld n) ~init:B.Nil (fun bld v B.Nil ->
          f bld v;
          B.Nil)
    in
    ()
  in
  let p =
    build
      ~buffers:[ buffer 0 Ssa_buffer.Input; buffer 1 Ssa_buffer.Output ]
      (fun bld ->
        for_ bld 2 (fun bld t ->
            for_ bld 3 (fun bld d ->
                for_ bld 2 (fun bld h ->
                    for_ bld 4 (fun bld w ->
                        let zero = idx bld 0 in
                        let at =
                          B.Coord (Expr.Coord.make ~n:zero ~t ~d ~h ~w ~c:zero)
                        in
                        let x =
                          B.load_f64 bld (buf 0)
                            ~decode:Ssa_op.Decode.F32_to_f64 at
                        in
                        B.store_f64 bld (buf 1) ~encode:Ssa_op.Encode.F32_round
                          at
                          (B.f64_binary bld Expr.Value.Add x x))))))
  in
  let alias = Ssa_effects.Distinct_buffers in
  let q, report =
    Ssa_opt.run ~alias
      ~passes:
        [
          Ssa_opt.guards;
          Ssa_opt.simplify;
          Ssa_opt.hoist ~alias;
          Ssa_opt.collapse;
        ]
      p
  in
  let run p =
    let out = Array.make 48 0. in
    let memory =
      memory
        [
          (0, floats (Array.init 48 (fun i -> float_of_int (i * 3))));
          (1, floats out);
        ]
    in
    let r = run_result p ~memory in
    (Result.is_ok r, Array.to_list out)
  in
  let rec loops (r : Ssa_region.t) =
    List.fold_left
      (fun n -> function
        | Ssa_stmt.For { body; _ } -> n + 1 + loops body | _ -> n)
      0 r.Ssa_region.body
  in
  Fmt.pr "loops 4 -> %d; collapse rounds %d; extents %a; same result: %b@."
    (loops q.Ssa_program.entry)
    (List.assoc "collapse" report)
    Fmt.(list ~sep:(any " ") (list ~sep:(any "x") int64))
    (List.map
       (fun (b : Ssa_buffer.t) -> Expr.Coord.to_list b.Ssa_buffer.extents)
       q.Ssa_program.buffers)
    (run p = run q);
  [%expect
    {| loops 4 -> 2; collapse rounds 2; extents 1x1x6x1x8x1 1x1x6x1x8x1; same result: true |}]
