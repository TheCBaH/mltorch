open Loop_ir

(* T3.2: the per-node twin of [loop_sweep_test.ml]'s whole-graph sweep.
   [Loop_node_program.lower] is exercised directly (not merely its
   constituent library calls) for every output of every node of every walked
   subject, and compared against the same [Kernel_eval] reference the whole-
   graph sweep uses, via [Loop_check.compare] so the comparison semantics
   (bitwise, NaN = NaN) can't drift from that sweep's own. Nothing may
   disagree: this table is the pre-JS view of what each milestone still has
   to fix (plan S3, "no route without a walk"). *)

type tally = {
  mutable agree : int;
  mutable not_a_kernel : int;
  mutable refused : int;
  mutable disagreements : string list;
}

let tallies : (string, tally) Hashtbl.t = Hashtbl.create 64

let tally target =
  match Hashtbl.find_opt tallies target with
  | Some t -> t
  | None ->
      let t =
        { agree = 0; not_a_kernel = 0; refused = 0; disagreements = [] }
      in
      Hashtbl.add tallies target t;
      t

let verify_output t g node ~bind (output, _oid) =
  match Loop_node_program.kernel g node ~output with
  | Error _ -> t.not_a_kernel <- t.not_a_kernel + 1
  | Ok kernel -> (
      let reference = Kernel_eval.run_plan (Fusion_plan.default kernel) ~bind in
      match Err.payload (Loop_node_program.lower g node ~output) with
      | Error (`Adapt _) ->
          (* Shouldn't happen: [kernel] above just proved this same
             (g, node, output) adapts. Counted here rather than assumed
             unreachable, so a future drift between the two calls shows up
             as a number instead of an exception. *)
          t.not_a_kernel <- t.not_a_kernel + 1
      | Error (`Lower _) -> t.refused <- t.refused + 1
      | Ok program -> (
          let loop = Loop_interp.run program ~bind in
          match Loop_check.compare ~reference ~loop with
          | Loop_check.Agree | Loop_check.Agree_on_failure _ ->
              t.agree <- t.agree + 1
          | Loop_check.Refused _ -> t.refused <- t.refused + 1
          | Loop_check.Disagree d ->
              t.disagreements <-
                Fmt.str "%a" Loop_check.Disagreement.pp d :: t.disagreements))

let verify _ppf (s : Native_op_walk.Subject.t) =
  let g = s.Native_op_walk.Subject.graph in
  let t = tally s.Native_op_walk.Subject.target in
  let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
  List.iter
    (fun (node : Graph_ir.node) ->
      List.iter
        (verify_output t g node ~bind)
        (Output_ordinal.indexed node.Graph_ir.Node.outputs))
    g.Graph_ir.Graph.nodes;
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

let sweep () =
  List.iteri
    (fun index (m : Native_op_walk.op) ->
      ignore
        (Walk_core.Walk.run m ~verify ~ppf:silent
           ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
           ~steps:5))
    Native_op_walk.all_walks

let%expect_test "no walked node disagrees with the Kernel_eval reference" =
  sweep ();
  let rows =
    Hashtbl.fold (fun target t acc -> (target, t) :: acc) tallies []
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter
    (fun (target, t) ->
      List.iter (fun d -> Fmt.pr "%s DISAGREES: %s@." target d) t.disagreements)
    rows;
  Fmt.pr "disagreements: %d@."
    (List.fold_left (fun n (_, t) -> n + List.length t.disagreements) 0 rows);
  [%expect {|
    disagreements: 0 |}]

let%expect_test "what lowers and what is refused, by op, per node/output" =
  let rows =
    Hashtbl.fold (fun target t acc -> (target, t) :: acc) tallies []
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter
    (fun (target, t) ->
      Fmt.pr "%-28s agree=%d refused=%d%s@." target t.agree t.refused
        (if t.not_a_kernel > 0 then Fmt.str " not-a-kernel=%d" t.not_a_kernel
         else ""))
    rows;
  [%expect
    {|
    adaptive_avg_pool2d          agree=6 refused=0
    add                          agree=6 refused=0
    add_i64                      agree=6 refused=0
    add_scalar                   agree=6 refused=0
    amax                         agree=6 refused=0
    arange                       agree=6 refused=0
    arange_i64                   agree=6 refused=0
    avg_pool2d                   agree=6 refused=0
    batch_norm                   agree=6 refused=0
    batch_norm_no_stats          agree=18 refused=0
    batched_matmul               agree=6 refused=0
    bitwise_not                  agree=6 refused=0
    bmm                          agree=6 refused=0
    clamp                        agree=6 refused=0
    clone                        agree=6 refused=0
    conv2d                       agree=6 refused=0
    conv2d_padding               agree=6 refused=0
    convolution                  agree=6 refused=0
    cumsum                       agree=6 refused=0
    div                          agree=6 refused=0
    div_scalar                   agree=6 refused=0
    eq_scalar                    agree=6 refused=0
    eq_tensor                    agree=6 refused=0
    expand                       agree=6 refused=0
    eye                          agree=6 refused=0
    gelu                         agree=6 refused=0
    gt_scalar                    agree=6 refused=0
    hardsigmoid                  agree=6 refused=0
    hardswish                    agree=6 refused=0
    hardtanh                     agree=6 refused=0
    index_tensor                 agree=6 refused=0
    layer_norm                   agree=6 refused=0
    linear                       agree=6 refused=0
    lstm                         agree=18 refused=0
    max_dim                      agree=12 refused=0
    max_pool2d                   agree=6 refused=0
    max_pool2d_with_indices      agree=12 refused=0
    mean                         agree=6 refused=0
    mul                          agree=6 refused=0
    mul_i64                      agree=6 refused=0
    mul_scalar                   agree=6 refused=0
    mul_scalar_i64               agree=6 refused=0
    ne_scalar                    agree=6 refused=0
    ne_tensor                    agree=6 refused=0
    pad                          agree=6 refused=0
    permute                      agree=6 refused=0
    permute_i64                  agree=6 refused=0
    pow                          agree=6 refused=0
    relu                         agree=6 refused=0
    reshape                      agree=6 refused=0
    reshape_i64                  agree=6 refused=0
    rms_norm                     agree=6 refused=0
    sdpa                         agree=6 refused=0
    sigmoid                      agree=6 refused=0
    silu                         agree=6 refused=0
    slice                        agree=6 refused=0
    softmax                      agree=6 refused=0
    split_with_sizes             agree=26 refused=0
    split_with_sizes_i64         agree=12 refused=0
    sqrt                         agree=6 refused=0
    sub                          agree=6 refused=0
    sub_i64                      agree=6 refused=0
    sum                          agree=6 refused=0
    to_copy_bool                 agree=6 refused=0
    to_copy_float_i64            agree=6 refused=0
    to_copy_long                 agree=6 refused=0
    unbind                       agree=16 refused=0
    unbind_i64                   agree=18 refused=0
    upsample_bicubic2d           agree=6 refused=0
    upsample_bilinear2d          agree=6 refused=0
    upsample_nearest2d           agree=6 refused=0
    vector_norm                  agree=6 refused=0
    zeros                        agree=6 refused=0 |}]
