(* The canonical pipeline: that both [fold] branches run, and that each is
   idempotent. See .ai/native_transform_design.md §12h.

   Idempotence is the property worth pinning. "Canonical" only means anything if
   a second application changes nothing — otherwise the Native4D domain check is
   asking a question whose answer depends on how many times the pipeline was
   run. It is asserted as "the second run produces an empty map", which is
   stronger than comparing two printed graphs and is exactly the signal
   [Pass.fixpoint] itself uses to detect convergence. *)

open Graph_ir

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let nhwc ~h ~w ~c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c

let perm_of pairs =
  Permute.Permute.of_fn (fun a ->
      Option.value (List.assoc_opt a pairs) ~default:a)

(* Something for each half of the pipeline: an inverse permute pair for
   [relayout] to cancel, a dead branch for [Dce], and a max-pool whose indices
   are discarded for [Dce] + [Drop_pool_indices] together. *)
let mixed () =
  build "mixed"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~h:4 ~w:4 ~c:3) () in
     let* up = permute (perm_of Axis.[ (H, W); (W, H) ]) x in
     let* back = permute (perm_of Axis.[ (H, W); (W, H) ]) up in
     let* values, indices =
       max_pool2d_with_indices
         {
           ceil_mode = false;
           kernel = { h = Dim.extent 2; w = Dim.extent 2 };
           stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
           pad =
             { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
         }
         back
     in
     let* () = discard indices in
     let* _dead = sqrt values in
     relu values)

let changed map =
  not
    (Correspondence.is_empty (Graph_map.values map)
    && Node_map.is_empty (Graph_map.nodes map))

let node_count g = List.length g.Graph.nodes

(* Run the pipeline, then run it again on its own output. *)
let twice ~fold g =
  let open Err.Syntax in
  let* (Rewrite.Origin state) =
    (Rewrite.origin g :> (Rewrite.origin, Pass.error) Err.t)
  in
  let* (Rewrite.Step (once, first)) =
    Pass.run_all state [ Pipeline.canonical ~fold ]
  in
  let+ (Rewrite.Step (_, second)) =
    Pass.run_all once [ Pipeline.canonical ~fold ]
  in
  (Rewrite.graph once, changed first, changed second)

let report ~fold g =
  Format.printf "fold=%b: " fold;
  match twice ~fold g with
  | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
  | Ok (g, first, second) ->
      Format.printf "%d nodes, first run changed=%b, second changed=%b@."
        (node_count g) first second

let%expect_test "pipeline: compatibility fold flag is idempotent" =
  let g = mixed () in
  Format.printf "before: %d nodes@." (node_count g);
  report ~fold:false g;
  report ~fold:true g;
  [%expect
    {|
    before: 6 nodes
    fold=false: 2 nodes, first run changed=true, second changed=false
    fold=true: 2 nodes, first run changed=true, second changed=false |}]

(* [fold] is retained only for source compatibility. The two calls must produce
   the same canonical graph even when a future fixture gives the pipeline
   symbolic constants to fold. *)
let%expect_test
    "pipeline: the compatibility fold flag does not choose structure" =
  let g = mixed () in
  let printed fold =
    match twice ~fold g with
    | Error _ -> "failed"
    | Ok (g, _, _) -> Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "same graph: %b@." (String.equal (printed false) (printed true));
  [%expect {| same graph: true |}]
