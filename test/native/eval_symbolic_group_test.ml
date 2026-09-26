(* Project step 19, Section C wrapper migration: [Eval_symbolic.run] builds
   ONE [Region_group.t] per multi-output Region-authored node (today: only
   Lstm) and hands every sibling stage a [Grouped] reference into it,
   instead of each independently building its own projected program (see
   lib/native/eval_symbolic.ml's [process_node]). This pins the two
   structural claims that migration makes: all three of one Lstm node's
   stages share one physically-identical [Region_group.t] instance, and two
   distinct Lstm nodes -- even with an identical configuration -- get two
   distinct instances, never merged or cached (design record §3.1's
   "grouping identity is scoped to one node invocation"). *)

let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let k = 2 (* hidden_size *)
let isz = 2 (* input width *)
let seq = 2
let batch = 1
let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:k
let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

let params : Lstm.Lstm.params =
  { hidden_size = k; input_size = isz; batch_first = false }

(* One Lstm node (single layer, single direction) reading fresh graph
   inputs; returns its three output edge ids. *)
let lstm_node () =
  Graph_builder.(
    let* input_id = input ~shape:seq_shape ~name:"input" () in
    let* wih_id =
      input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"weight_ih" ()
    in
    let* whh_id =
      input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"weight_hh" ()
    in
    let* h0_id = input ~shape:state_shape ~name:"h0" () in
    let* c0_id = input ~shape:state_shape ~name:"c0" () in
    let layer : Lstm.Lstm.Layer.t =
      {
        forward = { weight_ih = wih_id; weight_hh = whh_id; bias = None };
        reverse = None;
      }
    in
    lstm params ~input:input_id ~layers:[ layer ] ~h0:h0_id ~c0:c0_id ())

let lstm_node_outputs (g : Graph_ir.graph) =
  List.filter_map
    (fun (n : Graph_ir.node) ->
      match n.Graph_ir.Node.op with
      | Graph_ir.Lstm _ -> Some n.Graph_ir.Node.outputs
      | _ -> None)
    g.Graph_ir.Graph.nodes

let group_of (prog : Stage_program.t) id =
  let stage =
    List.find
      (fun (st : Stage_program.Stage.t) -> Tensor_id.equal st.id id)
      prog.Stage_program.stages
  in
  match stage.computation with
  | Region_group.Ref.Grouped (g, ordinal) -> (g, ordinal)
  | Region_group.Ref.Solo _ ->
      invalid_arg "eval_symbolic_group_test: expected a Grouped stage"

let%expect_test
    "Eval_symbolic: one Lstm node's three stages share one Region_group.t \
     instance" =
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      (Graph_builder.build ~name:"one_lstm"
         ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
         (lstm_node ()))
  in
  let prog = Eval_symbolic.run g in
  let outs =
    match lstm_node_outputs g with [ outs ] -> outs | _ -> assert false
  in
  let groups_and_ordinals = List.map (group_of prog) outs in
  let groups = List.map fst groups_and_ordinals in
  let ordinals = List.map snd groups_and_ordinals in
  let same_instance =
    match groups with [ a; b; c ] -> a == b && b == c | _ -> false
  in
  Fmt.pr "same group instance across all three stages: %b@." same_instance;
  Fmt.pr "ordinals in emitter order: %s@."
    (String.concat ","
       (List.map
          (fun (o : Region_group.Ordinal.t) -> string_of_int (o :> int))
          ordinals));
  [%expect
    {|
    same group instance across all three stages: true
    ordinals in emitter order: 0,1,2 |}]

let%expect_test
    "Eval_symbolic: two distinct Lstm nodes get two distinct Region_group.t \
     instances" =
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      (Graph_builder.build ~name:"two_lstm"
         ~outputs:(fun ((out1, hn1, cn1), (out2, hn2, cn2)) ->
           [ out1; hn1; cn1; out2; hn2; cn2 ])
         Graph_builder.(
           let* first = lstm_node () in
           let* second = lstm_node () in
           return (first, second)))
  in
  let prog = Eval_symbolic.run g in
  let outs =
    match lstm_node_outputs g with
    | [ outs1; outs2 ] -> (outs1, outs2)
    | _ -> assert false
  in
  let first_group = fst (group_of prog (List.hd (fst outs))) in
  let second_group = fst (group_of prog (List.hd (snd outs))) in
  Fmt.pr "distinct group instances: %b@." (first_group != second_group);
  [%expect {| distinct group instances: true |}]
