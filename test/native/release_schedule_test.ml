(* The release schedule on hand-built graphs over a toy op, since the schedule
   is generic in the op type: which edges go after which node, which inputs go
   before the first, what counts as a reader, and the static peak it implies.
   See .ai/ (tensor release). *)

open Graph_common

(* [Pool] stands for the argmax-style ops: its second output is an index. *)
type op = Map of Tensor_id.t list | Pool of Tensor_id.t | Sink of Tensor_id.t

let operands = function Map xs -> xs | Pool x | Sink x -> [ x ]
let is_sink = function Sink _ -> true | Map _ | Pool _ -> false

let is_index_output op output =
  match op with
  | Pool _ -> Output_ordinal.equal output Output_ordinal.one
  | Map _ | Sink _ -> false

let t = Tensor_id.of_int
let ts = List.map t

let node id op outputs =
  { Node.id = Node_id.of_int id; op; outputs = ts outputs }

let c4 = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4

(* Every edge is [C=4]; [fmts] overrides the default F32 per edge. *)
let graph ?(fmts = []) ~inputs ~outputs nodes =
  let ids =
    inputs
    @ List.concat_map
        (fun (n : op Node.t) -> List.map Tensor_id.to_int n.outputs)
        nodes
  in
  let tensors =
    List.fold_left
      (fun m i ->
        let fmt =
          Option.value (List.assoc_opt i fmts)
            ~default:(Payload.Fmt Payload.F32)
        in
        Tensor_id.Map.add (t i)
          (Tensor_sig.create ~id:(t i) ~name:"" ~shape:c4 ~fmt ())
          m)
      Tensor_id.Map.empty ids
  in
  {
    Graph.nodes;
    root = { Group.id = Group_id.of_int 0; label = None; items = [] };
    tensors;
    inputs = ts inputs;
    input_kinds = Tensor_id.Map.empty;
    outputs = ts outputs;
  }

let pp_ids = Fmt.(brackets (list ~sep:sp Tensor_id.pp))

let print ?(retain = Release_schedule.Retain.Only Tensor_id.Set.empty) g =
  let s = Release_schedule.schedule ~operands ~is_sink ~retain g in
  Fmt.pr "@[<v>initial: %a@," pp_ids (Release_schedule.initial s);
  List.iter
    (fun (n : op Node.t) ->
      match Release_schedule.after s n.id with
      | [] -> ()
      | ids -> Fmt.pr "after %a: %a@," Node_id.pp n.id pp_ids ids)
    g.Graph.nodes;
  let unread =
    Tensor_id.Map.fold
      (fun id _ acc ->
        if Release_schedule.has_reader s id then acc else id :: acc)
      g.Graph.tensors []
  in
  Fmt.pr "no reader: %a@]@." pp_ids (List.rev unread)

(* t0 -> n0 -> t1, t0 -> n1 -> t2, (t1, t2) -> n2 -> t3 *)
let diamond () =
  graph ~inputs:[ 0 ] ~outputs:[ 3 ]
    [
      node 0 (Map [ t 0 ]) [ 1 ];
      node 1 (Map [ t 0 ]) [ 2 ];
      node 2 (Map [ t 1; t 2 ]) [ 3 ];
    ]

let%expect_test "an edge read by two nodes goes after the later one" =
  print (diamond ());
  [%expect
    {|
    initial: []
    after n1: [t0]
    after n2: [t1 t2]
    no reader: [] |}]

let%expect_test "an edge read twice by one node is released once" =
  print (graph ~inputs:[ 0 ] ~outputs:[ 1 ] [ node 0 (Map [ t 0; t 0 ]) [ 1 ] ]);
  [%expect {|
    initial: []
    after n0: [t0]
    no reader: [] |}]

let%expect_test "a dead output of a multi-output node goes after its producer" =
  print
    (graph ~inputs:[ 0 ] ~outputs:[ 3 ]
       [ node 0 (Map [ t 0 ]) [ 1; 2 ]; node 1 (Map [ t 1 ]) [ 3 ] ]);
  [%expect
    {|
    initial: []
    after n0: [t2 t0]
    after n1: [t1]
    no reader: [t2] |}]

let%expect_test "a graph output a later node reads is never released" =
  print
    (graph ~inputs:[ 0 ] ~outputs:[ 1; 2 ]
       [ node 0 (Map [ t 0 ]) [ 1 ]; node 1 (Map [ t 1 ]) [ 2 ] ]);
  [%expect {|
    initial: []
    after n0: [t0]
    no reader: [] |}]

(* The shape [Discard] has in [Graph_ir]: it sinks a pool's index output. *)
let discarded_index () =
  graph ~inputs:[ 0 ] ~outputs:[ 3 ]
    [
      node 0 (Pool (t 0)) [ 1; 2 ];
      node 1 (Sink (t 2)) [];
      node 2 (Map [ t 1 ]) [ 3 ];
    ]

let%expect_test "a sink is not a reader" =
  print (discarded_index ());
  [%expect
    {|
    initial: []
    after n0: [t2 t0]
    after n2: [t1]
    no reader: [t2] |}]

let%expect_test "an input read by two nodes goes after the later one" =
  print
    (graph ~inputs:[ 0 ] ~outputs:[ 2 ]
       [ node 0 (Map [ t 0 ]) [ 1 ]; node 1 (Map [ t 0; t 1 ]) [ 2 ] ]);
  [%expect {|
    initial: []
    after n1: [t0 t1]
    no reader: [] |}]

let%expect_test "an input nothing reads goes before the first node" =
  print
    (graph ~inputs:[ 0; 1 ] ~outputs:[ 2 ]
       [ node 0 (Map [ t 0 ]) [ 2 ]; node 1 (Sink (t 1)) [] ]);
  [%expect {|
    initial: [t1]
    after n0: [t0]
    no reader: [t1] |}]

let chain () =
  graph ~inputs:[ 0 ] ~outputs:[ 3 ]
    [
      node 0 (Map [ t 0 ]) [ 1 ];
      node 1 (Map [ t 1 ]) [ 2 ];
      node 2 (Map [ t 2 ]) [ 3 ];
    ]

let%expect_test "Only {x} keeps an intermediate" =
  print ~retain:(Only (Tensor_id.Set.singleton (t 1))) (chain ());
  [%expect
    {|
    initial: []
    after n0: [t0]
    after n2: [t2]
    no reader: [] |}]

let%expect_test "All releases nothing, and still knows the readers" =
  print ~retain:All (discarded_index ());
  print ~retain:All
    (graph ~inputs:[ 0; 1 ] ~outputs:[ 2 ] [ node 0 (Map [ t 0 ]) [ 2 ] ]);
  [%expect
    {|
    initial: []
    no reader: [t2]
    initial: []
    no reader: [t1] |}]

(* ---- peak_bytes ----------------------------------------------------------- *)

let print_peak g =
  List.iter
    (fun (name, retain) ->
      let s = Release_schedule.schedule ~operands ~is_sink ~retain g in
      Fmt.pr "%s: %a@." name
        (Core.Pretty.err_result ~ok:Fmt.int64 ~error:(fun ppf -> function
          | `Missing_tensor_sig id -> Fmt.pf ppf "no sig %a" Tensor_id.pp id
          | `Numel_over_limit b -> Vec6.Numel_bound.pp ppf b
          | `Peak_bytes_overflow id ->
              Fmt.pf ppf "overflow at %a" Tensor_id.pp id))
        (Release_schedule.peak_bytes ~is_index_output g s))
    [
      ("All", Release_schedule.Retain.All);
      ("Only empty", Release_schedule.Retain.Only Tensor_id.Set.empty);
    ]

(* F32 16 bytes, I64 32, Bool 4, all [C=4]. Under [Only]: t0+t1 = 32, then
   t0+t1+t2 = 64 at n1 (t1 still live while n1 runs), t0+t2+t3 = 52 at n2. *)
let%expect_test "peak_bytes over an F32 -> I64 -> Bool chain" =
  print_peak
    (graph
       ~fmts:[ (2, Payload.Fmt Payload.I64); (3, Payload.Fmt Payload.Bool) ]
       ~inputs:[ 0 ] ~outputs:[ 3 ]
       [
         node 0 (Map [ t 0 ]) [ 1 ];
         node 1 (Map [ t 1 ]) [ 2 ];
         node 2 (Map [ t 2 ]) [ 3 ];
       ]);
  [%expect {|
    All: 68
    Only empty: 64 |}]

(* The I64 index nothing reads is never allocated, so it counts under neither
   setting: t0 + t1 + t3 at most. *)
let%expect_test "peak_bytes skips an index output nothing reads" =
  print_peak
    (graph
       ~fmts:[ (2, Payload.Fmt Payload.I64) ]
       ~inputs:[ 0 ] ~outputs:[ 3 ]
       [
         node 0 (Pool (t 0)) [ 1; 2 ];
         node 1 (Sink (t 2)) [];
         node 2 (Map [ t 1 ]) [ 3 ];
       ]);
  [%expect {|
    All: 48
    Only empty: 48 |}]

let%expect_test "peak_bytes counts a read index output" =
  print_peak
    (graph
       ~fmts:[ (2, Payload.Fmt Payload.I64) ]
       ~inputs:[ 0 ] ~outputs:[ 3 ]
       [ node 0 (Pool (t 0)) [ 1; 2 ]; node 1 (Map [ t 1; t 2 ]) [ 3 ] ]);
  [%expect {|
    All: 80
    Only empty: 80 |}]
