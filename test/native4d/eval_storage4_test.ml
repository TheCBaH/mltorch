(* [Eval_storage4.storage_script] must be [Eval_direct.storage_script] on the
   equivalent Native graph: same events in the same order, same blocks, same
   arenas, under every layout/ownership combination. Both graphs are built with
   one tensor-id sequence, so ids line up. The graph covers what a script has
   to get right: a used constant, an UNUSED one (no block), an input forwarded
   as a graph output (one block, longest lifetime), a repeated operand, and a
   two-output node whose index output is either live or dead. *)

open Native4d

let vec6 = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:4 ~c:1
let shape4 = Shape4.of_ints ~n:1 ~h:2 ~w:4 ~c:1

let native ~live_index =
  Graph_builder.build ~name:"script"
    ~outputs:(fun (y, m, i) -> if live_index then [ y; m; i ] else [ y; m ])
    Graph_builder.(
      let* x = input ~shape:vec6 () in
      let* y = input ~shape:vec6 () in
      let* c = constant ~shape:vec6 () in
      let* _unused = constant ~shape:vec6 () in
      let* a = add x c in
      let* b = mul a a in
      let* m, i = max_dim { Reduce.MaxDim.axis = Axis.W; keepdim = true } b in
      return (y, m, i))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let native4d ~live_index =
  Builder.build
    ~outputs:(fun (y, m, i) -> if live_index then [ y; m; i ] else [ y; m ])
    Builder.(
      let* x = input ~shape:shape4 () in
      let* y = input ~shape:shape4 () in
      let* c = constant ~shape:shape4 () in
      let* _unused = constant ~shape:shape4 () in
      let* a = add x c in
      let* b = mul a a in
      let* mi =
        max_dim4 { Ops4_max_dim.Max_dim4.axis = Axis4.W; keepdim = true } b
      in
      return (y, List.nth mi 0, List.nth mi 1))
  |> Err.or_raise ~pp_error:Builder.pp_error

let configs =
  let open Storage_script in
  List.concat_map
    (fun layout ->
      List.concat_map
        (fun constants ->
          List.map
            (fun inputs -> { Config.layout; constants; inputs })
            Ownership.[ Borrowed; Copied ])
        Ownership.[ Borrowed; Copied ])
    Layout.[ Separate; Shared_execution ]

let script_n ?retain ~live_index config =
  Eval_direct.storage_script ?retain config (native ~live_index)
  |> Err.or_raise ~pp_error:Eval_direct.pp_error

let script_4 ?retain ~live_index config =
  Eval_storage4.storage_script ?retain config (native4d ~live_index)
  |> Err.or_raise ~pp_error:Eval_direct4.pp_error

let events_equal a b =
  List.equal Storage_script.Event.equal (Storage_script.events a)
    (Storage_script.events b)

let%expect_test "agrees with Native under every config, live and dead index" =
  List.iter
    (fun (retain_name, retain) ->
      List.iter
        (fun live_index ->
          List.iter
            (fun config ->
              let n = script_n ~retain ~live_index config
              and d = script_4 ~retain ~live_index config in
              Fmt.pr "%s live_index=%b %a: events equal %b, %d events@."
                retain_name live_index Storage_script.Config.pp config
                (events_equal n d)
                (List.length (Storage_script.events d)))
            configs)
        [ true; false ])
    [
      ("all", Release_schedule.Retain.All);
      ("none", Release_schedule.Retain.Only Tensor_id.Set.empty);
    ];
  [%expect
    {|
    all live_index=true layout=separate constants=borrowed inputs=borrowed: events equal true, 18 events
    all live_index=true layout=separate constants=borrowed inputs=copied: events equal true, 20 events
    all live_index=true layout=separate constants=copied inputs=borrowed: events equal true, 18 events
    all live_index=true layout=separate constants=copied inputs=copied: events equal true, 20 events
    all live_index=true layout=shared_execution constants=borrowed inputs=borrowed: events equal true, 18 events
    all live_index=true layout=shared_execution constants=borrowed inputs=copied: events equal true, 20 events
    all live_index=true layout=shared_execution constants=copied inputs=borrowed: events equal true, 18 events
    all live_index=true layout=shared_execution constants=copied inputs=copied: events equal true, 20 events
    all live_index=false layout=separate constants=borrowed inputs=borrowed: events equal true, 16 events
    all live_index=false layout=separate constants=borrowed inputs=copied: events equal true, 18 events
    all live_index=false layout=separate constants=copied inputs=borrowed: events equal true, 16 events
    all live_index=false layout=separate constants=copied inputs=copied: events equal true, 18 events
    all live_index=false layout=shared_execution constants=borrowed inputs=borrowed: events equal true, 16 events
    all live_index=false layout=shared_execution constants=borrowed inputs=copied: events equal true, 18 events
    all live_index=false layout=shared_execution constants=copied inputs=borrowed: events equal true, 16 events
    all live_index=false layout=shared_execution constants=copied inputs=copied: events equal true, 18 events
    none live_index=true layout=separate constants=borrowed inputs=borrowed: events equal true, 18 events
    none live_index=true layout=separate constants=borrowed inputs=copied: events equal true, 20 events
    none live_index=true layout=separate constants=copied inputs=borrowed: events equal true, 18 events
    none live_index=true layout=separate constants=copied inputs=copied: events equal true, 20 events
    none live_index=true layout=shared_execution constants=borrowed inputs=borrowed: events equal true, 18 events
    none live_index=true layout=shared_execution constants=borrowed inputs=copied: events equal true, 20 events
    none live_index=true layout=shared_execution constants=copied inputs=borrowed: events equal true, 18 events
    none live_index=true layout=shared_execution constants=copied inputs=copied: events equal true, 20 events
    none live_index=false layout=separate constants=borrowed inputs=borrowed: events equal true, 16 events
    none live_index=false layout=separate constants=borrowed inputs=copied: events equal true, 18 events
    none live_index=false layout=separate constants=copied inputs=borrowed: events equal true, 16 events
    none live_index=false layout=separate constants=copied inputs=copied: events equal true, 18 events
    none live_index=false layout=shared_execution constants=borrowed inputs=borrowed: events equal true, 16 events
    none live_index=false layout=shared_execution constants=borrowed inputs=copied: events equal true, 18 events
    none live_index=false layout=shared_execution constants=copied inputs=borrowed: events equal true, 16 events
    none live_index=false layout=shared_execution constants=copied inputs=copied: events equal true, 18 events |}]

let%expect_test "the script's shape, once" =
  Fmt.pr "%a@." Storage_script.pp
    (script_4 ~retain:(Release_schedule.Retain.Only Tensor_id.Set.empty)
       ~live_index:false
       Storage_script.
         {
           Config.layout = Layout.Separate;
           constants = Ownership.Borrowed;
           inputs = Ownership.Copied;
         });
  [%expect
    {|
    layout=separate constants=borrowed inputs=copied; policy standard
    -- model init
    alloc constant t2 float32 32 bytes align 64 in no arena
    -- input population
    alloc input t0 float32 32 bytes align 64 in inputs
    alloc input t1 float32 32 bytes align 64 in outputs
    node n0
    alloc intermediate t4 float32 32 bytes align 64 in intermediates
    free t0
    node n1
    alloc intermediate t5 float32 32 bytes align 64 in intermediates
    free t4
    node n2
    alloc output t6 float32 8 bytes align 64 in outputs
    free t5
    -- result publication
    -- result release
    free t1
    free t6 |}]

(* The retention argument reaches the release schedule: keeping every edge
   moves intermediates to [Output] and delays their [Free]. *)
let%expect_test "retain is honoured" =
  let config = List.hd configs in
  let retained =
    Eval_storage4.storage_script ~retain:Release_schedule.Retain.All config
      (native4d ~live_index:true)
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let released =
    Eval_storage4.storage_script
      ~retain:(Release_schedule.Retain.Only Tensor_id.Set.empty) config
      (native4d ~live_index:true)
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  Fmt.pr "differ: %b@." (not (events_equal retained released));
  [%expect {| differ: true |}]
