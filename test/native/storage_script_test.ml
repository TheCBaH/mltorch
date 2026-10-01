(* [Eval_direct.storage_script]: every block a run touches, by role and arena,
   with the run's boundaries. *)

open Graph_ir
module S = Storage_script

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty

(* [x] is an input and [w] a constant, both read by [a] and both also graph
   outputs; [a] and [b] are intermediates, [c] and [d] (from the second input
   [u]) are outputs. *)
let roles () =
  Graph_fixtures.buildn "roles"
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 4) () in
      let* w = constant ~shape:(Graph_fixtures.s1c 4) () in
      let* u = input ~shape:(Graph_fixtures.s1c 2) () in
      let* a = add x w in
      let* b = relu a in
      let* c = relu b in
      let* d = relu u in
      return [ c; x; w; d ])

let config layout ~constants ~inputs = { S.Config.layout; constants; inputs }

let script ?(retain = only_empty) config g =
  Err.or_raise ~pp_error:Eval_direct.pp_error
    (Eval_direct.storage_script ~retain config g)

let peak s where =
  Err.or_raise
    ~pp_error:(fun ppf (`Peak_bytes_overflow id) -> Tensor_id.pp ppf id)
    (S.peak_bytes s ~where)

let in_arena id (b : S.Block.t) =
  Option.equal S.Arena_id.equal b.S.Block.arena (Some id)

let show s =
  Fmt.pr "%a@." S.pp s;
  List.iter
    (fun id ->
      Fmt.pr "peak %a: %a@." S.Arena_id.pp id Core.Storage_units.Byte_size.pp
        (peak s (in_arena id)))
    S.Arena_id.all

let%expect_test "separate arenas, everything copied" =
  show
    (script
       (config S.Layout.Separate ~constants:S.Ownership.Copied
          ~inputs:S.Ownership.Copied)
       (roles ()));
  [%expect
    {|
    layout=separate constants=copied inputs=copied; policy standard
    -- model init
    alloc constant t1 float32 16 bytes align 64 in constants
    -- input population
    alloc input t0 float32 16 bytes align 64 in outputs
    alloc input t2 float32 8 bytes align 64 in inputs
    node n0
    alloc intermediate t3 float32 16 bytes align 64 in intermediates
    node n1
    alloc intermediate t4 float32 16 bytes align 64 in intermediates
    free t3
    node n2
    alloc output t5 float32 16 bytes align 64 in outputs
    free t4
    node n3
    alloc output t6 float32 8 bytes align 64 in outputs
    free t2
    -- result publication
    -- result release
    free t0
    free t5
    free t6
    peak constants: 16
    peak execution: 0
    peak inputs: 8
    peak intermediates: 32
    peak outputs: 40 |}]

let%expect_test "shared execution, borrowed inputs" =
  show
    (script
       (config S.Layout.Shared_execution ~constants:S.Ownership.Copied
          ~inputs:S.Ownership.Borrowed)
       (roles ()));
  [%expect
    {|
    layout=shared_execution constants=copied inputs=borrowed; policy standard
    -- model init
    alloc constant t1 float32 16 bytes align 64 in constants
    -- input population
    alloc input t0 float32 16 bytes align 64 in no arena
    alloc input t2 float32 8 bytes align 64 in no arena
    node n0
    alloc intermediate t3 float32 16 bytes align 64 in execution
    node n1
    alloc intermediate t4 float32 16 bytes align 64 in execution
    free t3
    node n2
    alloc output t5 float32 16 bytes align 64 in execution
    free t4
    node n3
    alloc output t6 float32 8 bytes align 64 in execution
    -- result publication
    -- result release
    free t5
    free t6
    peak constants: 16
    peak execution: 32
    peak inputs: 0
    peak intermediates: 0
    peak outputs: 0 |}]

(* The whole script is compared, config and policy first. *)
let%expect_test "scripts differ by config, retain and policy" =
  let g = roles () in
  let separate =
    config S.Layout.Separate ~constants:S.Ownership.Copied
      ~inputs:S.Ownership.Copied
  in
  let shared = { separate with S.Config.layout = S.Layout.Shared_execution } in
  let pos a b =
    Fmt.pr "%a@."
      (Fmt.option ~none:(Fmt.any "equal") Alloc_script.Position.pp)
      (S.first_difference a b)
  in
  pos (script separate g) (script separate g);
  pos (script separate g) (script shared g);
  pos (script separate g)
    (script ~retain:Release_schedule.Retain.All separate g);
  [%expect {|
    equal
    @0
    @4 |}]

(* A retained intermediate is an output: it lives to the result's release. *)
let%expect_test "a retained intermediate" =
  let g = roles () in
  let s =
    script
      ~retain:
        (Release_schedule.Retain.Only
           (Tensor_id.Set.singleton (Tensor_id.of_int 3)))
      (config S.Layout.Separate ~constants:S.Ownership.Borrowed
         ~inputs:S.Ownership.Borrowed)
      g
  in
  List.iter
    (function
      | S.Event.Alloc { S.Block.alloc; role; arena } ->
          Fmt.pr "%a %a %a@." Tensor_id.pp alloc.Alloc_script.Alloc.id S.Role.pp
            role
            (Fmt.option ~none:(Fmt.any "-") S.Arena_id.pp)
            arena
      | S.Event.Free id -> Fmt.pr "free %a@." Tensor_id.pp id
      | S.Event.Boundary b -> Fmt.pr "-- %a@." S.Boundary.pp b
      | S.Event.Node _ -> ())
    (S.events s);
  [%expect
    {|
    -- model init
    t1 constant -
    -- input population
    t0 input -
    t2 input -
    t3 output outputs
    t4 intermediate intermediates
    t5 output outputs
    free t4
    t6 output outputs
    -- result publication
    -- result release
    free t3
    free t5
    free t6 |}]
