(* [Storage_run]: arenas for every storage role. Each check here is one a broken
   ownership rule would turn red: results equal a release-only run's, a live
   lease is never overwritten, constants are never poisoned or reclaimed, and
   a failed run publishes nothing. *)

open Graph_ir
module S = Storage_script
module E = Eval_direct_arena_test

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let pp_error ppf e = Storage_run.pp_error ppf (Err.Error.kind e)
let ok r = Err.or_raise ~pp_error:Storage_run.pp_error r
let config layout ~constants ~inputs = { S.Config.layout; constants; inputs }

let separate_copied =
  config S.Layout.Separate ~constants:S.Ownership.Copied
    ~inputs:S.Ownership.Copied

let shared_copied =
  config S.Layout.Shared_execution ~constants:S.Ownership.Copied
    ~inputs:S.Ownership.Copied

let configs =
  [
    separate_copied;
    shared_copied;
    config S.Layout.Separate ~constants:S.Ownership.Borrowed
      ~inputs:S.Ownership.Borrowed;
    config S.Layout.Shared_execution ~constants:S.Ownership.Borrowed
      ~inputs:S.Ownership.Copied;
  ]

let plan ?(retain = only_empty) config g =
  let script =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.storage_script ~retain config g)
  in
  Err.or_raise
    ~pp_error:(fun ppf e -> Arena_run.pp_error ppf (e :> Arena_run.error))
    (Storage_plan.create script)

let prepare ?constants p g =
  let constants = Option.value constants ~default:(E.bound g Input.Constant) in
  Err.or_raise
    ~pp_error:(fun ppf e ->
      Storage_run.pp_error ppf
        (match e with
        | #Arena.error as e -> (e :> Storage_run.error)
        | `Missing_constant _ as e -> e
        | #Tensor.dst_error as e -> (e :> Storage_run.error)))
    (Constant_arena.create p ~constants)

let runner ?poison ?max_outstanding ?constants p g =
  ok (Storage_run.create ?poison ?max_outstanding p (prepare ?constants p g))

let run ?(retain = only_empty) ?inputs r g =
  Storage_run.run r ~retain g
    ~inputs:(Option.value inputs ~default:(E.bound g Input.Input))

let outputs lease =
  Err.or_raise
    ~pp_error:(fun ppf `Lease_released -> Fmt.string ppf "released")
    (Result_lease.with_outputs lease Fun.id)

(* Outputs bit-identical to a release-only run, or the same error. *)
let agrees name g lease_result =
  let reference = E.run g in
  match (reference, lease_result) with
  | Ok a, Ok lease ->
      let b = outputs lease in
      Result_lease.release lease;
      if
        List.for_all
          (fun id ->
            Tensor.equal_bits (Tensor_id.Map.find id a)
              (Tensor_id.Map.find id b))
          g.Graph.outputs
      then None
      else Some (name ^ ": OUTPUTS DIFFER")
  | Error a, Error b ->
      let a = Fmt.str "%a" E.pp_error a and b = Fmt.str "%a" pp_error b in
      if String.equal a b then None else Some (name ^ ": ERRORS DIFFER")
  | Ok _, Error e -> Some (Fmt.str "%s: FAILED: %a" name pp_error e)
  | Error _, Ok _ -> Some (name ^ ": ONE FAILED")

let%expect_test "every layout and ownership equals a release run" =
  let bad = ref 0 and runs = ref 0 in
  List.iter
    (fun config ->
      List.iter
        (fun (name, build) ->
          let g = build () in
          let r = runner ~poison:Arena.Poison.A (plan config g) g in
          incr runs;
          match agrees name g (run r g) with
          | None -> ()
          | Some msg ->
              incr bad;
              Fmt.pr "%a %s@." S.Config.pp config msg)
        (("roles", Storage_script_test.roles) :: Graph_fixtures.all))
    configs;
  Fmt.pr "%d runs, %d differ@." !runs !bad;
  [%expect {| 184 runs, 0 differ |}]

(* Every cell a run reads was written by it: two poisons, same bits. *)
let%expect_test "paired poisons agree" =
  let bad = ref 0 in
  List.iter
    (fun (name, build) ->
      let g = build () in
      let p = plan shared_copied g in
      let a = run (runner ~poison:Arena.Poison.A p g) g
      and b = run (runner ~poison:Arena.Poison.B p g) g in
      match (a, b) with
      | Ok a, Ok b ->
          let a = outputs a and b = outputs b in
          if
            not
              (List.for_all
                 (fun id ->
                   Tensor.equal_bits (Tensor_id.Map.find id a)
                     (Tensor_id.Map.find id b))
                 g.Graph.outputs)
          then begin
            incr bad;
            Fmt.pr "%s: POISON VISIBLE@." name
          end
      | _ -> ())
    Graph_fixtures.all;
  Fmt.pr "%d differ@." !bad;
  [%expect {| 0 differ |}]

let roles () = Storage_script_test.roles ()

(* [E.bound]'s float values of [kind], each mapped through [f]. *)
let bound_with f (g : graph) kind =
  List.map
    (fun (id, _) ->
      let shape = (Tensor_id.Map.find id g.Graph.tensors).Tensor_sig.shape in
      ( id,
        Tensor.materialize shape (fun c ->
            f (float_of_int (((Vec6.offset shape c :> int) mod 7) - 3) /. 4.))
      ))
    (E.bound g kind)

let values lease =
  Tensor_id.Map.bindings (outputs lease)
  |> List.map (fun (id, t) -> Fmt.str "%a=%a" Tensor_id.pp id Tensor.pp t)
  |> String.concat " "

(* With one result arena, a second run while a result is live is refused; the
   first result is intact, and after its release the arena is reused. With two,
   the second run gets the other arena and the first result is still intact:
   returning from a run never licenses overwriting its results. *)
let%expect_test "a live result is never overwritten" =
  let g = roles () in
  let other = bound_with (fun v -> v +. 10.) g Input.Input in
  List.iter
    (fun config ->
      List.iter
        (fun max_outstanding ->
          let r =
            runner ~poison:Arena.Poison.B ~max_outstanding (plan config g) g
          in
          let first = ok (run r g) in
          let before = values first in
          let second = run ~inputs:other r g in
          Fmt.pr
            "%a, %Ld result arena(s): second run %s, first %s, %Ld \
             outstanding@."
            S.Layout.pp config.S.Config.layout max_outstanding
            (match Err.payload second with
            | Ok _ -> "ran"
            | Error e -> Fmt.str "refused (%a)" Storage_run.pp_error e)
            (if String.equal before (values first) then "intact"
             else "OVERWRITTEN")
            (Storage_run.outstanding r);
          Result_lease.release first;
          (match Err.payload second with
          | Ok l -> Result_lease.release l
          | Error _ -> ());
          Fmt.pr "  after release: %s, %Ld outstanding@."
            (match Err.payload (run r g) with
            | Ok l ->
                Result_lease.release l;
                "ran"
            | Error e -> Fmt.str "refused (%a)" Storage_run.pp_error e)
            (Storage_run.outstanding r))
        [ 1L; 2L ])
    [ separate_copied; shared_copied ];
  [%expect
    {|
    separate, 1 result arena(s): second run refused (arena: already in use by a run), first intact, 1 outstanding
      after release: ran, 0 outstanding
    separate, 2 result arena(s): second run ran, first intact, 2 outstanding
      after release: ran, 0 outstanding
    shared_execution, 1 result arena(s): second run refused (arena: already in use by a run), first intact, 1 outstanding
      after release: ran, 0 outstanding
    shared_execution, 2 result arena(s): second run ran, first intact, 2 outstanding
      after release: ran, 0 outstanding |}]

let%expect_test "copy-out releases the lease, and scoped access then fails" =
  let g = roles () in
  let r = runner (plan separate_copied g) g in
  let lease = ok (run r g) in
  let before = values lease in
  let copies, copied =
    Err.or_raise
      ~pp_error:(fun ppf -> function
        | `Lease_released as e -> Result_lease.pp_lease_released ppf e
        | (`Quant_missing _ | #Tensor.dst_error) as e ->
            Storage_run.pp_error ppf e)
      (Result_lease.copy_out lease)
  in
  Fmt.pr "copied %Ld results, %a bytes; released %b, %Ld outstanding@."
    copied.Arena.Copies.count Core.Storage_units.Byte_size.pp
    copied.Arena.Copies.bytes
    (Result_lease.released lease)
    (Storage_run.outstanding r);
  (* The next run may now overwrite the arena; the copies stay. *)
  let l2 = ok (run ~inputs:(bound_with (fun v -> -.v) g Input.Input) r g) in
  Fmt.pr "copies intact: %b@."
    (String.equal before
       (Tensor_id.Map.bindings copies
       |> List.map (fun (id, t) -> Fmt.str "%a=%a" Tensor_id.pp id Tensor.pp t)
       |> String.concat " "));
  Result_lease.release l2;
  Fmt.pr "%a@."
    Fmt.(result ~ok:(any "accessible") ~error:(any "released"))
    (Err.payload (Result_lease.with_outputs lease ignore));
  [%expect
    {|
    copied 4 results, 56 bytes; released true, 0 outstanding
    copies intact: true
    released |}]

(* Constants are prepared once, read by every run, never poisoned: a replaced
   model is a new version in new storage, and a result forwarding an old
   constant still reads the old values. *)
let%expect_test "constant versions outlive their model" =
  let g = roles () in
  let p = plan separate_copied g in
  let v1 = E.bound g Input.Constant in
  let r1 = runner ~poison:Arena.Poison.A ~constants:v1 p g in
  let a = ok (run r1 g) and _ = () in
  let a_values = values a in
  Result_lease.release a;
  let b = ok (run r1 g) in
  Fmt.pr "repeated run equal: %b@." (String.equal a_values (values b));
  let v2 = bound_with (fun v -> v *. 3.) g Input.Constant in
  let r2 = runner ~constants:v2 p g in
  let c = ok (run r2 g) in
  Fmt.pr "v1 lease intact after v2 ran: %b; versions differ: %b@."
    (String.equal a_values (values b))
    (not
       (Constant_arena.Generation.equal
          (Constant_arena.generation (Result_lease.constants b))
          (Constant_arena.generation (Result_lease.constants c))));
  Fmt.pr "v2 differs: %b@." (not (String.equal a_values (values c)));
  [%expect
    {|
    repeated run equal: true
    v1 lease intact after v2 ran: true; versions differ: true
    v2 differs: true |}]

let%expect_test "a failed run publishes nothing and holds nothing" =
  let g = E.fails_midway () in
  List.iter
    (fun config ->
      let r = runner (plan config g) g in
      Fmt.pr "%a: %a; %Ld outstanding; again: %a@." S.Layout.pp
        config.S.Config.layout
        Fmt.(result ~ok:(any "ran") ~error:pp_error)
        (run r g)
        (Storage_run.outstanding r)
        Fmt.(result ~ok:(any "ran") ~error:pp_error)
        (run r g))
    [ separate_copied; shared_copied ];
  [%expect
    {|
    separate: to_copy: Long target has no exact I64 output for a f16 source; 0 outstanding; again: to_copy: Long target has no exact I64 output for a f16 source
    shared_execution: to_copy: Long target has no exact I64 output for a f16 source; 0 outstanding; again: to_copy: Long target has no exact I64 output for a f16 source |}]

(* A copied input is copied into its slot once per run; a borrowed one is
   read in place. Neither run changes the caller's input. *)
let%expect_test "copied and borrowed inputs" =
  let g = roles () in
  List.iter
    (fun inputs ->
      let config =
        config S.Layout.Separate ~constants:S.Ownership.Copied ~inputs
      in
      let r = runner (plan config g) g in
      let given = E.bound g Input.Input in
      let snapshot =
        List.map
          (fun (id, t) -> Fmt.str "%a=%a" Tensor_id.pp id Tensor.pp t)
          given
      in
      let l = ok (run ~inputs:given r g) in
      Result_lease.release l;
      let report = ok (Storage_run.report r) in
      Fmt.pr "%a: input copies %Ld (%a bytes); caller's inputs unchanged %b@."
        S.Ownership.pp inputs report.Storage_run.Report.input_copies.count
        Core.Storage_units.Byte_size.pp report.input_copies.bytes
        (snapshot
        = List.map
            (fun (id, t) -> Fmt.str "%a=%a" Tensor_id.pp id Tensor.pp t)
            given))
    [ S.Ownership.Borrowed; S.Ownership.Copied ];
  [%expect
    {|
    borrowed: input copies 0 (0 bytes); caller's inputs unchanged true
    copied: input copies 2 (24 bytes); caller's inputs unchanged true |}]

(* Every role's slots start at their alignment, under a host request too. *)
let%expect_test "alignment for every role" =
  let module B = Core.Storage_units.Byte_offset in
  let g = roles () in
  let host = Alignment_policy.with_host Alignment_policy.page_alignment in
  let script =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.storage_script ~alignment:host ~retain:only_empty
         separate_copied g)
  in
  let p =
    Err.or_raise
      ~pp_error:(fun ppf e -> Arena_run.pp_error ppf (e :> Arena_run.error))
      (Storage_plan.create script)
  in
  List.iter
    (fun (id, ap) ->
      Fmt.pr "%a:" S.Arena_id.pp id;
      List.iter
        (fun (s : Arena_plan.Slot.t) ->
          Fmt.pr " %a@@%a%s" Tensor_id.pp s.id B.pp s.offset
            (if B.is_aligned s.offset Alignment_policy.page_alignment then ""
             else " MISALIGNED"))
        (Arena_plan.slots ap);
      Fmt.pr "@.")
    (Storage_plan.arenas p);
  let r = runner p g in
  Fmt.pr "%a@."
    Fmt.(result ~ok:(any "ran") ~error:pp_error)
    (Err.map Result_lease.release (run r g));
  [%expect
    {|
    constants: t1@0
    inputs: t2@0
    intermediates: t3@0 t4@4096
    outputs: t0@4096 t5@0 t6@8192
    ran |}]

(* Shared execution lets dead inputs and intermediates back later outputs; the
   figures are reported, not assumed to shrink. *)
let%expect_test "separate and shared footprints" =
  List.iter
    (fun name ->
      let g = (List.assoc name Graph_fixtures.all) () in
      List.iter
        (fun config ->
          let r = runner (plan config g) g in
          Result_lease.release (ok (run r g));
          let report = ok (Storage_run.report r) in
          (* Not the whole report: its constant version is a global count. *)
          Fmt.pr "%s %a: %a; result arena %a bytes@." name S.Layout.pp
            config.S.Config.layout Storage_plan.Footprint.pp
            report.Storage_run.Report.footprint Core.Storage_units.Byte_size.pp
            report.result_arena_bytes)
        [ separate_copied; shared_copied ])
    [ "residual"; "chain" ];
  [%expect
    {|
    residual separate: constant_bytes=0 execution_bytes=112 borrowed_bytes=0 outside_bytes=0; result arena 16 bytes
    residual shared_execution: constant_bytes=0 execution_bytes=144 borrowed_bytes=0 outside_bytes=0; result arena 144 bytes
    chain separate: constant_bytes=396 execution_bytes=472 borrowed_bytes=0 outside_bytes=0; result arena 108 bytes
    chain shared_execution: constant_bytes=396 execution_bytes=236 borrowed_bytes=0 outside_bytes=0; result arena 236 bytes |}]
