(* The Native4D twin of eval_direct_gc_test.ml. [Eval_direct4] has no per-node
   seam to stash an intermediate from, so this watches the one tensor the test
   can hand over without keeping: an input only the env holds. Under [Only] it
   leaves the env after its last reader and is garbage once the run returns;
   under [All] the result still holds it. See .ai/ (tensor release). *)

open Native4d

let shape = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:4

let graph () =
  Builder.build
    ~outputs:(fun o -> [ o ])
    (let open Builder in
     let* x = input ~shape () in
     let* a = relu x in
     sqrt a)
  |> Err.or_raise ~pp_error:Builder.pp_error

let collect () =
  Gc.full_major ();
  Gc.full_major ()

let input_alive_after retain =
  let g = graph () in
  let slot = Weak.create 1 in
  (* Built and handed over inside a function, so nothing on this frame keeps a
     reference once [run] has it. *)
  let run () =
    let x = Tensor.materialize (Shape4.to_vec6 shape) (fun _ -> 4.) in
    Weak.set slot 0 (Some x);
    Eval_direct4.run ~retain g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ]
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let env = (Sys.opaque_identity run) () in
  collect ();
  let alive = Weak.check slot 0 in
  ignore (Sys.opaque_identity env);
  alive

let%expect_test "Only empty: an input only the env held is collectable" =
  Fmt.pr "alive after: %b@."
    (input_alive_after (Release_schedule.Retain.Only Tensor_id.Set.empty));
  [%expect {| alive after: false |}]

let%expect_test "All: the result keeps the input" =
  Fmt.pr "alive after: %b@." (input_alive_after Release_schedule.Retain.All);
  [%expect {| alive after: true |}]
