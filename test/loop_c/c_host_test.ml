open Graph_ir
open Loop_ir

(* The whole-model C backend against the per-node reference evaluator: the same
   graph, constants and inputs, outputs compared bitwise. *)

let tensor_of ~salt (sg : Tensor_sig.t) =
  let v c =
    float_of_int
      ((((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7) - 3)
    /. 4.
  in
  Tensor.materialize sg.Tensor_sig.shape v

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let sig_of_edge (b : Loop_bundle.t) id =
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id then Some buf.Loop_buffer.sg else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations
  |> Option.get

let map_of l id = List.assoc_opt id l

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "c_host_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

let prepare g dir =
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
  in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let p =
    Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
      (Loop_c_exec.Host.prepare ~dir b ~constants:(map_of constants))
  in
  (b, constants, p)

let check g p b constants ~salt ~poison =
  let inputs =
    List.map
      (fun id -> (id, tensor_of ~salt (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  match Err.payload (Loop_c_exec.Host.run ~poison p ~bind:(map_of inputs)) with
  | Error e -> Fmt.pr "run failed: %a@." Loop_c_exec.Host.pp_error e
  | Ok outs ->
      List.iter2
        (fun id t ->
          Fmt.pr "output t%d identical to the reference: %b@."
            (Tensor_id.to_int id)
            (bits t = bits (Tensor_id.Map.find id reference)))
        g.Graph.outputs outs

let%expect_test "chain: compiled model matches the reference, repeatedly" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let b, constants, p = prepare g dir in
      check g p b constants ~salt:0 ~poison:false;
      check g p b constants ~salt:2 ~poison:true;
      check g p b constants ~salt:5 ~poison:true;
      let st = (Loop_c_exec.Host.bundle_c p).Loop_bundle_c.stats in
      Fmt.pr "%d invocations, %d kernels@." st.Loop_bundle_c.invocations
        st.Loop_bundle_c.distinct_kernels);
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    3 invocations, 3 kernels |}]

(* ---- the process contract: every rejection has its own exit status ---------- *)

module H = Loop_c_exec.Host
module Proc = Loop_c_exec.Proc

let exit_of = function
  | Ok (Proc.Exited n, _) -> Printf.sprintf "exit %d" n
  | Ok (Proc.Signaled s, _) ->
      (* OCaml's own (negative) signal numbering. *)
      if s = Sys.sigkill then "SIGKILL" else Printf.sprintf "signal %d" s
  | Error m -> "not started: " ^ m

let last_line s =
  match List.rev (List.filter (( <> ) "") (String.split_on_char '\n' s)) with
  | l :: _ -> l
  | [] -> ""

let%expect_test "the binary rejects a bad command line and bad payloads" =
  (* A directory name with a space: no shell is involved anywhere. *)
  let base = Proc.temp_dir "c_host_test" in
  let dir = Filename.concat base "with space" in
  Fun.protect ~finally:(fun () -> Proc.remove_tree base) @@ fun () ->
  let g = Native_test.Graph_fixtures.chain () in
  let b, constants, p = prepare g dir in
  ignore constants;
  let inputs = Filename.concat dir "in.bin"
  and outputs = Filename.concat dir "out.bin" in
  let bind =
    map_of
      (List.map
         (fun id -> (id, tensor_of ~salt:1 (sig_of_edge b id)))
         b.Loop_bundle.inputs)
  in
  Err.or_raise ~pp_error:H.pp_error (H.write_inputs p ~bind ~path:inputs);
  let exe = H.executable p in
  let scrub text =
    (* The temporary directory's name is not deterministic. *)
    let rec go s =
      match Str.search_forward (Str.regexp_string base) s 0 with
      | i ->
          String.sub s 0 i ^ "$DIR"
          ^ go
              (String.sub s
                 (i + String.length base)
                 (String.length s - i - String.length base))
      | exception Not_found -> s
    in
    go text
  in
  let report name r =
    Fmt.pr "%s: %s%s@." name (exit_of r)
      (match r with
      | Ok (_, log) -> scrub (" | " ^ last_line log)
      | Error _ -> "")
  in
  let good = H.command p ~inputs ~outputs in
  report "good" (Proc.run good);
  Fmt.pr "output published: %b@." (Sys.file_exists outputs);
  report "no arguments" (Proc.run [ exe ]);
  report "unknown flag" (Proc.run (good @ [ "--bogus" ]));
  report "missing inputs file"
    (Proc.run (H.command p ~inputs:(Filename.concat dir "absent.bin") ~outputs));
  (* Truncated, wrong-role and wrong-identity payloads. *)
  let bytes = Proc.read_file inputs in
  let variant name f =
    let path = Filename.concat dir (name ^ ".bin") in
    Proc.write_file path (f bytes);
    report name (Proc.run (H.command p ~inputs:path ~outputs))
  in
  variant "truncated" (fun s -> String.sub s 0 (String.length s - 1));
  variant "header only" (fun s -> String.sub s 0 10);
  variant "empty" (fun _ -> "");
  variant "wrong role" (fun s ->
      String.mapi (fun i c -> if i = 12 then '\001' else c) s);
  variant "wrong identity" (fun s ->
      String.mapi
        (fun i c -> if i = 40 then Char.chr (Char.code c lxor 1) else c)
        s);
  variant "bad magic" (fun s ->
      String.mapi (fun i c -> if i = 0 then 'X' else c) s);
  (* The output path may not be an input file, and weights are not inputs. *)
  report "output aliases inputs"
    (Proc.run (H.command p ~inputs ~outputs:inputs));
  report "weights as inputs"
    (Proc.run (H.command p ~inputs:(H.weights_path p) ~outputs));
  Fmt.pr "inputs intact: %b@." (Proc.read_file inputs = bytes);
  [%expect
    {|
    good: exit 0 |
    output published: true
    no arguments: exit 2 | usage: $DIR/with space/model --weights FILE --inputs FILE --outputs FILE [--poison] [--time] [--repeat N]
    unknown flag: exit 2 | usage: $DIR/with space/model --weights FILE --inputs FILE --outputs FILE [--poison] [--time] [--repeat N]
    missing inputs file: exit 3 | $DIR/with space/model: cannot open inputs file $DIR/with space/absent.bin: No such file or directory
    truncated: exit 3 | $DIR/with space/model: inputs file $DIR/with space/truncated.bin has 191 bytes, expected 192
    header only: exit 3 | $DIR/with space/model: inputs file $DIR/with space/header only.bin is truncated (10 bytes)
    empty: exit 3 | $DIR/with space/model: inputs file $DIR/with space/empty.bin is truncated (0 bytes)
    wrong role: exit 3 | $DIR/with space/model: inputs file $DIR/with space/wrong role.bin has the wrong payload role
    wrong identity: exit 3 | $DIR/with space/model: inputs file $DIR/with space/wrong identity.bin was made for a different model
    bad magic: exit 3 | $DIR/with space/model: inputs file $DIR/with space/bad magic.bin is not a payload file
    output aliases inputs: exit 3 | $DIR/with space/model: the output file aliases an input file
    weights as inputs: exit 3 | $DIR/with space/model: inputs file $DIR/with space/weights.bin has 512 bytes, expected 192
    inputs intact: true |}]

let%expect_test "compiler failures are typed and leave the sources behind" =
  let dir = Proc.temp_dir "c_host_test" in
  Fun.protect ~finally:(fun () -> Proc.remove_tree dir) @@ fun () ->
  let g = Native_test.Graph_fixtures.chain () in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
  in
  let constants =
    map_of
      (List.map
         (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
         b.Loop_bundle.constants)
  in
  let attempt name compiler =
    (match Err.payload (H.prepare ~compiler ~dir b ~constants) with
    | Ok _ -> Fmt.pr "%s: prepared@." name
    | Error (`Compiler_unavailable _) ->
        Fmt.pr "%s: compiler unavailable@." name
    | Error (`Compile_failed (Proc.Exited n, _)) ->
        Fmt.pr "%s: compile failed, exit %d@." name n
    | Error e -> Fmt.pr "%s: %a@." name H.pp_error e);
    Fmt.pr "  sources kept: %b@."
      (Sys.file_exists (Filename.concat dir "model_infer.c"))
  in
  attempt "missing compiler" [ "no-such-compiler-xyz" ];
  attempt "rejected flag" [ "gcc"; "-fno-such-flag-xyz" ];
  [%expect
    {|
    missing compiler: compiler unavailable
      sources kept: true
    rejected flag: compile failed, exit 1
      sources kept: true |}]

let%expect_test "a child killed by a signal is classified, not decoded" =
  Fmt.pr "%s@." (exit_of (Proc.run [ "sh"; "-c"; "kill -KILL $$" ]));
  Fmt.pr "%s@." (exit_of (Proc.run [ "sh"; "-c"; "exit 7" ]));
  [%expect {|
    SIGKILL
    exit 7 |}]

(* ---- inference failures carry the invocation, not the kernel --------------- *)

let poisoned_bundle b ~at failure =
  let invocations =
    List.mapi
      (fun i (inv : Loop_bundle.invocation) ->
        if i <> at then inv
        else
          let p = inv.Loop_bundle.program in
          let always =
            Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0)
          in
          {
            inv with
            Loop_bundle.program =
              {
                p with
                Loop_program.body =
                  Loop_stmt.Fail_if (always, failure) :: p.Loop_program.body;
              };
          })
      b.Loop_bundle.invocations
  in
  { b with Loop_bundle.invocations }

let%expect_test
    "a failing invocation is reported by position, decoded like the interpreter"
    =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let b0 =
        Err.or_raise ~pp_error:Loop_bundle.pp_error
          (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
      in
      let constants =
        map_of
          (List.map
             (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b0 id)))
             b0.Loop_bundle.constants)
      in
      let bind =
        map_of
          (List.map
             (fun id -> (id, tensor_of ~salt:0 (sig_of_edge b0 id)))
             b0.Loop_bundle.inputs)
      in
      let buffer =
        List.hd
          (List.nth b0.Loop_bundle.invocations 2).Loop_bundle.program
            .Loop_program.buffers
      in
      let cases =
        [
          ("division by zero", 1, Loop_failure.I64_division_by_zero);
          ( "coordinate out of range",
            2,
            Loop_failure.Load_out_of_range
              {
                buffer;
                coord =
                  Expr.Coord.make ~n:(Loop_index.Const 0)
                    ~t:(Loop_index.Const 0) ~d:(Loop_index.Const 0)
                    ~h:(Loop_index.Const 0) ~w:(Loop_index.Const 99)
                    ~c:(Loop_index.Const 0);
              } );
        ]
      in
      List.iter
        (fun (name, at, failure) ->
          let b = poisoned_bundle b0 ~at failure in
          let sub = Filename.concat dir (string_of_int at) in
          let p =
            Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
              (Loop_c_exec.Host.prepare ~dir:sub b ~constants)
          in
          match Err.payload (Loop_c_exec.Host.run p ~bind) with
          | Error (`Inference_failed (i, e)) ->
              Fmt.pr "%s: invocation %d, %a@." name i Loop_interp.pp_error e
          | Error e -> Fmt.pr "%s: %a@." name Loop_c_exec.Host.pp_error e
          | Ok _ -> Fmt.pr "%s: no failure@." name)
        cases);
  [%expect
    {|
    division by zero: invocation 1, I64 division by zero
    coordinate out of range: invocation 2, t8[0,0,0,0,99,0] out of range on axis W: 99 |}]
