open Graph_ir
open Loop_ir

(* A failing kernel through a SSA-made C kernel: an embedding whose index is
   past the table, run by the host binary, must report the failing invocation
   and the interpreter's row, under the direct frontend and under the
   structured one alike. The record names its site by index, so this is where
   the site table (made from the optimized Loop program) meets the converted
   program's own failure order. *)

let vocab = 5
let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:vocab ~c:3

let graph indices_shape =
  Graph_builder.build ~name:"embedding_failure"
    ~outputs:(fun o -> [ o ])
    Graph_builder.(
      let* weight = input ~shape:weight_shape ~name:"weight" () in
      let* indices =
        constant ~shape:indices_shape ~fmt:Payload.(Fmt I64) ~name:"indices" ()
      in
      embedding
        {
          Embedding.Embedding.indices_rank = Rank.of_int 1;
          padding_idx = Aten_int.Index.of_int (-1);
        }
        ~weight ~indices)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let i64_tensor shape cells =
  let data = Bigarray.(Array1.of_array int64 c_layout cells) in
  Tensor.Tensor
    {
      Tensor.shape;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

let weight =
  Tensor.materialize weight_shape (fun c ->
      float_of_int
        ((10 * Dim.to_int (Vec6.get c Axis.W)) + Dim.to_int (Vec6.get c Axis.C)))

let run ~frontend cells =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:(Array.length cells) in
  let g = graph shape in
  let indices = i64_tensor shape cells in
  let dir = Loop_c_exec.Proc.temp_dir "ssa_failure_host" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () ->
      let b =
        Err.or_raise ~pp_error:Loop_bundle.pp_error
          (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
      in
      let sites =
        List.fold_left
          (fun n (inv : Loop_bundle.invocation) ->
            n + Array.length (Loop_js_failure.sites inv.Loop_bundle.program))
          0 b.Loop_bundle.invocations
      in
      let weight_id, indices_id =
        match (b.Loop_bundle.inputs, b.Loop_bundle.constants) with
        | [ w ], [ i ] -> (w, i)
        | _ -> failwith "unexpected embedding graph shape"
      in
      let result =
        match
          Err.payload
            (Loop_c_exec.Host.prepare
               ~kernel:
                 (Ssa_backends.c_via ~frontend
                    ~pipeline:Ssa_backends.Pipeline.Exact) ~dir b
               ~constants:(fun id ->
                 if Tensor_id.equal id indices_id then Some indices else None))
        with
        | Error e -> Fmt.str "prepare: %a" Loop_c_exec.Host.pp_error e
        | Ok p -> (
            match
              Err.payload
                (Loop_c_exec.Host.run p ~bind:(fun id ->
                     if Tensor_id.equal id weight_id then Some weight else None))
            with
            | Ok _ -> "ran"
            | Error e -> Fmt.str "%a" Loop_c_exec.Host.pp_error e)
      in
      (sites, result))

let%expect_test
    "an index past the table fails the same way from either frontend" =
  List.iter
    (fun (name, cells) ->
      List.iter
        (fun frontend ->
          let sites, result = run ~frontend cells in
          Fmt.pr "%s, %s: %d site(s): %s@." name
            (Ssa_backends.Frontend.name frontend)
            sites result)
        [ Ssa_backends.Frontend.Direct; Ssa_backends.Frontend.Structured ])
    [ ("in range", [| 3L; 0L; 4L |]); ("past the table", [| 3L; 9L; 4L |]) ];
  [%expect
    {|
    in range, direct: 1 site(s): ran
    in range, structured: 1 site(s): ran
    past the table, direct: 1 site(s): inference failed in invocation 0: gather index 9 out of range [-5, 4]
    past the table, structured: 1 site(s): inference failed in invocation 0: gather index 9 out of range [-5, 4] |}]
