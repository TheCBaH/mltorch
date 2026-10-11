(* The ordinary graph runner over Native4D: a body is a graph, run by the
   four-axis direct evaluator on the tensors imported for its inputs. *)

open Inference_compare_schema
module Tid = Graph_ir.Tensor_id

type error = Graph_runner.error

let pp_error = Graph_runner.pp_error

let identity ?artifact () : Engine_identity.t =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  {
    engine = id Id.Engine.of_string "native4d";
    route = id Id.Engine_route.of_string "eval-direct4";
    build_digest =
      Aten_runner.sha_digest Digest.Build.make
        "inference-compare-pt2/native4d/eval-direct4/v1";
    configuration_digest =
      Aten_runner.sha_digest Digest.Configuration.make
        "dot-accumulation=binary64;float-to-int=checked";
    artifact_digest = artifact;
    environment_digest =
      Aten_runner.sha_digest Digest.Environment.make
        (Sys.os_type ^ ":" ^ string_of_int Sys.word_size);
  }

module Runner = Graph_runner.Make (struct
  type op = Native4d.Op.t
  type compiled = unit

  let compile _ = Ok ()

  let execute () (g : Native4d.Graph.graph) ~inputs =
    match Native4d.Eval_direct4.run g ~inputs with
    | Error e ->
        Error
          (Format.asprintf "Native4D: %a" Native4d.Eval_direct4.pp_error
             (Err.Error.kind e))
    | Ok env ->
        let rec outs acc = function
          | [] -> Ok (List.rev acc)
          | id :: rest -> (
              match Tid.Map.find_opt id env with
              | None -> Error "a declared output was not produced"
              | Some packed -> outs ((id, packed) :: acc) rest)
        in
        Result.map (List.map snd) (outs [] g.outputs)

  let release () = ()
end)

let create ?artifact () = Runner.create ~identity:(identity ?artifact)
