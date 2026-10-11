(* The ordinary graph runner over the symbolic evaluators: a body becomes its
   whole-graph stage program, which is then grounded on the tensors imported
   for its inputs. Native and Native4D differ only in how the program is built;
   a body the symbolic form cannot express is a refusal, never a direct run. *)

open Inference_compare_schema
module Tid = Graph_ir.Tensor_id

let identity ~engine ~route ?artifact () : Engine_identity.t =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  {
    engine = id Id.Engine.of_string engine;
    route = id Id.Engine_route.of_string route;
    build_digest =
      Aten_runner.sha_digest Digest.Build.make
        (Printf.sprintf "inference-compare-pt2/%s/%s/v1" engine route);
    configuration_digest =
      Aten_runner.sha_digest Digest.Configuration.make
        "dot-accumulation=binary64;float-to-int=checked";
    artifact_digest = artifact;
    environment_digest =
      Aten_runner.sha_digest Digest.Environment.make
        (Sys.os_type ^ ":" ^ string_of_int Sys.word_size);
  }

(* The program is built at compile so that a graph outside the symbolic form
   is refused when the body is prepared. *)
module Engine (X : sig
  type op

  val name : string
  val program : op Graph_common.Graph.t -> Stage_program.t
end) =
struct
  type op = X.op
  type compiled = Stage_program.t

  let compile g =
    try Ok (X.program g)
    with Err.Exn.E e ->
      Error (Format.asprintf "%s: %a" X.name Err.Exn.pp_kind e)

  let execute (program : Stage_program.t) (g : op Graph_common.Graph.t) ~inputs
      =
    let rec missing = function
      | [] -> None
      | (id, _) :: rest ->
          if List.exists (fun (i, _) -> Tid.equal i id) inputs then missing rest
          else Some id
    in
    match missing program.inputs with
    | Some id ->
        Error
          (Printf.sprintf "%s: no value for input t%d" X.name (Tid.to_int id))
    | None -> (
        let bind id = snd (List.find (fun (i, _) -> Tid.equal i id) inputs) in
        match Stage_program.ground program ~bind with
        | Error e ->
            Error
              (Format.asprintf "%s: %a" X.name Stage_program.pp_error
                 (Err.Error.kind e))
        | Ok env ->
            (* An output that is a graph input is not a stage. *)
            let value id =
              match Tid.Map.find_opt id env with
              | Some p -> Some p
              | None ->
                  Option.map snd
                    (List.find_opt (fun (i, _) -> Tid.equal i id) inputs)
            in
            let rec outs acc = function
              | [] -> Ok (List.rev acc)
              | id :: rest -> (
                  match value id with
                  | None -> Error "a declared output was not produced"
                  | Some packed -> outs (packed :: acc) rest)
            in
            outs [] g.outputs)

  let release _ = ()
end

module Native_engine = Engine (struct
  type op = Graph_ir.op

  let name = "Native symbolic"
  let program g = Eval_symbolic.run g
end)

module Native4d_engine = Engine (struct
  type op = Native4d.Op.t

  let name = "Native4D symbolic"
  let program g = Native4d.Eval_symbolic4.run g
end)

module Native = Graph_runner.Make (Native_engine)
module Native4d_symbolic = Graph_runner.Make (Native4d_engine)

let native_identity = identity ~engine:"native" ~route:"eval-symbolic"
let native4d_identity = identity ~engine:"native4d" ~route:"eval-symbolic4"

let create_native ?artifact () =
  Native.create ~identity:(fun () -> native_identity ?artifact ())

let create_native4d ?artifact () =
  Native4d_symbolic.create ~identity:(fun () -> native4d_identity ?artifact ())
